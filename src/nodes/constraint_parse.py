"""ConstraintParseNode — CMN-C1-021 pre_process スロット（自然言語 → 構造化スペック）.

released AgentCore SDK (v1.0.0) のノード契約:
- `FunctionNode` を継承し `execute(self, state) -> dict` のみ実装
- **変更したキーだけ**を返す（full state を返さない。node_history は reducer=operator.add のため
  full state コピー返却は履歴を二重化する）
- node_history は返さない（BaseNode.__call__ が 1 件追記する）
- S-1 は `required_trust_level` を `BaseNode.__call__` が自動強制（pre_process=entry 相当）

責務（docs/02_design.md §3, §9）: 入力検証（空 / 長さ上限 / credential 混入 = S-2）/ LLM で
自然言語 → spec 抽出（構造化 problem_spec が与えられていれば LLM をスキップ）/ 構造妥当性 +
問題サイズ上限チェック。入力は state["problem_spec"] / input_context["problem_spec"]（構造化）
または state["problem_description"] / state["user_input"]（自然言語）から取る。
"""

from __future__ import annotations

import json
import math
import re

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

from src.security import detect_credentials
from src.services.llm_client import LLMNotConfiguredError, generate_text, resolve_llm_client

# LLM 出力から JSON 本体を取り出す（```json ... ``` フェンス対応）
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_VALID_OPS = {"<=", ">=", "=="}
_VALID_TYPES = {"continuous", "integer", "binary"}
_VALID_SENSE = {"minimize", "maximize"}


def _is_number(x) -> bool:
    """bool を除く**有限**の数値（int/float）か。

    LLM の "infinity" 等の文字列を弾くだけでは足りない: `json.loads()` は既定で
    `NaN` / `Infinity` / `-Infinity` を float として受理するため、それが素通りすると
    PuLP へ NaN 係数が渡る。有限性まで検査する。
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


_PROMPT_TEMPLATE = """\
あなたは数理最適化の定式化エンジンです。次の問題を**線形**最適化スペックの JSON に変換してください。
出力は JSON のみ。説明文やコードは一切含めないこと。

JSON 契約:
{{
  "sense": "minimize" | "maximize",
  "variables": [{{"name": str, "type": "continuous"|"integer"|"binary", "low": number|null, "up": number|null}}],
  "objective": {{"<変数名>": <係数>, ...}},
  "constraints": [{{"coeffs": {{"<変数名>": <係数>}}, "op": "<="|">="|"==", "rhs": number}}]
}}

問題:
{problem}
"""


class ConstraintParseNode(FunctionNode):
    # S-1: pre_process(entry 相当) が agent の trust gate を BaseNode.__call__ で強制（ADR-006）。
    #   caller_trust_level 不足 → execute() に到達せず status=error で停止。
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        llm_client: object | None = None,
        max_input_chars: int = 10000,
        max_variables: int = 200,
        max_constraints: int = 500,
    ) -> None:
        # 注入済みクライアント（graph が config["llm"] から解決したもの）を保持する。
        # 生成パラメータ（model / temperature / max_tokens）は注入側の持ち物なので
        # テンプレは保持しない（モデル ID をテンプレに書かない）。
        # ここでも解決しておく（graph は解決済みを渡すが、ノードを直接構築する
        # 呼び出し元が生の BaseLLM を渡すと `generate` が無く**黙って未注入扱い**に
        # なるため）。解決は冪等。
        self._llm = resolve_llm_client(llm_client)
        self._max_input_chars = int(max_input_chars)
        self._max_variables = int(max_variables)
        self._max_constraints = int(max_constraints)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _client(self):
        return self._llm

    @staticmethod
    def _extract_json(text: str) -> dict:
        m = _FENCE.search(text)
        payload = m.group(1) if m else text
        return json.loads(payload)

    def _validate_spec(self, spec: dict) -> tuple[bool, str]:
        """構造妥当性。OK なら (True, "")、NG なら (False, 理由)。"""
        if not isinstance(spec, dict):
            return False, "spec is not an object"
        if str(spec.get("sense", "")).lower() not in _VALID_SENSE:
            return False, f"invalid sense: {spec.get('sense')!r}"
        variables = spec.get("variables")
        if not isinstance(variables, list) or not variables:
            return False, "variables must be a non-empty list"
        names = set()
        for v in variables:
            if not isinstance(v, dict) or not isinstance(v.get("name"), str) or not v["name"]:
                return False, "variable missing valid name"
            if str(v.get("type", "continuous")).lower() not in _VALID_TYPES:
                return False, f"invalid variable type: {v.get('type')!r}"
            for bound_key in ("low", "up"):
                bv = v.get(bound_key)
                if bv is not None and not _is_number(bv):
                    return False, f"variable {v['name']!r} {bound_key} must be a number or null"
            if v["name"] in names:
                # 重複名は set に畳まれ、solver 側も後勝ちで 1 変数になる。
                # variable_count が実態とずれるので spec 段で弾く。
                return False, f"duplicate variable name: {v['name']!r}"
            names.add(v["name"])
        objective = spec.get("objective")
        if not isinstance(objective, dict) or not objective:
            return False, "objective must be a non-empty object"
        for var_name, coeff in objective.items():
            if var_name not in names:
                return False, f"objective references unknown variable {var_name!r}"
            if not _is_number(coeff):
                return False, f"objective coefficient for {var_name!r} must be a number"
        constraints = spec.get("constraints")
        if not isinstance(constraints, list):
            return False, "constraints must be a list"
        for i, c in enumerate(constraints):
            if not isinstance(c, dict):
                return False, f"constraint[{i}] is not an object"
            if c.get("op") not in _VALID_OPS:
                return False, f"constraint[{i}] invalid op: {c.get('op')!r}"
            if not _is_number(c.get("rhs")):
                return False, f"constraint[{i}] rhs must be a number"
            coeffs = c.get("coeffs")
            if not isinstance(coeffs, dict) or not coeffs:
                return False, f"constraint[{i}] coeffs must be a non-empty object"
            for var_name, coeff in coeffs.items():
                if var_name not in names:
                    return False, f"constraint[{i}] references unknown variable {var_name!r}"
                if not _is_number(coeff):
                    return False, f"constraint[{i}] coefficient for {var_name!r} must be a number"
        return True, ""

    @staticmethod
    def _fail(code: str, message: str) -> dict:
        return {
            "error": message,
            "error_code": code,
            "error_message": message,
            "spec_valid": False,
            "status": AgentStatus.ERROR.value,
        }

    # ── execute ──────────────────────────────────────────────────────────────
    def execute(self, state: dict) -> dict:
        # 上流のエラー/終了ステータスはパススルー（pre_process→main は無条件辺だが
        # 本ノードは entry なので通常は上流なし。直接呼び出しの堅牢性のため残す）。
        if state.get("error") or state.get("status") == AgentStatus.ERROR.value:
            return {"error": state.get("error"), "status": AgentStatus.ERROR.value}

        # 呼び出し元が構造化スペックを直接渡したか？（state 直接 or input_context 経由）
        # input_context は呼び出し元由来で dict とは限らない。非 dict で `.get` すると
        # AttributeError → framework の generic handler が **絶対パス入りの traceback** を
        # error_log に載せ、それが get_output() で呼び出し元まで返る。
        input_context = state.get("input_context")
        if not isinstance(input_context, dict):
            input_context = {}
        provided = state.get("problem_spec") or input_context.get("problem_spec")
        spec = None
        llm_used = False          # 構造化 spec 直接投入なら LLM は一度も呼ばれない
        model_name = ""           # 監査に載せるのは応答が名乗ったモデル名（config 値ではない）
        if provided:
            try:
                spec = provided if isinstance(provided, dict) else json.loads(provided)
            except (ValueError, TypeError):
                return self._fail("SPEC_PARSE_FAILED", "provided problem_spec is not valid JSON")
            # 直接 spec 投入経路も S-2（credential scan）を必ず通す（多層防御）
            if detect_credentials(json.dumps(spec, ensure_ascii=False, default=str)):
                return self._fail("S2_INJECTION_DETECTED", "credential pattern in provided problem_spec")
        else:
            description = (state.get("problem_description") or state.get("user_input") or "").strip()
            if not description:
                return self._fail("EMPTY_INPUT", "problem_description is empty")
            if len(description) > self._max_input_chars:
                return self._fail("PROBLEM_TOO_LARGE", f"input exceeds {self._max_input_chars} chars")
            if detect_credentials(description):
                return self._fail("S2_INJECTION_DETECTED", "credential pattern in input")
            # 自然言語 → spec の定式化には決定論代替が無い（自由文から線形計画の係数を
            # 導く規則は書けない）。したがって LLM 未設定・呼び出し失敗はどちらも縮退では
            # なく失敗として返す。理由は**例外の型名のみ**を載せる（str(exc) は依頼文の
            # 反射や資格情報を含みうるので State に入れない）。
            try:
                raw, model_name = generate_text(
                    self._client(), _PROMPT_TEMPLATE.format(problem=description)
                )
            except LLMNotConfiguredError:
                return self._fail(
                    "LLM_NOT_CONFIGURED",
                    "no LLM is configured for natural-language formulation",
                )
            except Exception as exc:  # noqa: BLE001 — provider 例外は型名だけ残す
                return self._fail("LLM_CALL_FAILED", f"LLM call failed: {type(exc).__name__}")
            if not raw.strip():
                # 空応答を「spec が取れた」と扱わない。
                return self._fail("SPEC_PARSE_FAILED", "LLM returned an empty response")
            try:
                spec = self._extract_json(raw)
            except (ValueError, TypeError) as exc:
                return self._fail(
                    "SPEC_PARSE_FAILED",
                    f"LLM output is not valid spec JSON: {type(exc).__name__}",
                )
            llm_used = True

        ok, reason = self._validate_spec(spec)
        if not ok:
            return self._fail("SPEC_PARSE_FAILED", f"spec invalid: {reason}")

        var_count = len(spec.get("variables", []))
        con_count = len(spec.get("constraints", []))
        if var_count > self._max_variables or con_count > self._max_constraints:
            return self._fail(
                "PROBLEM_TOO_LARGE",
                f"problem too large: {var_count} vars / {con_count} constraints "
                f"(max {self._max_variables}/{self._max_constraints})",
            )

        # S-4: at least one domain event per node — framework の監査契約
        emit_trace_event(
            "constraint_parse_complete",
            {
                "variable_count": var_count,
                "constraint_count": con_count,
                # 構造化 spec 経路と自然言語経路を監査で区別できるようにする。
                "llm_used": llm_used,
                "model": model_name or None,
            },
            state,
        )

        return {
            "problem_spec": json.dumps(spec, ensure_ascii=False),
            "spec_valid": True,
            "variable_count": var_count,
            "constraint_count": con_count,
            "status": AgentStatus.SUCCESS.value,
        }
