"""SolutionFormatNode — CMN-C1-021 post_process スロット（自然言語化 + S-3 + S-4）.

released AgentCore SDK 契約: FunctionNode を継承し execute(self, state) で
**部分 dict** を返す。post_process は route() が status==SUCCESS のときのみ到達する。

責務（docs/02_design.md §3, §8）:
- 求解結果（solver_status / solution / objective_value）を自然言語化（決定論サマリ + LLM 補助）
- S-3 出力ゲート: PII マスク（apply_masks）→ credential 残存チェック → S3_BLOCKED
- S-4 監査トレース: emit_trace_event でメタのみ記録（本文は記録しない）

🔴 **出力の並びは「決定論サマリが先頭・LLM の説明文がその後」で固定する。**
説明文は LLM が書く散文なので、ソルバが出した数値を再述・改竄しうる（レビューで
「最適値は 999」と書く LLM を刺すと、正しい 7.0 より先に読まれることを実測した）。
先に読まれるものを決定論的な事実にし、説明文は節見出しで「参考」と分けて、
食い違ったときにどちらが正かを読み手が判別できるようにする。
説明文を捨てはしない（読み手の理解を助ける価値があるため）。
"""

from __future__ import annotations

import json

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

from src.masking import apply_masks
from src.security import detect_credentials
from src.services.llm_client import generate_text, resolve_llm_client

_TEMPLATE_ID = "CMN-C1-021"

# LLM の説明文を包む節。決定論サマリと**出力上で区別できる**ようにするためのもので、
# 「どちらが正か」を読み手に明示する（順序だけでは区別が伝わらない）。
_LLM_SECTION_HEADING = "## 参考: 言語モデルによる説明"
_LLM_SECTION_NOTE = (
    "> 上の「求解結果」がソルバの決定論的な出力です。以下は言語モデルが生成した補足で、"
    "数値が食い違う場合は上の求解結果が正です。"
)

_STATUS_JA = {
    "OPTIMAL": "最適解が見つかりました",
    "INFEASIBLE": "制約をすべて満たす解は存在しません（実行不能）",
    "UNBOUNDED": "目的関数が非有界です（解が無限に改善可能）",
    "TIMEOUT": "制限時間内に最適性を確認できませんでした（暫定解を提示）",
    "UNDEFINED": "求解結果を確定できませんでした",
}


class SolutionFormatNode(FunctionNode):
    # S-1 多層防御: 後段ノードも明示宣言（ADR-006）
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        llm_client: object | None = None,
        mask_patterns: list | None = None,
    ) -> None:
        # 注入済みクライアント（graph が config["llm"] から解決したもの）。生成
        # パラメータは注入側の持ち物なのでテンプレは保持しない。
        # ここでも解決しておく（graph は解決済みを渡すが、ノードを直接構築する
        # 呼び出し元が生の BaseLLM を渡すと `generate` が無く**黙って未注入扱い**に
        # なるため）。解決は冪等。
        self._llm = resolve_llm_client(llm_client)
        self._mask_patterns = list(mask_patterns or [])

    def _client(self):
        return self._llm

    def _deterministic_summary(self, status: str, solution: dict, objective_value) -> str:
        lines = [f"## 求解結果: {status}", _STATUS_JA.get(status, status)]
        if objective_value is not None:
            lines.append(f"- 目的関数値: {objective_value}")
        if solution:
            lines.append("- 変数割当:")
            for name, value in solution.items():
                lines.append(f"    - {name} = {value}")
        return "\n".join(lines)

    def execute(self, state: dict) -> dict:
        if state.get("error") or state.get("status") == AgentStatus.ERROR.value:
            return {"error": state.get("error"), "status": AgentStatus.ERROR.value}

        status = state.get("solver_status", "UNDEFINED")
        try:
            solution = json.loads(state.get("solution") or "{}")
        except (ValueError, TypeError):
            solution = {}
        objective_value = state.get("objective_value")

        summary = self._deterministic_summary(status, solution, objective_value)

        # LLM による補助説明。**この AG は決定論サマリを常に持つ**ので、LLM 未設定・
        # 呼び出し失敗・空応答はいずれも縮退（説明文なし）で success を維持する
        # （自然言語 → spec の定式化と違い、ここには決定論代替が存在する）。
        # 例外は握り潰すが、理由を State に載せることはしない（str(exc) は依頼文の
        # 反射や資格情報を含みうる）。
        explanation = ""
        llm_model = ""
        try:
            prompt = (
                "次の最適化求解結果を、ビジネス担当者向けに 2〜3 文で平易に説明してください。\n"
                f"status={status}, objective={objective_value}, solution={json.dumps(solution, ensure_ascii=False)}"
            )
            text, llm_model = generate_text(self._client(), prompt)
            explanation = text.strip()
        except Exception:  # noqa: BLE001 — 説明文は補助。失敗しても決定論サマリを返す
            explanation = ""
            llm_model = ""

        # 決定論サマリが先頭。説明文があるときだけ「参考」節として後ろに足す
        # （未注入・失敗・空応答では summary だけ = 従来の縮退挙動そのまま）。
        sections = [summary]
        if explanation:
            sections += [_LLM_SECTION_HEADING, _LLM_SECTION_NOTE, explanation]
        candidate = "\n\n".join(sections).strip()

        # ── S-3 出力ゲート（ノード内多層防御） ──
        masked, _applied = apply_masks(candidate, self._mask_patterns)
        if detect_credentials(masked):
            # 出力をブロックする以上、同じ内容を含みうる求解ペイロードも返さない
            # （`solution` は変数名を、`objective_value` は値をそのまま持つ）。
            return {
                "error": "credential pattern in formatted output",
                "error_code": "S3_BLOCKED",
                "error_message": "credential pattern in formatted output",
                "formatted_output": None,
                "solution": None,
                "objective_value": None,
                "status": AgentStatus.ERROR.value,
            }

        # ── S-4 監査トレース（メタのみ。problem_description / solution 本文は記録しない） ──
        emit_trace_event(
            "optimization_complete",
            {
                "template_id": _TEMPLATE_ID,
                "solver_status": status,
                "variable_count": state.get("variable_count"),
                "constraint_count": state.get("constraint_count"),
                # 監査に載せるモデル名は **canonical response の `model`** から採る。
                # config の model 文字列は注入クライアントの選択を決めないので、
                # そのまま報告すると監査ログが実際と食い違う。
                "llm_used": bool(explanation),
                "model": (llm_model or None) if explanation else None,
            },
            state,
        )

        return {"formatted_output": masked, "status": AgentStatus.SUCCESS.value}
