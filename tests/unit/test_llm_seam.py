"""LLM 注入シームの回帰テスト — `config["llm"]` が本番経路で実際にノードへ届くこと。

旧実装は外側 Graph の `llm_client=` という独自コンストラクタ引数でしか LLM を受け取れず、
framework の実契約（`BaseGraph.__init__(self, config=None)`）に従う `src/api/server.py` は
それを渡さないため **既定ビルドでは自然言語経路が必ず失敗**していた。テストが緑だったのは
conftest が `agent._llm_client` を private 属性で直接書き換えていたためで、本番経路は一度も
通っていない。ここでは **`config["llm"]` だけ**を使い、private 属性には触らない。

期待値はテストにリテラルで書く（production の定数・関数から計算しない）。
"""

from __future__ import annotations

import inspect
import json
import logging

import pytest

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from shared.services.llm.base_llm import BaseLLM

from src.agent import OptimizationSolverAgent
from src.graph.graph import OptimizationSolverGraph
from src.nodes.constraint_parse import ConstraintParseNode
from src.nodes.solution_format import SolutionFormatNode
from src.services.llm_client import (
    BaseLLMAdapter,
    LLMClient,
    LLMNotConfiguredError,
    generate_text,
    resolve_llm_client,
)

# LLM が返すべき spec。x1+x2 を最大化 / x1<=4 / x2<=3 → 最適値 7.0（手計算）。
SPEC_JSON = json.dumps(
    {
        "sense": "maximize",
        "variables": [
            {"name": "x1", "type": "continuous", "low": 0},
            {"name": "x2", "type": "continuous", "low": 0},
        ],
        "objective": {"x1": 1.0, "x2": 1.0},
        "constraints": [
            {"coeffs": {"x1": 1.0}, "op": "<=", "rhs": 4.0},
            {"coeffs": {"x2": 1.0}, "op": "<=", "rhs": 3.0},
        ],
    }
)

NL_PROBLEM = "x1 と x2 の合計を最大化。x1 は 4 以下、x2 は 3 以下、どちらも 0 以上。"


class _FakeBaseLLM(BaseLLM):
    """`BaseLLM` の canonical 契約だけを実装した double。"""

    def __init__(self, *contents: str, model: str = "fake-model-1") -> None:
        self.calls: list = []
        self._contents = list(contents) or [""]
        self._model = model

    def complete(self, messages: list) -> dict:
        self.calls.append(messages)
        content = self._contents[min(len(self.calls) - 1, len(self._contents) - 1)]
        return {
            "content": content,
            "tool_calls": [],
            "model": self._model,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "model": self._model},
        }

    def stream(self, messages: list):
        yield self.complete(messages)["content"]

    def bind_tools(self, tools: list) -> "_FakeBaseLLM":
        return self


class _DelegatingBaseLLM(_FakeBaseLLM):
    """SDK 実クライアント（AnthropicClient 等）と同じ `__getattr__` 委譲を持つ double。

    `getattr(x, "generate")` は**バッチ用の別シグネチャ**を拾う。duck typing を
    `isinstance(BaseLLM)` より先に置くと、文字列 prompt を渡した瞬間に壊れる。
    """

    def __getattr__(self, name: str):
        if name == "generate":
            def _batch_generate(prompts, **kwargs):  # LangChain の batch API 相当
                raise AssertionError(
                    "delegated batch generate() was called with a single string prompt"
                )

            return _batch_generate
        raise AttributeError(name)


def _invoke(agent, user_input=NL_PROBLEM, input_context=None):
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, session_id="s")
    return agent.invoke(
        user_input=user_input, session_id="s", ctx=ctx, input_context=input_context or {}
    )


# ── 本番配線 ────────────────────────────────────────────────────────────────

def test_graph_constructor_takes_config_only():
    """独自コンストラクタ引数（`llm_client=` 等）を復活させない。"""
    params = list(inspect.signature(OptimizationSolverGraph.__init__).parameters)
    assert params == ["self", "config"]


def test_config_llm_reaches_the_parse_node_end_to_end():
    """`config["llm"]` だけで自然言語 → 定式化 → 求解が通ること（本番経路）。"""
    llm = _FakeBaseLLM(SPEC_JSON, "利益は最大 7 万円です。")
    out = _invoke(OptimizationSolverAgent(config={"llm": llm}))

    assert llm.calls, "config['llm'] がノードへ届いていない（LLM が一度も呼ばれていない）"
    assert out["status"] == "success"
    assert out["solver_status"] == "OPTIMAL"
    assert out["objective_value"] == pytest.approx(7.0)
    assert out["variable_count"] == 2
    assert "利益は最大 7 万円です。" in out["formatted_output"]


def test_natural_language_path_fails_without_llm():
    """LLM 未注入の既定ビルドでは自然言語経路は失敗する（決定論代替が無いため）。"""
    out = _invoke(OptimizationSolverAgent(config={}))
    assert out["status"] == "error"
    assert out["error_code"] == "LLM_NOT_CONFIGURED"
    # 型付きエラーであること = framework の generic node_error に落ちていない
    assert out.get("solver_status") is None


def test_structured_spec_path_never_calls_the_llm():
    """構造化 problem_spec 経路は LLM を一度も呼ばない（定式化が不要）。"""
    llm = _FakeBaseLLM("SHOULD NOT BE USED FOR PARSING")
    out = _invoke(
        OptimizationSolverAgent(config={"llm": llm}),
        user_input="",
        input_context={"problem_spec": SPEC_JSON},
    )
    assert out["status"] == "success"
    assert out["objective_value"] == pytest.approx(7.0)
    # 説明文の 1 回だけ（定式化では呼ばれない）
    assert len(llm.calls) == 1


def test_get_output_carries_output_key():
    """AGENTIC STAR runner は success 時に `output` を必須とする（無いと RuntimeError）。"""
    llm = _FakeBaseLLM(SPEC_JSON, "説明文")
    out = _invoke(OptimizationSolverAgent(config={"llm": llm}))
    assert out["status"] == "success"
    assert "output" in out
    assert out["output"] == out["formatted_output"]
    assert out["output"]


# ── resolve_llm_client（誤判定の防止）──────────────────────────────────────

def test_base_llm_is_wrapped_in_the_adapter():
    resolved = resolve_llm_client(_FakeBaseLLM("x"))
    assert isinstance(resolved, BaseLLMAdapter)


def test_isinstance_wins_over_delegated_generate():
    """`__getattr__` 委譲の `generate` を自前実装と誤判定しない。"""
    llm = _DelegatingBaseLLM(SPEC_JSON)
    resolved = resolve_llm_client(llm)
    assert isinstance(resolved, BaseLLMAdapter)
    text, model = generate_text(resolved, "prompt")   # 誤判定なら AssertionError
    assert text == SPEC_JSON
    assert model == "fake-model-1"


def test_own_generate_implementation_is_used_as_is():
    class _Own:
        def generate(self, prompt: str) -> str:
            return "own"

    own = _Own()
    assert resolve_llm_client(own) is own


def test_missing_llm_resolves_to_the_sentinel():
    assert isinstance(resolve_llm_client(None), LLMClient)
    with pytest.raises(LLMNotConfiguredError):
        resolve_llm_client(None).generate("x")


def test_none_content_is_normalised_to_empty_string():
    """`{"content": None}` を str() すると文字列 "None" が本文として返ってしまう。"""

    class _NullContent(_FakeBaseLLM):
        def complete(self, messages: list) -> dict:
            self.calls.append(messages)
            return {"content": None, "tool_calls": [], "model": "fake-model-1", "usage": {}}

    text, _ = generate_text(resolve_llm_client(_NullContent()), "prompt")
    assert text == ""


# ── 失敗の扱い ──────────────────────────────────────────────────────────────

def test_empty_llm_response_is_not_success():
    """空応答を「spec が取れた」と扱わない。"""
    out = _invoke(OptimizationSolverAgent(config={"llm": _FakeBaseLLM("   ")}))
    assert out["status"] == "error"
    assert out["error_code"] == "SPEC_PARSE_FAILED"


def test_provider_exception_message_is_not_persisted():
    """provider 例外の str(exc) は依頼文の反射や資格情報を含みうるので State に入れない。"""

    class _Boom:
        def generate(self, prompt: str) -> str:
            raise RuntimeError("upstream said: sk-" + "z" * 30)

    node = ConstraintParseNode(llm_client=_Boom())
    out = node.execute({"problem_description": NL_PROBLEM})
    assert out["error_code"] == "LLM_CALL_FAILED"
    blob = json.dumps(out, ensure_ascii=False)
    assert "sk-" not in blob
    assert "upstream said" not in blob
    assert "RuntimeError" in blob   # 型名だけは残す


def test_format_stage_degrades_to_deterministic_summary():
    """整形段は決定論サマリを持つので、LLM 未設定でも success を維持する。"""
    out = SolutionFormatNode().execute(
        {"solver_status": "OPTIMAL", "solution": json.dumps({"x1": 4.0}), "objective_value": 7.0}
    )
    assert out["status"] == "success"
    assert "OPTIMAL" in out["formatted_output"]
    assert "7.0" in out["formatted_output"]


# ── 監査 ────────────────────────────────────────────────────────────────────

def test_audit_reports_the_model_from_the_response(caplog):
    """監査のモデル名は canonical response の `model` から採る（config の文字列ではない）。"""
    llm = _FakeBaseLLM(SPEC_JSON, "説明文", model="responded-model-9")
    with caplog.at_level(logging.INFO, logger="agentcore.audit"):
        out = _invoke(
            OptimizationSolverAgent(config={"llm": llm, "model": "config-model-not-used"})
        )
    assert out["status"] == "success"
    msgs = [r.getMessage() for r in caplog.records if r.name == "agentcore.audit"]
    done = [m for m in msgs if '"event_type": "optimization_complete"' in m]
    assert len(done) == 1
    assert "responded-model-9" in done[0]
    assert "config-model-not-used" not in done[0]


def test_audit_marks_the_structured_path_as_llm_unused(caplog):
    llm = _FakeBaseLLM("説明文")
    with caplog.at_level(logging.INFO, logger="agentcore.audit"):
        _invoke(
            OptimizationSolverAgent(config={"llm": llm}),
            user_input="",
            input_context={"problem_spec": SPEC_JSON},
        )
    parse = [
        r.getMessage()
        for r in caplog.records
        if r.name == "agentcore.audit" and '"event_type": "constraint_parse_complete"' in r.getMessage()
    ]
    assert len(parse) == 1
    assert '"llm_used": false' in parse[0]


# ── レビュー H1: 構築後・compile 前の config 注入（旧 runner のライフサイクル）──

def test_llm_injected_after_construction_still_reaches_the_node():
    """AGENTIC STAR の runner には `agent_cls()` で構築してから config を埋める版が実在する。

    `__init__` で `config["llm"]` を固めると、その版では LLM が永久に未注入になる。
    設定の読み取りは register_nodes()（= compile 時）まで遅らせる。
    """
    agent = OptimizationSolverAgent()          # config なしで構築
    llm = _FakeBaseLLM(SPEC_JSON, "説明文")
    agent.config["llm"] = llm                  # 構築後に注入
    agent.compile()
    assert type(agent._nodes["pre_process"]._llm).__name__ == "BaseLLMAdapter"
    out = _invoke(agent)
    assert llm.calls
    assert out["status"] == "success"
    assert out["objective_value"] == pytest.approx(7.0)


def test_runtime_limits_are_read_at_compile_time_too():
    agent = OptimizationSolverAgent()
    agent.config["max_variables"] = 1
    agent.compile()
    assert agent._nodes["pre_process"]._max_variables == 1


# ── レビュー M1: resolve_llm_client の誤判定パターン ────────────────────

def test_property_that_raises_does_not_break_construction():
    """`getattr()` は property を実行してしまう。判定だけで構築が落ちてはいけない。"""

    class _PropBoom:
        @property
        def complete(self):
            raise RuntimeError("boom")

        def generate(self, prompt: str) -> str:
            return "ok"

    resolved = resolve_llm_client(_PropBoom())    # ここで例外が出たら不合格
    assert resolved is not None


def test_instance_attribute_generate_is_recognised():
    """`self.generate = ...` で組み立てるクライアントを未注入と誤判定しない。"""

    class _InstGen:
        def __init__(self) -> None:
            self.generate = lambda prompt: "inst"

    resolved = resolve_llm_client(_InstGen())
    assert not isinstance(resolved, LLMClient)
    assert generate_text(resolved, "p")[0] == "inst"


def test_canonical_complete_wins_over_batch_generate():
    """`generate` がバッチ API の非 BaseLLM 型では canonical な `complete` を使う。"""

    class _Both:
        def generate(self, prompts, **kwargs):     # LangChain の batch API 相当
            raise AssertionError("batch generate() was selected")

        def complete(self, messages: list) -> dict:
            return {"content": "canonical", "model": "m1"}

    resolved = resolve_llm_client(_Both())
    assert isinstance(resolved, BaseLLMAdapter)
    assert generate_text(resolved, "p") == ("canonical", "m1")


def test_complete_only_non_base_llm_is_wrapped():
    """`BaseLLM` を継承しない canonical クライアントも adapter で受ける。"""

    class _CompleteOnly:
        def complete(self, messages: list) -> dict:
            return {"content": "c", "model": "m2"}

    assert isinstance(resolve_llm_client(_CompleteOnly()), BaseLLMAdapter)


# ── レビュー: 変異検査の盲点（プロンプト内容・role）────────────────────

def test_formulation_prompt_carries_the_problem_text():
    """プロンプトから問題文が落ちても fake は気づかない（実 LLM では致命的）。"""
    llm = _FakeBaseLLM(SPEC_JSON, "説明文")
    _invoke(OptimizationSolverAgent(config={"llm": llm}))
    first = json.dumps(llm.calls[0], ensure_ascii=False)
    assert NL_PROBLEM in first


def test_explanation_prompt_carries_the_solver_result():
    """説明段のプロンプトから求解結果が落ちると、LLM が数値を作文する余地が増える。"""
    llm = _FakeBaseLLM(SPEC_JSON, "説明文")
    _invoke(OptimizationSolverAgent(config={"llm": llm}))
    second = json.dumps(llm.calls[1], ensure_ascii=False)
    assert "OPTIMAL" in second
    assert "7.0" in second


def test_adapter_sends_a_user_role_message():
    llm = _FakeBaseLLM("x")
    generate_text(resolve_llm_client(llm), "hello")
    assert llm.calls[0] == [{"role": "user", "content": "hello"}]


def test_parse_audit_reports_the_model_from_the_response(caplog):
    llm = _FakeBaseLLM(SPEC_JSON, "説明文", model="responded-model-9")
    with caplog.at_level(logging.INFO, logger="agentcore.audit"):
        _invoke(OptimizationSolverAgent(config={"llm": llm}))
    parse = [r.getMessage() for r in caplog.records
             if r.name == "agentcore.audit"
             and '"event_type": "constraint_parse_complete"' in r.getMessage()]
    assert len(parse) == 1
    assert "responded-model-9" in parse[0]
    assert '"llm_used": true' in parse[0]


# ── レビュー M3: input_context が dict でない ──────────────────────────

def test_non_dict_input_context_does_not_leak_a_traceback():
    """非 dict の input_context で `.get` が落ちると、絶対パス入り traceback が返る。"""
    out = _invoke(OptimizationSolverAgent(config={}), input_context=["not-a-dict"])
    assert out["status"] == "error"
    blob = json.dumps(out, ensure_ascii=False)
    assert "Traceback" not in blob
    assert "site-packages" not in blob


# ── レビュー H4(一部): S3_BLOCKED は求解ペイロードも返さない ────────────

def test_s3_block_clears_the_solver_payload():
    out = SolutionFormatNode(llm_client=_FakeBaseLLM("api_key=supersecretvalue123")).execute(
        {"solver_status": "OPTIMAL", "solution": json.dumps({"x1": 4.0}), "objective_value": 7.0}
    )
    assert out["error_code"] == "S3_BLOCKED"
    assert out["formatted_output"] is None
    assert out["solution"] is None
    assert out["objective_value"] is None
