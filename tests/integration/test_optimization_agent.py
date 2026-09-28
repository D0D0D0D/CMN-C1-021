"""Integration tests — OptimizationSolverAgent E2E（released AgentCore SDK v1.0.0）.

構造化 problem_spec を input_context 経由で与え、LLM に依存せず full backbone
（initialize → pre_process(parse 検証) → main(実 CBC 求解) → post_process(整形 + S-3) →
finalize）を決定論的に検証する。

S-1（ADR-006）: 本番エントリは `invoke(user_input, session_id, ctx, input_context)`。
`BaseNode.__call__()` が entry ノードの required_trust_level(VERIFIED_EXTERNAL) を
`caller_trust_level`（ctx 由来）と照合し、不足時は status=error で停止する（例外ではない）。
"""

import json
import logging

import pytest

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.agent import OptimizationSolverAgent

CONFIG = {"model": "claude-sonnet-4-6", "temperature": 0.0, "max_tokens": 256,
          "solver_timeout_s": 10}

SPEC = {
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

INFEASIBLE_SPEC = {
    "sense": "maximize",
    "variables": [{"name": "x", "type": "continuous", "low": 0}],
    "objective": {"x": 1.0},
    "constraints": [
        {"coeffs": {"x": 1.0}, "op": ">=", "rhs": 5.0},
        {"coeffs": {"x": 1.0}, "op": "<=", "rhs": 4.0},
    ],
}


def _run(agent, spec=SPEC, level=TrustLevel.VERIFIED_EXTERNAL, session_id="s", with_ctx=True):
    """released invoke 署名で起動。構造化 spec は input_context 経由で渡す。"""
    ctx = InvocationContext(caller_trust_level=level, session_id=session_id) if with_ctx else None
    return agent.invoke(
        user_input="",
        session_id=session_id,
        ctx=ctx,
        input_context={"problem_spec": json.dumps(spec)},
    )


def test_invoke_solves_known_lp():
    out = _run(OptimizationSolverAgent(config=CONFIG))
    assert out.get("error") is None
    assert out["solver_status"] == "OPTIMAL"
    assert out["objective_value"] == pytest.approx(7.0)
    assert out["formatted_output"] and "OPTIMAL" in out["formatted_output"]
    assert out["status"] == "success"


def test_invoke_infeasible_is_explained():
    out = _run(OptimizationSolverAgent(config=CONFIG), spec=INFEASIBLE_SPEC)
    assert out.get("error") is None
    assert out["solver_status"] == "INFEASIBLE"
    assert "INFEASIBLE" in out["formatted_output"]


def test_invoke_lifecycle_trace_emitted(caplog):
    # S-4: post_process が emit_trace_event でドメインイベント optimization_complete を 1 件出す。
    #   v1.0.0 では backbone/各ノードの __call__ が node_start/node_complete を同じ
    #   agentcore.audit ロガーへ自動 emit するため、総件数ではなくドメインイベントを抽出して検証する。
    with caplog.at_level(logging.INFO, logger="agentcore.audit"):
        _run(OptimizationSolverAgent(config=CONFIG))
    domain = [r for r in caplog.records
              if r.name == "agentcore.audit" and '"event_type": "optimization_complete"' in r.getMessage()]
    assert len(domain) == 1
    msg = domain[0].getMessage()
    assert "optimization_complete" in msg
    assert "OPTIMAL" in msg


def test_s1_absent_context_default_deny():
    # ctx 不在 → caller_trust_level=ANONYMOUS → entry ノードが status=error で停止
    out = OptimizationSolverAgent(config=CONFIG).invoke(
        user_input="", input_context={"problem_spec": json.dumps(SPEC)})
    assert out["status"] == "error"
    assert out.get("solver_status") is None
    assert any("trust gate denied" in e.lower() for e in out["error_log"])


def test_s1_insufficient_trust_rejected():
    out = _run(OptimizationSolverAgent(config=CONFIG), level=TrustLevel.ANONYMOUS)
    assert out["status"] == "error"
    assert out.get("solver_status") is None
    assert any("trust gate denied" in e.lower() for e in out["error_log"])


def test_s1_internal_trust_allowed():
    out = _run(OptimizationSolverAgent(config=CONFIG), level=TrustLevel.INTERNAL)
    assert out.get("error") is None
    assert out["solver_status"] == "OPTIMAL"


def test_s2_scans_problem_spec_direct_input():
    # S-2 ゲートは problem_spec 直接入力も走査する（invoke 経路）
    bad = json.loads(json.dumps(SPEC))
    bad["variables"][0]["name"] = "v_api_key=supersecretvalue123"
    out = _run(OptimizationSolverAgent(config=CONFIG), spec=bad)
    assert out["error_code"] == "S2_INJECTION_DETECTED"
    assert out.get("solver_status") is None


def test_independent_runs_do_not_share_state():
    agent = OptimizationSolverAgent(config=CONFIG)
    out1 = _run(agent, session_id="a")
    out2 = _run(agent, spec=INFEASIBLE_SPEC, session_id="b")
    assert out1["solver_status"] == "OPTIMAL"
    assert out2["solver_status"] == "INFEASIBLE"
