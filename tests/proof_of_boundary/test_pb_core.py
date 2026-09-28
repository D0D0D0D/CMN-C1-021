"""CMN-C1-021 — Proof-of-Boundary Tests (released AgentCore SDK v1.0.0).

docs/02_design.md §12 / §8。フレームワーク境界・セキュリティ境界の検証。
released wheel の固定 5 ノード backbone（initialize→pre_process→main→route→post_process→
finalize）と node 契約に整合させて移行（旧 _invoke_impl / graph レベル gate フックは廃止）。
"""

import ast
import inspect
import json
import logging
import os

import ormsgpack
import pytest

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.agent import OptimizationSolverAgent
from src.nodes.constraint_parse import ConstraintParseNode
from src.nodes.solver_dispatch import SolverDispatchNode
from src.nodes.solution_format import SolutionFormatNode

CONFIG = {"model": "claude-sonnet-4-6", "temperature": 0.0, "max_tokens": 256,
          "solver_timeout_s": 10}
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

SPEC = {
    "sense": "maximize",
    "variables": [{"name": "x1", "type": "continuous", "low": 0},
                  {"name": "x2", "type": "continuous", "low": 0}],
    "objective": {"x1": 1.0, "x2": 1.0},
    "constraints": [{"coeffs": {"x1": 1.0}, "op": "<=", "rhs": 4.0},
                    {"coeffs": {"x2": 1.0}, "op": "<=", "rhs": 3.0}],
}

_VALID_TRUST = {TrustLevel.VERIFIED_EXTERNAL, TrustLevel.INTERNAL}


def _run(agent, level=TrustLevel.VERIFIED_EXTERNAL):
    ctx = InvocationContext(caller_trust_level=level, session_id="s")
    return agent.invoke(user_input="", session_id="s", ctx=ctx,
                        input_context={"problem_spec": json.dumps(SPEC)})


# ── PB-1: BaseNode → AuditLogger（S-4 trace 必発火）──
#   v1.0.0 では backbone(Initialize/Finalize) と各ノードの __call__ が node_start/node_complete を
#   同じ agentcore.audit ロガーへ自動 emit するため総件数は固定できない。ドメインイベント
#   （SolutionFormatNode.execute が emit する optimization_complete）が 1 件出ることを検証する。
def test_pb1_trace_event_fires(caplog):
    with caplog.at_level(logging.INFO, logger="agentcore.audit"):
        _run(OptimizationSolverAgent(config=CONFIG))
    domain = [r for r in caplog.records
              if r.name == "agentcore.audit" and '"event_type": "optimization_complete"' in r.getMessage()]
    assert len(domain) == 1


# ── PB-2: State serialization（出力は msgpack 安全 = checkpoint 安全）──
def test_pb2_output_is_msgpack_safe():
    out = _run(OptimizationSolverAgent(config=CONFIG))
    # Pydantic/dataclass/任意オブジェクトが混ざっていれば packb が失敗する
    ormsgpack.packb(out)
    for key, value in out.items():
        assert isinstance(value, (str, int, float, bool, type(None), list, dict)), (
            f"out['{key}'] is {type(value).__name__} — not msgpack-safe")


# ── PB-3: SolverDispatchNode → 外部ソルバ（実 CBC 求解境界）──
def test_pb3_solver_service_boundary_real_solve():
    out = SolverDispatchNode().execute({"problem_spec": json.dumps(SPEC)})
    assert out["solver_status"] == "OPTIMAL"
    assert out["solver_backend"] == "PULP_CBC_CMD"
    assert json.loads(out["solution"])["x1"] == pytest.approx(4.0)


# ── PB-4: Import isolation（src/ が Level 0 agenticstar を import しない）──
def test_pb4_import_isolation_no_agenticstar():
    violations = []
    for root, _, files in os.walk(_SRC):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            tree = ast.parse(open(path).read(), filename=path)
            for n in ast.walk(tree):
                mod = ""
                if isinstance(n, ast.Import):
                    mod = n.names[0].name
                elif isinstance(n, ast.ImportFrom):
                    mod = n.module or ""
                if mod.startswith("agenticstar"):
                    violations.append(f"{path}: {mod}")
    assert violations == [], "Level 0 import(s) found: " + "; ".join(violations)


# ── PB-5: Checkpoint safety（出力に invocation_context / credential なし）──
def test_pb5_checkpoint_safety():
    out = _run(OptimizationSolverAgent(config=CONFIG))
    assert "invocation_context" not in out
    blob = json.dumps(out, ensure_ascii=False, default=str)
    assert "sk-" not in blob


# ── PB-6: backbone 実行順序（initialize → pre → main → post → finalize）──
def test_pb6_node_execution_order():
    out = _run(OptimizationSolverAgent(config=CONFIG))
    assert out["node_history"] == [
        "InitializeNode",
        "ConstraintParseNode",
        "SolverDispatchNode",
        "SolutionFormatNode",
        "FinalizeNode",
    ]


# ── PB-7（本テンプレ固有）: 任意コード実行なし — solver は構造化データのみ受け付ける ──
def test_pb7_no_arbitrary_code_execution():
    malicious = {"sense": "minimize",
                 "variables": [{"name": "x", "type": "continuous", "low": 0}],
                 "objective": {"x": 1.0},
                 "constraints": [{"coeffs": {"x": "__import__('os').system('echo hacked')"},
                                  "op": ">=", "rhs": 1.0}]}
    out = SolverDispatchNode().execute({"problem_spec": json.dumps(malicious)})
    assert out.get("error_code") == "SOLVER_FAILED"
    assert out.get("solver_status") is None
    # コード文字列は eval されず float() 変換で ValueError になって止まる。
    # 理由に載るのは**例外の型名だけ**（例外メッセージは PuLP/CBC のサーバ側
    # 絶対パス等を含みうるので State に入れない）。
    assert "ValueError" in out.get("error_message", "")
    # 悪意ある入力そのものを呼び出し元へ反射しない
    assert "__import__" not in json.dumps(out, ensure_ascii=False)


# ── 契約: ノードは execute() のみ。_invoke_impl を実装しない ──
@pytest.mark.parametrize("cls", [ConstraintParseNode, SolverDispatchNode, SolutionFormatNode])
def test_node_contract_execute_only(cls):
    assert "execute" in cls.__dict__, f"{cls.__name__} must implement execute()"
    assert "_invoke_impl" not in cls.__dict__, (
        f"{cls.__name__} must NOT implement _invoke_impl() (node contract)")


# ── S-1 多層防御: 全ノードが required_trust_level を VERIFIED_EXTERNAL 以上で宣言（ADR-006）──
#   TrustLevel は文字列 enum（int 比較不可）のため有効値集合で判定する。
@pytest.mark.parametrize("cls", [ConstraintParseNode, SolverDispatchNode, SolutionFormatNode])
def test_all_nodes_declare_required_trust_level(cls):
    assert "required_trust_level" in cls.__dict__, (
        f"{cls.__name__} must explicitly declare required_trust_level (no implicit ANONYMOUS)")
    assert cls.required_trust_level in _VALID_TRUST, (
        f"{cls.__name__} required_trust_level must be VERIFIED_EXTERNAL or INTERNAL, "
        f"got {cls.required_trust_level}")


# ── execute() シグネチャ契約（v1.0.0）: execute(self, state) ──
#   v1.0.0 の BaseNode.__call__ は execute(state) を state 1 引数のみで呼ぶ。config は
#   渡されない厳密契約なので、余計な config 引数を持たないことを検証する。
@pytest.mark.parametrize("cls", [ConstraintParseNode, SolverDispatchNode, SolutionFormatNode])
def test_execute_method_signature(cls):
    sig = inspect.signature(cls.execute)
    params = list(sig.parameters)
    assert params == ["self", "state"], (
        f"{cls.__name__}.execute must be execute(self, state), got {params}")
