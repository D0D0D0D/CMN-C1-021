"""Unit tests — SolverService / SolverDispatchNode（実 CBC 求解・実行不能・整数・例外）."""

import json

import pytest

from src.services import solver_service
from src.nodes.solver_dispatch import SolverDispatchNode


def _spec(sense, variables, objective, constraints):
    return {"sense": sense, "variables": variables, "objective": objective,
            "constraints": constraints}


def test_solver_service_optimal_lp():
    # maximize x1 + x2 s.t. x1<=4, x2<=3 → 最適値 7
    spec = _spec(
        "maximize",
        [{"name": "x1", "type": "continuous", "low": 0},
         {"name": "x2", "type": "continuous", "low": 0}],
        {"x1": 1.0, "x2": 1.0},
        [{"coeffs": {"x1": 1.0}, "op": "<=", "rhs": 4.0},
         {"coeffs": {"x2": 1.0}, "op": "<=", "rhs": 3.0}],
    )
    res = solver_service.solve(spec)
    assert res["status"] == "OPTIMAL"
    assert res["objective_value"] == pytest.approx(7.0)
    assert res["solution"]["x1"] == pytest.approx(4.0)
    assert res["solution"]["x2"] == pytest.approx(3.0)
    assert res["backend"] == "PULP_CBC_CMD"


def test_solver_service_infeasible():
    spec = _spec(
        "maximize",
        [{"name": "x", "type": "continuous", "low": 0}],
        {"x": 1.0},
        [{"coeffs": {"x": 1.0}, "op": ">=", "rhs": 5.0},
         {"coeffs": {"x": 1.0}, "op": "<=", "rhs": 4.0}],
    )
    res = solver_service.solve(spec)
    assert res["status"] == "INFEASIBLE"


def test_solver_service_integer_variable():
    # maximize x s.t. 2x <= 7, x 整数 → x=3
    spec = _spec(
        "maximize",
        [{"name": "x", "type": "integer", "low": 0, "up": 10}],
        {"x": 1.0},
        [{"coeffs": {"x": 2.0}, "op": "<=", "rhs": 7.0}],
    )
    res = solver_service.solve(spec)
    assert res["status"] == "OPTIMAL"
    assert res["solution"]["x"] == pytest.approx(3.0)


def test_solver_service_unknown_variable_raises():
    spec = _spec(
        "minimize",
        [{"name": "x", "type": "continuous", "low": 0}],
        {"x": 1.0},
        [{"coeffs": {"ghost": 1.0}, "op": ">=", "rhs": 1.0}],
    )
    with pytest.raises(solver_service.SolverError):
        solver_service.solve(spec)


def test_dispatch_node_optimal():
    spec = _spec(
        "minimize",
        [{"name": "x", "type": "continuous", "low": 2}],
        {"x": 1.0},
        [{"coeffs": {"x": 1.0}, "op": ">=", "rhs": 2.0}],
    )
    out = SolverDispatchNode().execute({"problem_spec": json.dumps(spec)})
    assert out["solver_status"] == "OPTIMAL"
    assert out["objective_value"] == pytest.approx(2.0)
    assert json.loads(out["solution"])["x"] == pytest.approx(2.0)


def test_dispatch_node_bad_spec_is_solver_failed():
    bad = _spec("minimize", [{"name": "x", "type": "continuous"}], {"x": 1.0},
                [{"coeffs": {"ghost": 1.0}, "op": "<=", "rhs": 1.0}])
    out = SolverDispatchNode().execute({"problem_spec": json.dumps(bad)})
    assert out["error_code"] == "SOLVER_FAILED"


def test_dispatch_node_infeasible_is_not_error():
    spec = _spec("maximize", [{"name": "x", "type": "continuous", "low": 0}], {"x": 1.0},
                 [{"coeffs": {"x": 1.0}, "op": ">=", "rhs": 5.0},
                  {"coeffs": {"x": 1.0}, "op": "<=", "rhs": 4.0}])
    out = SolverDispatchNode().execute({"problem_spec": json.dumps(spec)})
    assert out.get("error") is None
    assert out["solver_status"] == "INFEASIBLE"


def test_dispatch_node_error_short_circuits():
    out = SolverDispatchNode().execute({"error": "prior"})
    assert out["error"] == "prior"
    assert "solver_status" not in out


# ── 変数の定義域が空（low > up）── 実 LLM が踏んだ経路の回帰 ────────────────
# 実 LLM は「x は 5 以上かつ 4 以下」を制約 2 本ではなく low=5 / up=4 と定式化した。
# そのまま CBC に渡すと **ソルバ実行そのものが失敗**し、正常な求解結果である
# INFEASIBLE ではなく SOLVER_FAILED（＋例外文にソルバ実行ファイルの絶対パス）になる。

_EMPTY_DOMAIN = {
    "sense": "maximize",
    "variables": [{"name": "x", "type": "continuous", "low": 5, "up": 4}],
    "objective": {"x": 1.0},
    "constraints": [],
}


def test_empty_variable_domain_is_infeasible_not_solver_failure():
    res = solver_service.solve(_EMPTY_DOMAIN)
    assert res["status"] == "INFEASIBLE"
    assert res["objective_value"] is None
    assert res["solution"] == {}


def test_empty_variable_domain_reaches_post_process_as_success():
    out = SolverDispatchNode().execute({"problem_spec": json.dumps(_EMPTY_DOMAIN)})
    assert out["status"] == "success"        # INFEASIBLE は正常な求解結果
    assert out["solver_status"] == "INFEASIBLE"
    assert out.get("error_code") is None


def test_unexpected_solver_exception_reports_type_name_only():
    """PuLP/CBC の例外文はソルバ実行ファイルの**サーバ側絶対パス**を含む。

    `str(exc)` を State に入れると get_output() 経由で呼び出し元まで返ってしまう。
    """
    class _Boom(Exception):
        def __str__(self) -> str:
            return "Pulp: Error while executing /srv/secret-path/cbc"

    def _raise(*args, **kwargs):
        raise _Boom()

    node = SolverDispatchNode()
    original = solver_service.solve
    solver_service.solve = _raise
    try:
        out = node.execute({"problem_spec": json.dumps(_EMPTY_DOMAIN)})
    finally:
        solver_service.solve = original

    assert out["error_code"] == "SOLVER_FAILED"
    blob = json.dumps(out, ensure_ascii=False)
    assert "/srv/secret-path" not in blob
    assert "_Boom" in blob            # 型名だけは残す


# ── レビュー H2: binary 変数の low/up ───────────────────────────────────
# PuLP は LpBinary の境界を無条件に 0..1 へ上書きする（pulp/pulp.py）。そのまま渡すと
# `binary, low=2` のような spec が黙って握り潰され、**入力の下限に違反する解を
# OPTIMAL として返す**。


def _binary(low, up, sense="maximize"):
    return {"sense": sense,
            "variables": [{"name": "x", "type": "binary", "low": low, "up": up}],
            "objective": {"x": 1.0}, "constraints": []}


def test_binary_bounds_outside_zero_one_are_infeasible():
    res = solver_service.solve(_binary(2, 3))
    assert res["status"] == "INFEASIBLE"
    assert res["solution"] == {}


def test_binary_fixed_to_one_is_respected():
    res = solver_service.solve(_binary(1, 1))
    assert res["status"] == "OPTIMAL"
    assert res["solution"]["x"] == pytest.approx(1.0)


def test_binary_fixed_to_zero_is_respected():
    # 最大化なのに up=0 で 0 に固定されること（0..1 に戻されていたら 1 が返る）
    res = solver_service.solve(_binary(None, 0))
    assert res["status"] == "OPTIMAL"
    assert res["solution"]["x"] == pytest.approx(0.0)
    assert res["objective_value"] == pytest.approx(0.0)


def test_binary_without_bounds_is_still_binary():
    res = solver_service.solve(_binary(None, None))
    assert res["status"] == "OPTIMAL"
    assert res["solution"]["x"] == pytest.approx(1.0)
