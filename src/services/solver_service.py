"""SolverService — 構造化スペックから PuLP モデルを決定論的に構築し求解する。

docs/02_design.md §6。**LLM が生成したコードは一切実行しない。** 構造化された
`problem_spec`（変数・係数・演算子・rhs の純データ）からのみモデルを組み立てるため、
コードインジェクション / サンドボックス脱出のリスクがない（サンドボックス基盤不要）。

対応: 線形計画（LP）/ 混合整数計画（MILP）。変数は continuous / integer / binary。
制約は線形（<=, >=, ==）。目的は minimize / maximize。

返却ステータス: OPTIMAL | INFEASIBLE | UNBOUNDED | TIMEOUT | UNDEFINED。
変数の定義域が空（low > up、binary で {0,1} と交わらない）場合はソルバを呼ばずに
INFEASIBLE を返す。binary の low/up は PuLP が黙って 0..1 に上書きするため、
{0,1} との共通部分をここで解決する（1 点なら Integer で固定）。
INFEASIBLE / UNBOUNDED / TIMEOUT はエラーではなく正常な求解結果として返す
（呼び出し側が graceful に説明する）。
"""

from __future__ import annotations

from typing import Any

import pulp

_SENSE = {"maximize": pulp.LpMaximize, "minimize": pulp.LpMinimize}
_CAT = {
    "continuous": pulp.LpContinuous,
    "integer": pulp.LpInteger,
    "binary": pulp.LpBinary,
}
# PuLP LpStatus(int) → 本サービスのステータス文字列
#   常に timeLimit を設定して solve するため、NotSolved(0) は実質「制限時間到達」を意味する
#   → TIMEOUT にマップ（暫定解の有無に依存せず判定。docs/02_design.md §6）。
_STATUS = {
    pulp.LpStatusOptimal: "OPTIMAL",        # 1
    pulp.LpStatusInfeasible: "INFEASIBLE",  # -1
    pulp.LpStatusUnbounded: "UNBOUNDED",    # -2
    pulp.LpStatusNotSolved: "TIMEOUT",      # 0 — timeLimit 到達
    pulp.LpStatusUndefined: "UNDEFINED",    # -3
}

BACKEND = "PULP_CBC_CMD"


class SolverError(Exception):
    """構造化スペックの不整合（未知変数参照・不正な sense/op 等）で送出。"""


def solve(problem_spec: dict, timeout_s: float = 10.0) -> dict[str, Any]:
    """problem_spec を求解する。

    Args:
        problem_spec: docs/02_design.md §4 の JSON 契約に従う dict。
        timeout_s:    CBC の制限時間（秒）。

    Returns:
        {status, solution (var→value dict), objective_value (float|None), backend}

    Raises:
        SolverError: スペックが線形最適化問題として組み立てられない場合。
    """
    sense_key = str(problem_spec.get("sense", "")).lower()
    if sense_key not in _SENSE:
        raise SolverError(f"unknown sense: {problem_spec.get('sense')!r}")

    variables = problem_spec.get("variables") or []
    if not variables:
        raise SolverError("no variables in problem_spec")

    prob = pulp.LpProblem("CMN_C1_021", _SENSE[sense_key])

    # ── 変数生成 ──
    var_map: dict[str, pulp.LpVariable] = {}
    empty_domain = False
    for v in variables:
        name = v.get("name")
        if not name or not isinstance(name, str):
            raise SolverError(f"variable missing valid name: {v!r}")
        cat = _CAT.get(str(v.get("type", "continuous")).lower())
        if cat is None:
            raise SolverError(f"unknown variable type: {v.get('type')!r}")
        low = v.get("low", 0)
        up = v.get("up", None)
        # low > up は変数の定義域が空 = 実行不能。CBC に渡すと**ソルバ実行そのものが
        # 失敗**し（`Pulp: Error while executing .../cbc`）、正常な求解結果である
        # INFEASIBLE ではなく SOLVER_FAILED になってしまう。数学的に答えが決まって
        # いるので、ソルバを呼ばずに INFEASIBLE を返す。
        # （実 LLM が「x は 5 以上かつ 4 以下」を low=5 / up=4 と定式化して踏んだ経路）
        if low is not None and up is not None and float(low) > float(up):
            empty_domain = True
            var_map[name] = pulp.LpVariable(name, lowBound=low, upBound=up, cat=cat)
            continue

        if cat == pulp.LpBinary:
            # 🔴 PuLP は LpBinary の境界を**無条件に 0..1 へ上書き**する
            # （pulp/pulp.py: `if cat == const.LpBinary: self.lowBound = 0; self.upBound = 1`）。
            # そのため `binary, low=2` のような spec は黙って握り潰され、
            # **入力の下限に違反する解を OPTIMAL として返して**しまう。
            # 指定区間と {0,1} の共通部分を自分で解決する。
            allowed = [b for b in (0, 1)
                       if (low is None or b >= float(low)) and (up is None or b <= float(up))]
            if not allowed:
                empty_domain = True
                var_map[name] = pulp.LpVariable(name, cat=pulp.LpBinary)
                continue
            if len(allowed) == 1:
                # 1 点に固定。LpBinary のままだと PuLP が 0..1 に戻すので Integer で固定する。
                fixed = allowed[0]
                var_map[name] = pulp.LpVariable(
                    name, lowBound=fixed, upBound=fixed, cat=pulp.LpInteger
                )
                continue
            var_map[name] = pulp.LpVariable(name, cat=pulp.LpBinary)
            continue

        var_map[name] = pulp.LpVariable(name, lowBound=low, upBound=up, cat=cat)

    if empty_domain:
        return {
            "status": "INFEASIBLE",
            "solution": {},
            "objective_value": None,
            "backend": BACKEND,
        }

    def _linexpr(coeffs: dict, where: str):
        terms = []
        for var_name, coeff in (coeffs or {}).items():
            if var_name not in var_map:
                raise SolverError(f"{where} references unknown variable {var_name!r}")
            terms.append(float(coeff) * var_map[var_name])
        return pulp.lpSum(terms)

    # ── 目的関数 ──
    objective = problem_spec.get("objective") or {}
    if not objective:
        raise SolverError("empty objective")
    prob += _linexpr(objective, "objective"), "objective"

    # ── 制約 ──
    for i, c in enumerate(problem_spec.get("constraints") or []):
        op = c.get("op")
        rhs = c.get("rhs")
        if op not in ("<=", ">=", "=="):
            raise SolverError(f"constraint[{i}] unknown op: {op!r}")
        if rhs is None:
            raise SolverError(f"constraint[{i}] missing rhs")
        lhs = _linexpr(c.get("coeffs"), f"constraint[{i}]")
        rhs = float(rhs)
        if op == "<=":
            prob += (lhs <= rhs), f"c{i}"
        elif op == ">=":
            prob += (lhs >= rhs), f"c{i}"
        else:
            prob += (lhs == rhs), f"c{i}"

    # ── 求解（CBC、タイムアウト制御、ログ抑制） ──
    solver = pulp.PULP_CBC_CMD(timeLimit=float(timeout_s), msg=0)
    raw_status = prob.solve(solver)
    status = _STATUS.get(raw_status, "UNDEFINED")

    # 変数値が得られているか（タイムアウトでも暫定解が出ることがある）
    solution: dict[str, float] = {}
    have_values = False
    for name, var in var_map.items():
        val = var.value()
        if val is not None:
            have_values = True
            solution[name] = round(float(val), 9)

    # OPTIMAL は最適値、TIMEOUT は暫定解の目的関数値（あれば）を返す
    objective_value = None
    if status in ("OPTIMAL", "TIMEOUT") and have_values:
        ov = prob.objective.value()
        objective_value = round(float(ov), 9) if ov is not None else None

    return {
        "status": status,
        "solution": solution,
        "objective_value": objective_value,
        "backend": BACKEND,
    }
