"""State schema for CMN-C1-021 OptimizationSolverAgent.

Released AgentCore SDK (v1.0.0) 準拠: `framework.schemas.agent_state.AgentState` を
直接継承する flat TypedDict。共有フィールド（user_input / input_context / status /
caller_trust_level / node_history / error_log / formatted_output / execution_time /
session_id / trace_id / correlation_id 等）は AgentState から継承する。

ここにはドメイン固有フィールドのみ追加する。primitives + JSON 文字列のみ（msgpack 安全
= LangGraph checkpoint 安全。docs/02_design.md §4）。
problem_spec / solution は JSON STRING として保持し、list/dict を直接 state に入れない（PB-2）。

NOTE: InvocationContext は State に保存しない（checkpoint に載せられない = PB-5）。trust は invoke() が
`caller_trust_level` として書き込む（AgentState 継承フィールド）。
"""

from __future__ import annotations

from typing import Optional

from framework.schemas.agent_state import AgentState


class OptimizationState(AgentState):
    """OptimizationSolverAgent のドメイン State（AgentState 拡張）。"""

    # ── Input（自然言語経路。構造化入力は input_context["problem_spec"] か state["problem_spec"]）──
    problem_description: Optional[str]   # 自然言語の最適化問題（max_input_chars 上限）

    # ── ConstraintParseNode 出力 ──
    problem_spec: Optional[str]          # JSON: {sense, variables[], objective{}, constraints[]}
    spec_valid: Optional[bool]           # 構造妥当性チェック結果
    variable_count: Optional[int]
    constraint_count: Optional[int]

    # ── SolverDispatchNode 出力 ──
    solver_status: Optional[str]         # OPTIMAL | INFEASIBLE | UNBOUNDED | TIMEOUT | UNDEFINED
    solution: Optional[str]              # JSON: {"x1": 4.0, "x2": 3.0}
    objective_value: Optional[float]     # 目的関数値（OPTIMAL/TIMEOUT 暫定解時）
    solver_backend: Optional[str]        # 使用ソルバ識別子

    # ── エラー伝播（status とは別にドメイン診断コードを保持）──
    error: Optional[str]
    error_code: Optional[str]            # S1_TRUST_DENIED / EMPTY_INPUT / PROBLEM_TOO_LARGE /
                                         # SPEC_PARSE_FAILED / S2_INJECTION_DETECTED / SOLVER_FAILED / S3_BLOCKED /
                                         # LLM_NOT_CONFIGURED / LLM_CALL_FAILED（自然言語経路のみ）
    error_message: Optional[str]
