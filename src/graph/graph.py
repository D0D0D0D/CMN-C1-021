"""OptimizationSolverGraph — CMN-C1-021 (released AgentCore SDK v1.0.0).

released L1 `AgentBaseGraph` は **固定 5 ノード backbone** を強制する:
    START → initialize → pre_process → main → {route} → post_process → finalize → END
                                            ↓ (RETRY, max 3)
                                         pre_process
`super().register_nodes()` が initialize / finalize を自動注入し、テンプレは
pre_process / main / post_process の 3 スロットを埋める（compile() が None を検出すると
MissingNodeError）。route() は state["status"] を読み、SUCCESS のときのみ post_process に
到達する。`add_edges()` / `route()` は override しない（L1 が配線する）。

論理 3 ノード（constraint_parse → solver_dispatch → solution_format）を 3 スロットへ 1:1 で
マッピングする（docs/02_design.md §2, §3）:
    pre_process  : ConstraintParseNode   (S-1 entry / S-2 入力ゲート)
    main         : SolverDispatchNode     (PuLP/CBC 決定論求解)
    post_process : SolutionFormatNode     (整形 + S-3 出力ゲート + S-4 監査)

旧 graph レベルの _security_gate_input/output/_emit_trace_event は released wheel に存在
しないため廃止し、S-2/S-3/S-4 を各ノード内に実装した（移行: SDK v1.0.0）。

LLM クライアントは **`config["llm"]` 経由でのみ**受け取る（同系テンプレート共通の
注入シーム）。旧実装の `llm_client=` という独自コンストラクタ引数は、framework の
実契約に従う呼び出し側（`src/api/server.py`）が渡さないため永久に未配線だった
（src/services/llm_client.py の docstring 参照）。
"""

from __future__ import annotations

from framework.graph.agent_base_graph import AgentBaseGraph

from src.nodes.constraint_parse import ConstraintParseNode
from src.nodes.solver_dispatch import SolverDispatchNode
from src.nodes.solution_format import SolutionFormatNode
from src.schemas.state import OptimizationState
from src.services.llm_client import resolve_llm_client


class OptimizationSolverGraph(AgentBaseGraph):
    """固定 3 スロット: pre_process(constraint_parse) → main(solver_dispatch) → post_process(solution_format)。"""

    def __init__(self, config: dict | None = None) -> None:
        # BaseGraph.__init__ が self.config を保持する。**設定はここでは読まない** —
        # 読み取りは register_nodes()（= compile 時）まで遅らせる。AGENTIC STAR の runner
        # には `agent_cls()` で構築してから config を埋める版が実在するため、__init__ で
        # 固めると LLM が永久に未注入のままになる。
        super().__init__(config=dict(config or {}))

    @property
    def name(self) -> str:
        return "OptimizationSolverAgent"

    @property
    def state_schema(self) -> type:
        return OptimizationState

    def register_nodes(self) -> None:
        super().register_nodes()  # initialize / finalize 自動注入 + pre/main/post スロット枠
        cfg = self.config or {}
        # LLM は config["llm"] 経由でのみ受け取る。未注入なら番兵が解決され、
        # 自然言語経路は LLM_NOT_CONFIGURED で失敗する（整形段は決定論サマリへ縮退）。
        # 生成パラメータ（model / temperature / max_tokens）は注入クライアントが持つ。
        llm_client = resolve_llm_client(cfg.get("llm"))
        self._nodes["pre_process"] = ConstraintParseNode(
            llm_client=llm_client,
            max_input_chars=int(cfg.get("max_input_chars", 10000)),
            max_variables=int(cfg.get("max_variables", 200)),
            max_constraints=int(cfg.get("max_constraints", 500)),
        )
        self._nodes["main"] = SolverDispatchNode(solver_timeout_s=float(cfg.get("solver_timeout_s", 10.0)))
        self._nodes["post_process"] = SolutionFormatNode(
            llm_client=llm_client,
            mask_patterns=list(cfg.get("mask_patterns", [])),
        )

    def get_output(self, state) -> dict:
        """最終 state からクライアント向け出力を構築（primitives + list/dict のみ = msgpack 安全）。"""
        return {
            # AGENTIC STAR runner は success 時に `output` を必須とする
            # （shared/bootstrap/marketplace_app.py: 無いと
            #  "returned status=success but output is missing" で RuntimeError）。
            # 既存キーは変えず追加のみ。
            "output": state.get("formatted_output"),
            "formatted_output": state.get("formatted_output"),
            "solver_status": state.get("solver_status"),
            "objective_value": state.get("objective_value"),
            "solution": state.get("solution"),
            "solver_backend": state.get("solver_backend"),
            "variable_count": state.get("variable_count"),
            "constraint_count": state.get("constraint_count"),
            "spec_valid": state.get("spec_valid"),
            "status": state.get("status"),
            "error": state.get("error"),
            "error_code": state.get("error_code"),
            "error_message": state.get("error_message"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "session_id": state.get("session_id"),
            "node_history": state.get("node_history", []),
            "error_log": state.get("error_log", []),
        }
