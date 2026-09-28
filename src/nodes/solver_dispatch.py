"""SolverDispatchNode — CMN-C1-021 main スロット（PuLP/CBC 決定論求解）.

released AgentCore SDK 契約: FunctionNode を継承し execute(self, state) で
**部分 dict** を返す。route() は main の status を読むため、成功時は必ず
status=SUCCESS を立てる（INFEASIBLE/UNBOUNDED/TIMEOUT も「正常な求解結果」= SUCCESS で
post_process に渡す）。SOLVER_FAILED のみ status=ERROR で finalize に落とす。

★ LLM 生成コードは実行しない。構造化データからのモデル構築のみ（サンドボックス不要）。
"""

from __future__ import annotations

import json

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

from src.services import solver_service


class SolverDispatchNode(FunctionNode):
    # S-1 多層防御: 後段ノードも明示宣言（entry 差し替え・単体実行でも S-1 を保つ / ADR-006）
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, solver_timeout_s: float = 10.0) -> None:
        self._timeout_s = float(solver_timeout_s)

    @staticmethod
    def _failed(message: str) -> dict:
        return {
            "error": message,
            "error_code": "SOLVER_FAILED",
            "error_message": message,
            "status": AgentStatus.ERROR.value,
        }

    def execute(self, state: dict) -> dict:
        # 上流（pre_process）のエラー/S-1 拒否はパススルー（pre_process→main は無条件辺）
        if state.get("error") or state.get("status") == AgentStatus.ERROR.value:
            return {"error": state.get("error"), "status": AgentStatus.ERROR.value}

        raw_spec = state.get("problem_spec")
        try:
            spec = raw_spec if isinstance(raw_spec, dict) else json.loads(raw_spec or "null")
        except (ValueError, TypeError):
            spec = None
        if not isinstance(spec, dict):
            return self._failed("no valid problem_spec for solver")

        try:
            result = solver_service.solve(spec, timeout_s=self._timeout_s)
        except solver_service.SolverError as exc:
            return self._failed(f"solver build failed: {exc}")
        except Exception as exc:  # 予期せぬソルバ例外も安全に伝播
            # 🔴 str(exc) を State に入れない。PuLP/CBC の例外文はソルバ実行ファイルの
            # **サーバ側絶対パス**を含み、それが get_output() で呼び出し元まで返る。
            # 理由として載せるのは例外の型名のみ。
            return self._failed(f"solver execution failed: {type(exc).__name__}")

        # S-4: at least one domain event per node — framework の監査契約
        emit_trace_event(
            "solver_dispatch_complete",
            {"solver_status": result["status"], "solver_backend": result["backend"]},
            state,
        )

        return {
            "solver_status": result["status"],
            "solution": json.dumps(result["solution"], ensure_ascii=False),
            "objective_value": result["objective_value"],
            "solver_backend": result["backend"],
            "status": AgentStatus.SUCCESS.value,
        }
