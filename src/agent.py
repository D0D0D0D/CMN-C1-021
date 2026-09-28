"""OptimizationSolverAgent — CMN-C1-021 (released AgentCore SDK v1.0.0).

Agent は Graph を継承し、本番エントリは L1 `AgentBaseGraph.invoke(user_input, session_id,
ctx, input_context)` を継承する（`.run()` は存在しない — 呼ぶと AttributeError になり、
S-1 trust ゲートも通らない）。`self.graph` 合成も不可（独自クラスは 5 層セキュリティを迂回する）。

5 層セキュリティ（released SDK の実配線に整合）:
  S-1 required_trust_level : `BaseNode.__call__()` が各ノードで自動強制（ADR-006）。entry の
                             ConstraintParseNode が VERIFIED_EXTERNAL を要求（default-deny）。
  S-2 入力ゲート            : ConstraintParseNode.execute 内（credential scan）
  S-3 出力ゲート            : SolutionFormatNode.execute 内（PII マスク + credential ブロック）
  S-4 監査トレース          : SolutionFormatNode.execute 内 emit_trace_event（メタのみ）
  S-5 credential scan       : CI の静的スキャン（ソースに資格情報を置かない）

旧 graph レベルの _security_gate_input/output / _emit_trace_event / _invoke_impl フックは
released wheel に無いため廃止（移行: SDK v1.0.0）。
"""

from __future__ import annotations

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import OptimizationSolverGraph

TEMPLATE_ID = "CMN-C1-021"
# 有効値は ANONYMOUS / VERIFIED_EXTERNAL / INTERNAL の 3 つのみ（他の値は import 時に AttributeError）。
REQUIRED_TRUST_LEVEL = TrustLevel.VERIFIED_EXTERNAL


class OptimizationSolverAgent(OptimizationSolverGraph):
    """Cat 1 数理最適化エージェント。Graph を継承し invoke() を継承する。"""

    TEMPLATE_ID = TEMPLATE_ID
    REQUIRED_TRUST_LEVEL = REQUIRED_TRUST_LEVEL
