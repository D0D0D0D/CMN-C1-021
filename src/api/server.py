"""AgentCore Platform v1.0 — CMN-C1-021 HTTP entry point.

Entry points are adapters only — no business logic here. For platform-level
routing, AgentGateway calls agent.invoke() directly.

Runtime settings are loaded from config/config.yaml and passed through unchanged
(AgentRegistry does the same for the registry-served path). Without this, the
standalone adapter ran with config={} and every tunable — max_variables /
max_constraints / max_input_chars / solver_timeout_s / mask_patterns — silently
fell back to the in-code default.

この standalone server は config["llm"] を供給しない（モデル ID も具体クライアントも
テンプレに書かないのが方針）。したがって認証済みの /invoke でも、自然言語の
問題文だけを渡した呼び出しは LLM_NOT_CONFIGURED になるのが仕様。構造化 problem_spec
を input_context で渡す経路は LLM 不要で完走し、求解結果の説明文だけが省かれる。
実 LLM は AGENTIC STAR の entry adapter か、config へ BaseLLM を注入する呼び出し側が供給する。

Entry authentication is implemented by `entry_adapter.auth`:
  - a verified middleware trust (request.state.trust_level) is used as-is;
  - otherwise INVOKE_AUTH_TOKEN Bearer auth promotes ANONYMOUS →
    VERIFIED_EXTERNAL (_resolve_trust raises 401 on token mismatch);
  - an unauthenticated caller (no verified middleware trust AND no token
    promotion) is rejected with 401. The old fallback that defaulted such
    callers to ANONYMOUS and passed them straight into InvocationContext was
    fail-open (S-1 bypass) and is removed.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.utils.config_loader import load_config
from shared.secrets import factory as secrets_factory

from src.agent import OptimizationSolverAgent
from src.entry_adapter.auth import _require_verified_trust, _resolve_trust

app = FastAPI(title="OptimizationSolverAgent")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_YAML = _REPO_ROOT / "config" / "config.yaml"
# Absent config.yaml is tolerated, matching AgentRegistry._compile_and_cache()'s
# own `if exists() else {}` guard.
_config: dict[str, object] = load_config(str(_CONFIG_YAML)) if _CONFIG_YAML.exists() else {}

agent = OptimizationSolverAgent(config=_config)
agent.compile()
agent.provision_secrets(secrets_factory(namespace="cmn_c1_021", agent_name="optimization_solver_agent"))


def _ctx(request: Request, session_id: str) -> InvocationContext:
    """Build the InvocationContext AFTER entry authentication.

    Scope note: the entry_adapter part also exposes a tenant/org scope contract,
    but this template has NO scope model. OptimizationSolverAgent is a generic,
    use-case-agnostic Cat 1 solver — it holds no tenant/org resources and there
    is no code path that reads a per-request tenant/org scope. Enforcing
    authenticate_scope() would gate on a scope nothing consumes (STG-default
    theater), so only the auth core (_resolve_trust / _require_verified_trust)
    is wired.
    """
    trust, token_authenticated = _resolve_trust(request)
    # Unauthenticated = trust is still ANONYMOUS after resolution and no token
    # promotion. Test the trust VALUE, not attribute presence — a middleware
    # that explicitly stamps ANONYMOUS must still be treated as unauthenticated
    # (→ 401), not silently accepted as "verified because the attr exists".
    if trust is TrustLevel.ANONYMOUS and not token_authenticated:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    _require_verified_trust(trust)
    return InvocationContext(
        session_id=session_id or str(uuid4()),
        caller_trust_level=trust,
        caller_id=getattr(request.state, "caller_id", ""),
    )


class InvokeRequest(BaseModel):
    input: str = ""
    session_id: str = ""
    # problem_spec (JSON) may be supplied via input_context.
    input_context: dict = {}


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request):
    ctx = _ctx(request, req.session_id)
    with bound_secrets(agent._secrets_provider):
        return agent.invoke(req.input, session_id=ctx.session_id, ctx=ctx, input_context=req.input_context)


@app.get("/health")
def health():
    return {"status": "ok", "agent": "optimization_solver_agent"}
