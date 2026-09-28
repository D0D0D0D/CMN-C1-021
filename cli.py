"""AGENTIC STAR Marketplace entry adapter for CMN-C1-021."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from framework.secrets.base import SecretProvider

try:
    from framework.utils.config_loader import load_agent_config
except ImportError:  # AgentCore 1.0.0 / 1.0.1 compatibility
    from framework.utils.config_loader import load_config

    def load_agent_config(agent_dir: Path) -> dict[str, Any]:
        config_path = agent_dir / "config" / "config.yaml"
        return load_config(str(config_path)) if config_path.exists() else {}


from shared.services.llm.base_llm import BaseLLM

from src.agent import OptimizationSolverAgent

_NAMESPACE = "cmn"
_AGENT_NAME = "CMN-C1-021"
_PROVIDER_AGENT_NAME = "cmn_c1_021"

_AZURE_OPENAI_SECRET_KEYS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
)
_ALLOWED_SECRET_KEYS = frozenset(_AZURE_OPENAI_SECRET_KEYS)


class _AllowlistedSecretProvider(SecretProvider):
    """Delegate only this adapter's declared Azure connection values."""

    def __init__(self, delegate: SecretProvider) -> None:
        self._delegate = delegate
        self._namespace = _NAMESPACE
        self._agent_name = _PROVIDER_AGENT_NAME

    def get(self, key: str, default: str | None = None) -> str | None:
        if key not in _ALLOWED_SECRET_KEYS:
            return default
        value = self._delegate.get(key)
        return value if value else default


class _LazyAzureLLM(BaseLLM):  # type: ignore[misc]
    """Build an Azure client from freshly resolved secrets for every call."""

    def __init__(
        self,
        provider_factory: Callable[[], SecretProvider],
        *,
        tools: tuple[Any, ...] = (),
    ) -> None:
        self._provider_factory = provider_factory
        self._tools = tools

    def _client(self) -> BaseLLM:
        # This optional integration is absent from AgentCore 1.0.0/1.0.1's
        # registry-only wheel, so importing it must remain call-time lazy.
        from shared.services.llm.azure_openai_client import AzureOpenAIClient

        provider = self._provider_factory()
        client = AzureOpenAIClient(
            {
                "api_key": provider.require("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": provider.require("AZURE_OPENAI_ENDPOINT"),
                "azure_deployment": provider.require("AZURE_OPENAI_DEPLOYMENT"),
            }
        )
        return client.bind_tools(list(self._tools)) if self._tools else client

    def complete(self, messages: list[Any]) -> dict[str, Any]:
        return cast(dict[str, Any], self._client().complete(messages))

    def stream(self, messages: list[Any]) -> Iterator[str]:
        yield from self._client().stream(messages)

    def bind_tools(self, tools: list[Any]) -> "_LazyAzureLLM":
        return type(self)(self._provider_factory, tools=tuple(tools))


def _require_azure_endpoint(provider: SecretProvider) -> str:
    """Reject ``AZURE_OPENAI_ENDPOINT`` values that are not a bare Azure resource endpoint.

    ``AzureOpenAIClient`` appends its own ``/openai/v1/`` path, so anything
    beyond a scheme and host silently produces a broken request URL, and the
    failure surfaces inside ``complete()``. A node that swallows that exception
    degrades quietly while the invocation still reports ``status: success``,
    which is why this has to be a startup check. The value is reported here,
    never repaired.
    """

    endpoint = provider.require("AZURE_OPENAI_ENDPOINT")
    parts = urlsplit(endpoint)
    try:
        port = parts.port
    except ValueError:
        port = -1
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or port == -1
        or parts.path.strip("/")
        or parts.query
        or parts.fragment
        or any(character.isspace() for character in endpoint)
    ):
        raise ValueError(
            "Marketplace environment variable 'AZURE_OPENAI_ENDPOINT' must be a bare Azure "
            "resource endpoint -- scheme and host only, with no path (such as "
            "'/openai'), query or fragment"
        )
    return endpoint


class MarketplaceGraph(OptimizationSolverAgent):
    """Marketplace-only input hardening and Azure LLM wiring."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        marketplace_config = dict(config or {})
        # register_nodes() runs during compile(), before the runner provisions
        # secrets. The wrapper itself is credential-free and therefore safe to
        # place in config now; it resolves the current provider only on calls.
        if marketplace_config.get("llm") is None:
            marketplace_config["llm"] = _LazyAzureLLM(lambda: self._secrets_provider)
        super().__init__(config=marketplace_config)

    def invoke(
        self,
        user_input: str,
        session_id: str = "",
        ctx: Any | None = None,
        input_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # conversation_history is unverified free text outside this agent's
        # S-2 gate. Rebuild context so no current or future runner key enters
        # State. Plain user_input is the sole Marketplace problem input.
        del input_context

        # Marketplace cannot resume AWAITING_HUMAN. Preserve all other frozen
        # InvocationContext fields; absence remains absence for S-1 rejection.
        marketplace_ctx = replace(ctx, hitl_allowed=False) if ctx is not None else None
        return cast(
            dict[str, Any],
            super().invoke(
                user_input,
                session_id=session_id,
                ctx=marketplace_ctx,
                input_context={},
            ),
        )

    def provision_secrets(self, provider: SecretProvider) -> None:
        # Always delegate to the provider the runner hands us. This used to
        # prefer shared.secrets.env_provider.EnvProvider "when installed",
        # which made the adapter's behaviour depend on the installed SDK
        # version rather than on its own contract: that module is absent in
        # 1.0.1 and present from 1.0.2, so the same code took the
        # `except ImportError` branch on one wheel and the `else` branch on the
        # next — silently DISCARDING the runner's provider and reading only the
        # process environment. Narrowing is this adapter's own job and is done
        # below by _AllowlistedSecretProvider, which exposes exactly the three
        # declared Azure keys regardless of how wide the delegate's view is.
        allowlisted_provider = _AllowlistedSecretProvider(provider)
        for key in _AZURE_OPENAI_SECRET_KEYS:
            allowlisted_provider.require(key)
        _require_azure_endpoint(allowlisted_provider)
        super().provision_secrets(allowlisted_provider)


def main() -> None:
    # Marketplace support is an optional SDK extra and is absent from the
    # 1.0.x test wheel. Keep this import out of module collection.
    from shared.bootstrap import run_agent_marketplace

    run_agent_marketplace(
        MarketplaceGraph,
        agent_name=_AGENT_NAME,
        namespace=_NAMESPACE,
        config=load_agent_config(Path(__file__).resolve().parent),
    )


if __name__ == "__main__":
    main()
