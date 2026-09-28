"""Unit tests for the AGENTIC STAR Marketplace entry adapter."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import Any

import pytest

import cli
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.base import MissingSecret, SecretProvider
from src.agent import OptimizationSolverAgent


class _DictProvider(SecretProvider):
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.requested: list[str] = []

    def get(self, key: str, default: str | None = None) -> str | None:
        self.requested.append(key)
        return self.values.get(key, default)


def _azure_values() -> dict[str, str]:
    return {
        "AZURE_OPENAI_API_KEY": "key-one",
        "AZURE_OPENAI_ENDPOINT": "https://one.example",
        "AZURE_OPENAI_DEPLOYMENT": "deployment-one",
    }


def test_marketplace_graph_installs_lazy_llm_before_compile() -> None:
    graph = cli.MarketplaceGraph(config={"solver_timeout_s": 10})

    assert isinstance(graph.config["llm"], cli._LazyAzureLLM)
    graph.compile()


def test_invoke_discards_runner_input_context_and_disables_hitl(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_invoke(
        self: OptimizationSolverAgent,
        user_input: str,
        session_id: str = "",
        ctx: InvocationContext | None = None,
        input_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del self
        captured.update(
            user_input=user_input,
            session_id=session_id,
            ctx=ctx,
            input_context=input_context,
        )
        return {"status": "success"}

    monkeypatch.setattr(OptimizationSolverAgent, "invoke", fake_invoke)
    original_ctx = InvocationContext(
        session_id="session-1",
        caller_id="marketplace",
        caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
        hitl_allowed=True,
    )

    result = cli.MarketplaceGraph().invoke(
        "maximize x",
        session_id="session-1",
        ctx=original_ctx,
        input_context={"conversation_history": ["unverified"], "problem_spec": "bypass"},
    )

    assert result == {"status": "success"}
    assert captured["input_context"] == {}
    assert captured["ctx"].hitl_allowed is False
    assert captured["ctx"].caller_trust_level is TrustLevel.VERIFIED_EXTERNAL
    assert original_ctx.hitl_allowed is True


def test_invoke_preserves_missing_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_invoke(self: OptimizationSolverAgent, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del self, args
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(OptimizationSolverAgent, "invoke", fake_invoke)

    cli.MarketplaceGraph().invoke("problem", ctx=None)

    assert captured["ctx"] is None


@pytest.mark.parametrize("missing_key", cli._AZURE_OPENAI_SECRET_KEYS)
def test_provision_secrets_fails_closed_for_each_missing_azure_value(missing_key: str) -> None:
    values = _azure_values()
    del values[missing_key]

    with pytest.raises(MissingSecret):
        cli.MarketplaceGraph().provision_secrets(_DictProvider(values))


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview",
        "https://example.services.ai.azure.example/openai",
        "https://example.services.ai.azure.example/v1",
        "https://example.services.ai.azure.example?api-version=2024-10-21",
        "example.services.ai.azure.example",
    ],
    ids=["api-path", "openai-path", "base-path", "query-only", "missing-scheme"],
)
def test_unusable_azure_endpoint_shape_fails_during_startup(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
) -> None:
    """AzureOpenAIClient appends its own path, so anything but a bare endpoint breaks.

    The resulting error surfaces inside complete(); a node that swallows it
    degrades silently while the invocation still reports status: success, so the
    adapter has to reject the shape at startup instead.
    """

    values = _azure_values()
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "gate-test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gate-test-deployment")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", endpoint)
    values["AZURE_OPENAI_ENDPOINT"] = endpoint

    with pytest.raises(ValueError, match="bare"):
        cli.MarketplaceGraph().provision_secrets(_DictProvider(values))


def test_provisioned_provider_exposes_only_allowlisted_keys() -> None:
    values = _azure_values() | {"UNDECLARED_SECRET": "must-not-pass"}
    graph = cli.MarketplaceGraph()

    graph.provision_secrets(_DictProvider(values))

    assert graph._secrets_provider.get("AZURE_OPENAI_API_KEY") == "key-one"
    assert graph._secrets_provider.get("UNDECLARED_SECRET") is None


def test_provision_secrets_uses_the_runner_provider_even_when_env_provider_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner's provider must be used regardless of the installed SDK version.

    `shared.secrets.env_provider` is absent in agentcore 1.0.1 and present from
    1.0.2. The adapter previously preferred that module "when installed", so the
    same code discarded the runner's provider on 1.0.2+ and read only the
    process environment — three tests here went red purely on a wheel bump. This
    pins the version-independent contract: the delegate is always the provider
    the runner passed, and EnvProvider is never constructed.
    """
    constructed: list[tuple[str, str]] = []

    class FakeEnvProvider(_DictProvider):
        def __init__(self, namespace: str, agent_name: str) -> None:
            constructed.append((namespace, agent_name))
            super().__init__(_azure_values())

    env_module = ModuleType("shared.secrets.env_provider")
    env_module.EnvProvider = FakeEnvProvider  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, env_module.__name__, env_module)
    graph = cli.MarketplaceGraph()
    runner_values = _azure_values() | {"AZURE_OPENAI_DEPLOYMENT": "deployment-from-runner"}

    graph.provision_secrets(_DictProvider(runner_values))

    # The runner's value wins — not the environment provider's.
    assert graph._secrets_provider.require("AZURE_OPENAI_DEPLOYMENT") == "deployment-from-runner"
    assert constructed == []


def test_lazy_llm_resolves_rotated_values_and_builds_a_fresh_client_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_configs: list[dict[str, str]] = []

    class FakeAzureOpenAIClient:
        def __init__(self, config: dict[str, str]) -> None:
            created_configs.append(dict(config))
            self._config = config

        def complete(self, messages: list[Any]) -> dict[str, Any]:
            return {"content": str(messages), "model": self._config["azure_deployment"]}

    azure_module = ModuleType("shared.services.llm.azure_openai_client")
    azure_module.AzureOpenAIClient = FakeAzureOpenAIClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, azure_module.__name__, azure_module)

    values = _azure_values()
    graph = cli.MarketplaceGraph()
    graph.provision_secrets(_DictProvider(values))
    lazy_llm = graph.config["llm"]

    first = lazy_llm.complete([{"role": "user", "content": "first"}])
    values.update(
        AZURE_OPENAI_API_KEY="key-two",
        AZURE_OPENAI_ENDPOINT="https://two.example",
        AZURE_OPENAI_DEPLOYMENT="deployment-two",
    )
    second = lazy_llm.complete([{"role": "user", "content": "second"}])

    assert first["model"] == "deployment-one"
    assert second["model"] == "deployment-two"
    assert len(created_configs) == 2
    assert created_configs[0]["api_key"] == "key-one"
    assert created_configs[1]["api_key"] == "key-two"


def test_runner_order_compile_then_provision_supports_natural_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = {
        "sense": "maximize",
        "variables": [{"name": "x", "type": "continuous", "low": 0, "up": None}],
        "objective": {"x": 1},
        "constraints": [{"coeffs": {"x": 1}, "op": "<=", "rhs": 4}],
    }

    class FakeAzureOpenAIClient:
        def __init__(self, config: dict[str, str]) -> None:
            self._model = config["azure_deployment"]

        def complete(self, messages: list[Any]) -> dict[str, Any]:
            prompt = messages[0]["content"]
            content = json.dumps(spec) if "JSON 契約" in prompt else "補足説明"
            return {"content": content, "model": self._model}

    azure_module = ModuleType("shared.services.llm.azure_openai_client")
    azure_module.AzureOpenAIClient = FakeAzureOpenAIClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, azure_module.__name__, azure_module)

    graph = cli.MarketplaceGraph(config={"solver_timeout_s": 10})
    graph.compile()
    graph.provision_secrets(_DictProvider(_azure_values()))
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)

    result = graph.invoke("x を最大化。x は4以下。", ctx=ctx, input_context={"conversation_history": ["ignore"]})

    assert result["status"] == "success"
    assert result["solver_status"] == "OPTIMAL"
    assert result["objective_value"] == 4.0
    assert "補足説明" in result["formatted_output"]


def test_lazy_azure_import_is_not_required_at_module_import() -> None:
    assert "shared.services.llm.azure_openai_client" not in sys.modules
