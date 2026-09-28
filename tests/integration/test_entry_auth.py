"""Entry-adapter authentication tests.

Verifies the fail-closed entry auth wired into src/api/server.py._ctx():
  - an unauthenticated caller (no verified middleware trust, no token) → 401;
  - a caller carrying a valid INVOKE_AUTH_TOKEN Bearer header → not 401
    (promoted to VERIFIED_EXTERNAL);
  - a caller carrying a wrong Bearer token → 401.

Requires the AgentCore wheel (framework.* / shared.*) and pulp to import
src.api.server (agent.compile at module import); skipped otherwise, matching
the rest of the suite that depends on the installed wheel.
"""

from __future__ import annotations

import importlib
import json

import pytest

pytest.importorskip("framework")
pytest.importorskip("pulp")
fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", "test-entry-token")
    server = importlib.import_module("src.api.server")
    importlib.reload(server)
    return fastapi_testclient.TestClient(server.app)


def test_unauthenticated_invoke_is_rejected(client):
    resp = client.post("/invoke", json={"input": "maximize x subject to x <= 1"})
    assert resp.status_code == 401


def test_wrong_token_is_rejected(client):
    resp = client.post(
        "/invoke",
        json={"input": "maximize x subject to x <= 1"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_valid_token_passes_entry_auth(client):
    resp = client.post(
        "/invoke",
        json={"input": "maximize x subject to x <= 1"},
        headers={"Authorization": "Bearer test-entry-token"},
    )
    # Entry auth must NOT reject a validly-tokened caller. Downstream solve may
    # return any non-auth status; only assert the entry gate did not 401/403.
    assert resp.status_code not in (401, 403)


def test_health_requires_no_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200


# ── Standalone adapter の構築契約（config 受け渡し / LLM 未注入）─────────────
# 旧実装は `OptimizationSolverAgent()` を素で呼んでおり、config/config.yaml が
# **一度も読まれていなかった**（max_variables / solver_timeout_s 等が全部コード既定）。

_AUTH = {"Authorization": "Bearer test-entry-token"}

# config/config.yaml の実値をリテラルで書く（production から計算しない）。
_EXPECTED_CONFIG = {"max_variables": 200, "max_constraints": 500,
                    "max_input_chars": 10000, "solver_timeout_s": 10}

# 手計算で最適値が分かる LP（x1+x2 最大化 / x1<=4 / x2<=3 → 7.0）。
_SPEC = {
    "sense": "maximize",
    "variables": [{"name": "x1", "type": "continuous", "low": 0},
                  {"name": "x2", "type": "continuous", "low": 0}],
    "objective": {"x1": 1.0, "x2": 1.0},
    "constraints": [{"coeffs": {"x1": 1.0}, "op": "<=", "rhs": 4.0},
                    {"coeffs": {"x2": 1.0}, "op": "<=", "rhs": 3.0}],
}


def test_server_passes_config_yaml_to_the_agent(client):
    server = importlib.import_module("src.api.server")
    for key, value in _EXPECTED_CONFIG.items():
        assert server.agent.config.get(key) == value


def test_natural_language_without_llm_reports_llm_not_configured(client):
    # standalone server は config["llm"] を供給しない（方針: モデル ID も具体
    # クライアントもテンプレに書かない）。自然言語経路は決定論代替が無いので失敗が正。
    resp = client.post(
        "/invoke",
        json={"input": "x1+x2 を最大化。x1<=4, x2<=3, どちらも 0 以上。"},
        headers=_AUTH,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["error_code"] == "LLM_NOT_CONFIGURED"


def test_structured_spec_completes_without_an_llm(client):
    # 構造化 problem_spec 経路は LLM を使わないので standalone でも完走する。
    resp = client.post(
        "/invoke",
        json={"input": "", "session_id": "s", "input_context": {"problem_spec": json.dumps(_SPEC)}},
        headers=_AUTH,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["solver_status"] == "OPTIMAL"
    assert body["objective_value"] == 7.0
    # AGENTIC STAR runner が success 時に必須とする `output` キー
    assert body["output"] == body["formatted_output"]
    assert "OPTIMAL" in body["output"]
