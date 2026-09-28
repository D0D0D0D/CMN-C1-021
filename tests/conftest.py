"""tests/conftest.py — CMN-C1-021 test bootstrap (released AgentCore SDK v1.0.0).

The L1 framework (``framework.*`` / ``shared.*``) is provided by the installed
``agenticstar-agentcore`` wheel — there are no vendored stubs and no sys.modules stubbing
(migrated from the interim stub model, SDK v1.0.0).

This only ensures the repo root is importable so ``import src...`` works when running
``pytest tests/`` without an editable install.
"""

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture
def agent_config():
    # 実 LLM 検証ハーネス用。本番 config/config.yaml を基底に、追加設定は不要。
    return {}


@pytest.fixture
def case_payload():
    # 🔴 この AG の中核能力は「自然言語の問題文 → ソルバ仕様」の定式化。
    # 構造化 problem_spec を渡すと ConstraintParseNode は LLM を一切呼ばないため、
    # 中核経路が未検証のまま PASS する（2026-08-18 の記録がその状態だった）。
    # 手計算で最適値が分かる問題を自然言語で渡す（x1+x2 最大化, x1<=4, x2<=3 → 7）。
    return (
        "製品 x1 と x2 を作ります。利益はどちらも 1 個あたり 1 万円です。"
        "x1 は最大 4 個、x2 は最大 3 個までしか作れません。"
        "どちらも 0 個以上の連続量として、利益の合計を最大化してください。"
    )


@pytest.fixture
def live_input_context():
    # problem_spec を渡さない = 自然言語経路（LLM が定式化する）を通す。
    return {}
