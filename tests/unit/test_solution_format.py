"""Unit tests — SolutionFormatNode（自然言語化 / 決定論サマリ / S-3 マスク・ブロック）."""

import json

from src.nodes.solution_format import SolutionFormatNode


class _FakeLLM:
    def __init__(self, payload: str):
        self._payload = payload

    def generate(self, prompt: str) -> str:
        return self._payload


def _state(status="OPTIMAL", solution=None, objective=7.0):
    return {
        "solver_status": status,
        "solution": json.dumps(solution if solution is not None else {"x1": 4.0, "x2": 3.0}),
        "objective_value": objective,
    }


def test_optimal_summary_contains_objective_and_assignments():
    out = SolutionFormatNode(llm_client=_FakeLLM("説明文")).execute(_state())
    assert out.get("error") is None
    fo = out["formatted_output"]
    assert "OPTIMAL" in fo
    assert "7.0" in fo
    assert "x1 = 4.0" in fo and "x2 = 3.0" in fo


def test_infeasible_explained_not_errored():
    out = SolutionFormatNode(llm_client=_FakeLLM("")).execute(
        _state(status="INFEASIBLE", solution={}, objective=None))
    assert out.get("error") is None
    assert "INFEASIBLE" in out["formatted_output"]


def test_credential_in_llm_explanation_blocked():
    # LLM が credential を吐いた場合 S-3 で停止
    out = SolutionFormatNode(llm_client=_FakeLLM("api_key=supersecretvalue123")).execute(_state())
    assert out["error_code"] == "S3_BLOCKED"
    assert out["formatted_output"] is None


def test_llm_failure_is_non_fatal_summary_remains():
    class _Boom:
        def generate(self, prompt):
            raise RuntimeError("LLM down")
    out = SolutionFormatNode(llm_client=_Boom()).execute(_state())
    assert out.get("error") is None
    assert "OPTIMAL" in out["formatted_output"]   # 決定論サマリは残る


def test_pii_mask_applied():
    # mynumber パターンを含む説明をマスク（apply_masks）
    out = SolutionFormatNode(
        llm_client=_FakeLLM("マイナンバーは 1234-5678-9012 です"),
        mask_patterns=["mynumber"],
    ).execute(_state())
    assert "1234-5678-9012" not in out["formatted_output"]
    assert "[REDACTED:mynumber]" in out["formatted_output"]


def test_error_state_short_circuits():
    out = SolutionFormatNode().execute({"error": "prior"})
    assert out["error"] == "prior"
    assert "formatted_output" not in out


# ── 出力の並び: 決定論サマリが先頭・LLM の説明文はその後（H3 裁定 = 案②）──────
# 説明文は LLM が書く散文で、ソルバの数値を再述・改竄しうる。先に読まれるものを
# 決定論的な事実にする。期待値はすべてリテラルで書く（production から計算しない）。

# レビューで実測した「矛盾する数値を書く LLM」と同型。ソルバの解は objective=7.0 /
# x1=4.0 / x2=3.0 なので、999 はどこにも根拠がない。
_CONTRADICTING = "最適値は 999 で、x1=999 が正解です。"


def _formatted(llm_payload):
    return SolutionFormatNode(llm_client=_FakeLLM(llm_payload)).execute(
        _state(solution={"x1": 4.0, "x2": 3.0}, objective=7.0)
    )["formatted_output"]


def test_deterministic_summary_comes_before_the_llm_explanation():
    fo = _formatted(_CONTRADICTING)
    # 決定論サマリの見出しが先頭にあること
    assert fo.startswith("## 求解結果: OPTIMAL")
    # ソルバの数値が LLM の作り話より先に読まれること
    assert fo.index("目的関数値: 7.0") < fo.index("999")
    assert fo.index("x1 = 4.0") < fo.index("999")


def test_llm_explanation_is_kept_and_marked_as_advisory():
    fo = _formatted(_CONTRADICTING)
    # 説明文は捨てない
    assert _CONTRADICTING in fo
    # ただし「参考」節として区別でき、どちらが正かが出力上に書いてある
    assert "## 参考: 言語モデルによる説明" in fo
    assert "数値が食い違う場合は上の求解結果が正です" in fo
    assert fo.index("## 求解結果: OPTIMAL") < fo.index("## 参考: 言語モデルによる説明")


def test_degraded_output_has_no_advisory_section():
    # LLM 未注入 → 決定論サマリのみ（従来の縮退挙動を壊さない）
    fo = SolutionFormatNode().execute(
        _state(solution={"x1": 4.0}, objective=7.0)
    )["formatted_output"]
    assert fo.startswith("## 求解結果: OPTIMAL")
    assert "## 参考: 言語モデルによる説明" not in fo
    assert "目的関数値: 7.0" in fo
