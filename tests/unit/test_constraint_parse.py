"""Unit tests — ConstraintParseNode（入力検証 / LLM 抽出 / 構造化入力 / サイズ上限）."""

import json

from src.nodes.constraint_parse import ConstraintParseNode

SPEC = {
    "sense": "maximize",
    "variables": [
        {"name": "x1", "type": "continuous", "low": 0, "up": None},
        {"name": "x2", "type": "continuous", "low": 0, "up": None},
    ],
    "objective": {"x1": 1.0, "x2": 1.0},
    "constraints": [
        {"coeffs": {"x1": 1.0}, "op": "<=", "rhs": 4.0},
        {"coeffs": {"x2": 1.0}, "op": "<=", "rhs": 3.0},
    ],
}


class _FakeLLM:
    def __init__(self, payload: str):
        self._payload = payload

    def generate(self, prompt: str) -> str:
        return self._payload


def test_parse_from_natural_language_via_llm():
    node = ConstraintParseNode(llm_client=_FakeLLM(json.dumps(SPEC)))
    out = node.execute({"problem_description": "x1+x2 を最大化, x1<=4, x2<=3"})
    assert out.get("error") is None
    assert out["spec_valid"] is True
    assert out["variable_count"] == 2
    assert out["constraint_count"] == 2
    assert json.loads(out["problem_spec"])["sense"] == "maximize"


def test_parse_strips_json_fence():
    fenced = "```json\n" + json.dumps(SPEC) + "\n```"
    node = ConstraintParseNode(llm_client=_FakeLLM(fenced))
    out = node.execute({"problem_description": "maximize ..."})
    assert out["spec_valid"] is True


def test_provided_structured_spec_skips_llm():
    # llm_client なしでも、構造化 problem_spec を直接渡せば検証して通す
    node = ConstraintParseNode(llm_client=_FakeLLM("SHOULD NOT BE USED"))
    out = node.execute({"problem_spec": json.dumps(SPEC)})
    assert out["spec_valid"] is True
    assert out["variable_count"] == 2


def test_empty_input():
    out = ConstraintParseNode().execute({"problem_description": "   "})
    assert out["error_code"] == "EMPTY_INPUT"
    assert out["spec_valid"] is False


def test_credential_in_input_blocked():
    out = ConstraintParseNode().execute(
        {"problem_description": "solve with key sk-" + "a" * 25})
    assert out["error_code"] == "S2_INJECTION_DETECTED"


def test_invalid_llm_json():
    out = ConstraintParseNode(llm_client=_FakeLLM("not json at all")).execute(
        {"problem_description": "maximize x"})
    assert out["error_code"] == "SPEC_PARSE_FAILED"


def test_spec_unknown_variable_in_constraint():
    bad = json.loads(json.dumps(SPEC))
    bad["constraints"][0]["coeffs"] = {"ghost": 1.0}
    out = ConstraintParseNode(llm_client=_FakeLLM(json.dumps(bad))).execute(
        {"problem_description": "x"})
    assert out["error_code"] == "SPEC_PARSE_FAILED"


def test_problem_too_large():
    node = ConstraintParseNode(llm_client=_FakeLLM(json.dumps(SPEC)), max_variables=1)
    out = node.execute({"problem_description": "x"})
    assert out["error_code"] == "PROBLEM_TOO_LARGE"


def test_provided_spec_credential_blocked():
    # 構造化 spec 直接投入経路でも S-2（credential scan）を通す（回帰）
    bad = json.loads(json.dumps(SPEC))
    bad["variables"][0]["name"] = "x_sk-" + "a" * 25   # credential パターンを変数名に混入
    # 目的/制約の参照名も揃える必要はない（credential 検出が validate より先に走る）
    out = ConstraintParseNode().execute({"problem_spec": json.dumps(bad)})
    assert out["error_code"] == "S2_INJECTION_DETECTED"


def test_non_numeric_bound_rejected_early():
    # LLM が "low": "infinity" を返すケースを SPEC_PARSE_FAILED で早期に弾く
    bad = json.loads(json.dumps(SPEC))
    bad["variables"][0]["low"] = "infinity"
    out = ConstraintParseNode(llm_client=_FakeLLM(json.dumps(bad))).execute(
        {"problem_description": "x"})
    assert out["error_code"] == "SPEC_PARSE_FAILED"


def test_non_numeric_coefficient_rejected():
    bad = json.loads(json.dumps(SPEC))
    bad["objective"]["x1"] = "lots"
    out = ConstraintParseNode(llm_client=_FakeLLM(json.dumps(bad))).execute(
        {"problem_description": "x"})
    assert out["error_code"] == "SPEC_PARSE_FAILED"


def test_error_state_short_circuits():
    out = ConstraintParseNode().execute({"error": "prior", "problem_description": "x"})
    assert out["error"] == "prior"


# ── レビュー M2: 有限性 / 変数名の一意性 ────────────────────────────────
# `json.loads()` は既定で NaN / Infinity / -Infinity を float として受理する。
# 有限性を検査しないと PuLP へ NaN 係数が渡る。


def _mutate(**kw):
    spec = json.loads(json.dumps(SPEC))
    for k, v in kw.items():
        if k == "low":
            spec["variables"][0]["low"] = v
        elif k == "coeff":
            spec["objective"]["x1"] = v
        elif k == "rhs":
            spec["constraints"][0]["rhs"] = v
    return spec


def test_json_literal_nan_is_rejected():
    raw = '{"sense":"maximize","variables":[{"name":"x","type":"continuous","low":0}],' \
          '"objective":{"x":NaN},"constraints":[]}'
    out = ConstraintParseNode(llm_client=_FakeLLM(raw)).execute({"problem_description": "x"})
    assert out["error_code"] == "SPEC_PARSE_FAILED"


def test_json_literal_infinity_bound_is_rejected():
    raw = '{"sense":"maximize","variables":[{"name":"x","type":"continuous","low":0,"up":Infinity}],' \
          '"objective":{"x":1},"constraints":[]}'
    out = ConstraintParseNode(llm_client=_FakeLLM(raw)).execute({"problem_description": "x"})
    assert out["error_code"] == "SPEC_PARSE_FAILED"


def test_infinite_rhs_is_rejected():
    out = ConstraintParseNode(llm_client=_FakeLLM(json.dumps(_mutate(rhs=float("inf"))))).execute(
        {"problem_description": "x"})
    assert out["error_code"] == "SPEC_PARSE_FAILED"


def test_duplicate_variable_names_are_rejected():
    # 重複名以外はすべて妥当な spec にする（未知変数参照など別の理由で落ちると
    # 重複チェックそのものを検証したことにならない）。
    spec = {
        "sense": "maximize",
        "variables": [{"name": "x1", "type": "continuous", "low": 0, "up": 4},
                      {"name": "x1", "type": "continuous", "low": 0, "up": 3}],
        "objective": {"x1": 1.0},
        "constraints": [{"coeffs": {"x1": 1.0}, "op": "<=", "rhs": 4.0}],
    }
    out = ConstraintParseNode(llm_client=_FakeLLM(json.dumps(spec))).execute(
        {"problem_description": "x"})
    assert out["error_code"] == "SPEC_PARSE_FAILED"


def test_constraint_count_limit_enforced():
    node = ConstraintParseNode(llm_client=_FakeLLM(json.dumps(SPEC)), max_constraints=1)
    out = node.execute({"problem_description": "x"})
    assert out["error_code"] == "PROBLEM_TOO_LARGE"
