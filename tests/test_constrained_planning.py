"""
Constrained decoding on the planner, the Definition of Done and the judge: with
act_protocol="json" these honour the SAME switch as the act-loop, sending
response_format=json_schema and reading a grammar-enforced object. Each degrades to
the free chat()+regex path if the server rejects response_format, and the judge
still HALTs on a real outage. These tests pin those paths with a faked LLM.
"""
import types

import pytest

from achilles import planner as P
from achilles import acceptance as A
from achilles.llm import JsonReply, LLMError
from achilles.tools import ToolContext


def _cfg(**over):
    base = dict(act_protocol="json", temperature=0.2, comfy_url="", max_tokens=0,
                judge_model="", judge_char_per_file=6000, judge_char_budget=24000)
    base.update(over)
    return types.SimpleNamespace(**base)


# ---- planner ---------------------------------------------------------------

def test_make_plan_json_returns_steps(monkeypatch):
    monkeypatch.setattr(P, "complete_json",
                        lambda *a, **k: JsonReply(obj={"steps": ["do a", "do b"]},
                                                  content=""))
    assert P.make_plan(_cfg(), "goal", "tree") == ["do a", "do b"]


def test_make_plan_json_strips_leftover_bullets(monkeypatch):
    # A weak model may still tuck a checkbox/number into the array item.
    monkeypatch.setattr(P, "complete_json",
                        lambda *a, **k: JsonReply(
                            obj={"steps": ["- [ ] first", "2. second", "  third  "]},
                            content=""))
    assert P.make_plan(_cfg(), "goal", "tree") == ["first", "second", "third"]


def test_make_plan_json_degrades_on_llmerror(monkeypatch):
    def _reject(*a, **k):
        raise LLMError("response_format not supported")
    monkeypatch.setattr(P, "complete_json", _reject)
    monkeypatch.setattr(P, "chat", lambda *a, **k: "- [ ] alpha\n- [ ] beta")
    assert P.make_plan(_cfg(), "goal", "tree") == ["alpha", "beta"]


def test_make_plan_json_parses_content_when_schema_ignored(monkeypatch):
    monkeypatch.setattr(P, "complete_json",
                        lambda *a, **k: JsonReply(obj=None,
                                                  content="- [ ] only step"))
    assert P.make_plan(_cfg(), "goal", "tree") == ["only step"]


def test_plan_system_steers_to_separate_files():
    # backlog #14: the planner is told to prefer separate small files for web pages.
    assert "styles.css" in P.PLAN_SYSTEM and "script.js" in P.PLAN_SYSTEM


def test_make_plan_native_does_not_use_json(monkeypatch):
    # act_protocol != "json" must leave the planner on the free chat() path.
    def _boom(*a, **k):
        raise AssertionError("complete_json must not be called in native mode")
    monkeypatch.setattr(P, "complete_json", _boom)
    monkeypatch.setattr(P, "chat", lambda *a, **k: "- [ ] s1")
    assert P.make_plan(_cfg(act_protocol="native"), "goal", "tree") == ["s1"]


# ---- Definition of Done ----------------------------------------------------

def test_make_acceptance_json_parses_criteria(monkeypatch):
    monkeypatch.setattr(A, "complete_json",
                        lambda *a, **k: JsonReply(
                            obj={"criteria": ["exists: index.html",
                                              "contains: index.html :: <canvas",
                                              "judge: looks professional"]},
                            content=""))
    crit = A.make_acceptance(_cfg(), "goal", "tree")
    kinds = [(c.kind, c.text) for c in crit]
    assert ("exists", "index.html") in kinds
    assert ("contains", "index.html :: <canvas") in kinds
    assert ("judge", "looks professional") in kinds


def test_make_acceptance_json_strips_model_supplied_bullet(monkeypatch):
    # The model often keeps the "- [ ]" from the prompt examples inside the array
    # item; the kind tag must still survive (not mis-parse as a judge line).
    monkeypatch.setattr(A, "complete_json",
                        lambda *a, **k: JsonReply(
                            obj={"criteria": ["- [ ] exists: index.html",
                                              "contains: index.html :: <h1>"]},
                            content=""))
    crit = A.make_acceptance(_cfg(), "goal", "tree")
    kinds = {c.kind for c in crit}
    assert kinds == {"exists", "contains"}          # neither collapsed into judge


def test_make_acceptance_json_degrades_on_llmerror(monkeypatch):
    def _reject(*a, **k):
        raise LLMError("no response_format")
    monkeypatch.setattr(A, "complete_json", _reject)
    monkeypatch.setattr(A, "chat", lambda *a, **k: "- [ ] exists: a.py")
    crit = A.make_acceptance(_cfg(), "goal", "tree")
    assert [(c.kind, c.text) for c in crit] == [("exists", "a.py")]


# ---- judge -----------------------------------------------------------------

def test_judge_json_maps_verdicts(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "complete_json",
                        lambda *a, **k: JsonReply(
                            obj={"verdicts": [{"pass": True, "reason": "found it"},
                                              {"pass": False, "reason": "missing"}]},
                            content=""))
    items = [A.Criterion("judge", "a"), A.Criterion("judge", "b")]
    out = A._judge(_cfg(), items, ToolContext(tmp_path), log=lambda *_: None)
    assert out == [(True, "found it"), (False, "missing")]


def test_judge_json_pads_missing_verdicts_as_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "complete_json",
                        lambda *a, **k: JsonReply(
                            obj={"verdicts": [{"pass": True, "reason": "ok"}]},
                            content=""))
    items = [A.Criterion("judge", "a"), A.Criterion("judge", "b")]
    out = A._judge(_cfg(), items, ToolContext(tmp_path), log=lambda *_: None)
    assert out[0] == (True, "ok")
    assert out[1][0] is False                       # missing → FAIL


def test_judge_json_degrades_to_text_then_still_halts(monkeypatch, tmp_path):
    # response_format rejected → fall through to the text judge; a real outage there
    # must still raise JudgeUnavailable (not silently fail every criterion).
    def _reject(*a, **k):
        raise LLMError("no response_format")

    def _outage(*a, **k):
        raise LLMError("connection refused")
    monkeypatch.setattr(A, "complete_json", _reject)
    monkeypatch.setattr(A, "chat", _outage)
    items = [A.Criterion("judge", "a")]
    with pytest.raises(A.JudgeUnavailable):
        A._judge(_cfg(), items, ToolContext(tmp_path), log=lambda *_: None)


def test_judge_json_falls_back_to_text_verdicts(monkeypatch, tmp_path):
    # response_format rejected, but the text judge answers fine → parse its lines.
    def _reject(*a, **k):
        raise LLMError("no response_format")
    monkeypatch.setattr(A, "complete_json", _reject)
    monkeypatch.setattr(A, "chat", lambda *a, **k: "1: PASS — good\n2: FAIL — nope")
    items = [A.Criterion("judge", "a"), A.Criterion("judge", "b")]
    out = A._judge(_cfg(), items, ToolContext(tmp_path), log=lambda *_: None)
    assert out[0][0] is True and out[1][0] is False
