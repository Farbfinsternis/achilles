"""Tests for spec.py — the Planungsmodus artefact and its pure layers.

The interview router, the spec.md round trip (incl. a goal that itself contains
code fences) and the two consumers (en_goal, verbatim_criteria) are deterministic
and pinned here. normalize() is exercised with a stubbed model.
"""

from achilles import spec as spec_mod
from achilles.llm import JsonReply, LLMError
from achilles.spec import (
    Spec, SLOTS, route_answer, render_spec, parse_spec, en_goal,
    verbatim_criteria, normalize,
)


class _Cfg:
    # "native" is the DEFAULT protocol; normalize() must still reach for the
    # constrained-JSON path (response_format is enforced independently of the act
    # loop), only degrading to chat() when the server rejects response_format.
    act_protocol = "native"
    temperature = 0.2
    model = "m"


def _no_response_format(monkeypatch):
    """Simulate a server that rejects response_format: complete_json raises, so
    normalize() falls through to the (stubbed) free chat() path."""
    def boom(*a, **k):
        raise LLMError("response_format unsupported")
    monkeypatch.setattr(spec_mod, "complete_json", boom)


def _spec():
    return Spec(
        source_language="de",
        original_goal="Baue eine Seite für die Bäckerei Sonnenschein.",
        purpose="Advertise a neighbourhood bakery on one page.",
        audience="Local walk-in customers.",
        features=["Hero with name and tagline", "Opening hours"],
        scope="Static HTML/CSS/JS, no backend.",
        ui_ux="Warm, rustic, mobile-first.",
        verbatim=["Bäckerei Sonnenschein", "Frische Brötchen täglich"],
    )


# ---- router ---------------------------------------------------------------

def test_route_answer_blank_and_skipwords_skip():
    assert route_answer("") == ("skip", "")
    assert route_answer("   ") == ("skip", "")
    assert route_answer("egal") == ("skip", "")
    assert route_answer("Weiß nicht") == ("skip", "")


def test_route_answer_literal_is_answer():
    assert route_answer("  Eine Bäckerei-Seite ") == ("answer", "Eine Bäckerei-Seite")


# ---- spec.md round trip ---------------------------------------------------

def test_render_parse_round_trip():
    s = _spec()
    back = parse_spec(render_spec(s))
    assert back.source_language == s.source_language
    assert back.original_goal == s.original_goal
    assert back.purpose == s.purpose
    assert back.audience == s.audience
    assert back.features == s.features
    assert back.scope == s.scope
    assert back.ui_ux == s.ui_ux
    assert back.verbatim == s.verbatim


def test_round_trip_goal_with_code_fences():
    # A goal that itself contains a ``` fence must survive the adaptive outer fence.
    s = _spec()
    s.original_goal = "Prompt:\n```js\nconsole.log('hi')\n```\nnoch mehr text"
    back = parse_spec(render_spec(s))
    assert back.original_goal == s.original_goal


def test_verbatim_quotes_stripped_on_parse():
    md = render_spec(_spec())
    assert '- "Bäckerei Sonnenschein"' in md          # rendered quoted
    assert parse_spec(md).verbatim[0] == "Bäckerei Sonnenschein"   # parsed unquoted


# ---- consumers ------------------------------------------------------------

def test_en_goal_carries_verbatim_with_no_translate_note():
    g = en_goal(_spec())
    assert "Purpose:" in g and "Prototype scope:" in g
    assert "do NOT translate" in g
    assert "Bäckerei Sonnenschein" in g               # literal, original language


def test_verbatim_criteria_are_contains_any():
    crits = verbatim_criteria(_spec())
    assert [c.kind for c in crits] == ["contains_any", "contains_any"]
    assert crits[0].text == "Bäckerei Sonnenschein"


def test_verbatim_criteria_empty_when_none():
    s = _spec()
    s.verbatim = []
    assert verbatim_criteria(s) == []


# ---- normalize() ----------------------------------------------------------

def test_normalize_uses_constrained_json_on_native(monkeypatch):
    # THE regression test: on the `native` default, normalize() must still enforce
    # the spec shape through the constrained-JSON channel and NEVER touch free chat()
    # — otherwise a weak model's prose reply collapses the two-layer split.
    obj = {
        "source_language": "de", "purpose": "A bakery page", "audience": "locals",
        "features": ["hero", "hours"], "scope": "static", "ui_ux": "warm",
        "verbatim": ["Bäckerei Sonnenschein"],
    }
    monkeypatch.setattr(spec_mod, "complete_json",
                        lambda *a, **k: JsonReply(obj=obj, content="", finish_reason="stop"))
    def fail_chat(*a, **k):
        raise AssertionError("chat() must not be called when response_format works")
    monkeypatch.setattr(spec_mod, "chat", fail_chat)
    s = normalize(_Cfg(), {"purpose": "eine Bäckerei-Seite"}, "Baue eine Bäckerei-Seite.")
    assert s.source_language == "de"
    assert s.purpose == "A bakery page"
    assert s.features == ["hero", "hours"]
    assert s.verbatim == ["Bäckerei Sonnenschein"]
    assert s.original_goal == "Baue eine Bäckerei-Seite."      # pinned verbatim


def test_normalize_parses_model_json_via_chat_fallback(monkeypatch):
    # When the server rejects response_format, normalize() degrades to free chat()
    # and still parses a clean JSON reply.
    _no_response_format(monkeypatch)
    payload = (
        '{"source_language":"de","purpose":"A bakery page","audience":"locals",'
        '"features":["hero","hours"],"scope":"static","ui_ux":"warm",'
        '"verbatim":["Bäckerei Sonnenschein"]}'
    )
    monkeypatch.setattr(spec_mod, "chat", lambda *a, **k: payload)
    s = normalize(_Cfg(), {"purpose": "eine Bäckerei-Seite"}, "Baue eine Bäckerei-Seite.")
    assert s.source_language == "de"
    assert s.purpose == "A bakery page"
    assert s.features == ["hero", "hours"]
    assert s.verbatim == ["Bäckerei Sonnenschein"]
    assert s.original_goal == "Baue eine Bäckerei-Seite."      # pinned verbatim


def test_normalize_falls_back_on_garbage(monkeypatch):
    # Unparseable model output (even on the chat fallback) must NOT crash the run;
    # degrade to raw answers.
    _no_response_format(monkeypatch)
    monkeypatch.setattr(spec_mod, "chat", lambda *a, **k: "sorry, I cannot do that")
    answers = {"purpose": "eine Bäckerei-Seite", "audience": ""}
    s = normalize(_Cfg(), answers, "Baue eine Bäckerei-Seite.")
    assert s.purpose == "eine Bäckerei-Seite"                  # raw answer kept
    assert s.audience == "general audience"                    # slot default
    assert s.verbatim == []                                    # cannot extract
    assert s.original_goal == "Baue eine Bäckerei-Seite."


def test_slots_catalogue_has_five_fields():
    assert [s.field for s in SLOTS] == ["purpose", "audience", "features", "scope", "ui_ux"]
