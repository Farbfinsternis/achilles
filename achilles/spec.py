"""
spec.py — the Planungsmodus artefact: structure a raw prompt BEFORE planning.

A raw prompt — often not even in the model's training language — is the biggest
failure source for a small model. In interview mode a FIXED question catalogue
(no smart model needed to ask) fills five slots, then ONE model call normalises
the answers into a machine-readable Spec and splits two language layers:

  * Reasoning/structure (Purpose/Audience/Features/Scope/UI) -> ENGLISH, so the
    planner and the Definition of Done think in the model's strong language.
  * Product content (names, copy, labels) -> ORIGINAL language, VERBATIM. These
    literals become contains_any acceptance anchors, so the acceptance check
    matches the output language by construction (never a translated needle).

The Spec is the input to make_plan (via en_goal) and to the Definition of Done
(via verbatim_criteria). The ORIGINAL goal is preserved verbatim as the content
truth pinned into the executor — the English view never reaches the executor,
or it would build an English UI.

Like planner.py/acceptance.py, the intelligence is the model's; this module owns
the format contract (the Spec sections, the normalise schema) so dumb code can
read smart output.
"""

import re
from dataclasses import dataclass

from .acceptance import Criterion
from .llm import chat, complete_json, LLMError


@dataclass(frozen=True)
class Slot:
    field: str          # "purpose"
    prompt: str         # the question shown to the user
    default: str        # fallback text, OR "" = infer from the goal in normalize()


@dataclass
class Spec:
    source_language: str
    original_goal: str          # verbatim, original language -> content pin
    purpose: str                # EN
    audience: str               # EN
    features: list              # EN, list[str]
    scope: str                  # EN
    ui_ux: str                  # EN (may be "")
    verbatim: list              # original language, verbatim, list[str] -> DoD


# The fixed interview catalogue — one slot per question, deterministic, no model
# involved in ASKING. An empty default means "infer this from the goal in the
# single normalize() call".
SLOTS = [
    Slot("purpose",  "Zweck des Projekts?",            ""),
    Slot("audience", "Zielgruppe?",                    "general audience"),
    Slot("features", "Kernfunktionen?",                ""),
    Slot("scope",    "Prototyp-Scope?",                "minimal working prototype"),
    Slot("ui_ux",    "UI/UX — Aussehen & Bedienung?",  ""),
]
_DEFAULTS = {s.field: s.default for s in SLOTS}

# Answers the router reads as "skip this slot, take the default". Everything else
# is taken literally as the answer — no model classification in the loop.
_SKIP_WORDS = {
    "", "egal", "ist egal", "weiß nicht", "weiss nicht", "keine ahnung", "k.a.",
    "ka", "skip", "überspringen", "ueberspringen", "-", "—", "n/a", "na",
    "idk", "dunno", "don't know", "dont know", "no idea",
}


def route_answer(raw: str) -> tuple:
    """The interview router: two intents, purely heuristic, no model call.

    Returns ("skip", "")   for a blank/"don't care" answer -> the slot default,
        or  ("answer", v)  for anything else -> the literal value.

    There is deliberately no `back` intent: correcting a slot happens at the spec
    approval gate (the user sees the assembled spec and edits it there), which
    keeps the interview deterministic on slow local hardware."""
    v = (raw or "").strip()
    if v.lower() in _SKIP_WORDS:
        return ("skip", "")
    return ("answer", v)


# ---- normalize(): the single model call ----------------------------------

SPEC_SYSTEM = """You turn a raw project request plus short interview answers into a structured
product spec for a coding agent.

CRITICAL language rule:
- Write Purpose, Audience, Core features, Prototype scope and UI/UX in ENGLISH
  (translate if the input is another language). This is reasoning for the agent.
- Extract the literal PRODUCT CONTENT strings the finished product must show
  (names, taglines, labels, headings, button text) and return them UNCHANGED in
  their ORIGINAL language under `verbatim`. NEVER translate these strings.
- Detect the source language of the request and return it as an ISO code in
  `source_language` (e.g. "de", "en").

For any slot whose answer is missing, infer a sensible value from the goal. Keep
each field short and concrete. `verbatim` holds only strings that must appear
literally in the product; if none are stated, return an empty list."""

# Constrained shape for normalize(): a flat object, one field per spec section.
# Like PLAN_SCHEMA/ACCEPT_SCHEMA the server grammar-forces this so a weak model
# cannot bury the spec in prose.
SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "source_language": {"type": "string"},
        "purpose": {"type": "string"},
        "audience": {"type": "string"},
        "features": {"type": "array", "items": {"type": "string"}},
        "scope": {"type": "string"},
        "ui_ux": {"type": "string"},
        "verbatim": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["source_language", "purpose", "audience", "features", "scope",
                "ui_ux", "verbatim"],
    "additionalProperties": False,
}
SPEC_JSON_NOTE = ('\n\nReturn ONLY a JSON object with keys source_language, '
                  "purpose, audience, features (array), scope, ui_ux, verbatim "
                  "(array of original-language literal strings).")


def _render_answers(answers: dict, original_goal: str, note: str = "") -> str:
    lines = ["Original goal (verbatim, source language):", original_goal, "",
             "Interview answers:"]
    for slot in SLOTS:
        a = (answers.get(slot.field) or "").strip()
        if a:
            shown = a
        elif slot.default:
            shown = f"(no answer — default: {slot.default})"
        else:
            shown = "(no answer — infer from the goal)"
        lines.append(f"- {slot.field}: {shown}")
    if note.strip():
        # A correction from the spec approval gate: apply it on top of the answers.
        lines += ["", f"Correction to apply to the spec: {note.strip()}"]
    lines += ["", "Produce the spec JSON now."]
    return "\n".join(lines)


def _loads(text: str):
    import json
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _spec_via_json(config, messages):
    """Constrained path. ALWAYS attempted, regardless of act_protocol: the spec is a
    one-shot structured extraction, and LM Studio grammar-enforces the response_format
    json_schema CONTENT channel independently of native tool-calling. This is the
    difference between planner and spec on the `native` default — the planner degrades
    to a prose parser, but a raw spec has no prose form, so a weak model's free chat()
    reply rarely parses as JSON and the whole two-layer split silently collapses to
    _fallback_spec. Returns the object dict, or None so normalize() falls back to free
    chat() when the server rejects response_format."""
    try:
        jr = complete_json(config, messages, SPEC_SCHEMA, temperature=config.temperature)
    except LLMError:
        return None                          # e.g. response_format unsupported → fall back
    if jr.obj is not None:
        return jr.obj
    return _loads(jr.content)                 # schema ignored: parse the text we got


def _spec_from_obj(obj: dict, original_goal: str) -> Spec:
    def s(key):
        return str(obj.get(key) or "").strip()

    def arr(key):
        return [str(x).strip() for x in (obj.get(key) or []) if str(x).strip()]

    # verbatim literals may arrive quoted ("Bäckerei"); strip one surrounding pair.
    verbatim = [v[1:-1] if len(v) >= 2 and v[0] == v[-1] == '"' else v
                for v in arr("verbatim")]
    return Spec(
        source_language=s("source_language") or "unknown",
        original_goal=original_goal,
        purpose=s("purpose"),
        audience=s("audience"),
        features=arr("features"),
        scope=s("scope"),
        ui_ux=s("ui_ux"),
        verbatim=verbatim,
    )


def _fallback_spec(answers: dict, original_goal: str) -> Spec:
    """Degraded spec when the model returns unparseable output (NOT when it is
    unreachable — that LLMError propagates). Uses the raw answers untranslated and
    the slot defaults; verbatim is empty (we cannot extract it without the model).
    The run stays alive and the spec approval gate lets the user fix it."""
    def val(field):
        a = (answers.get(field) or "").strip()
        return a or _DEFAULTS.get(field, "")

    feats = val("features")
    return Spec(
        source_language="unknown",
        original_goal=original_goal,
        purpose=val("purpose"),
        audience=val("audience"),
        features=[feats] if feats else [],
        scope=val("scope"),
        ui_ux=val("ui_ux"),
        verbatim=[],
    )


def normalize(config, answers: dict, original_goal: str, note: str = "") -> Spec:
    """The ONE model call of the Planungsmodus: translate the reasoning fields to
    English, extract verbatim product content in the original language, and infer
    any empty slot from the goal — all in a single call. Raises LLMError only when
    the model is unreachable; unparseable output degrades to _fallback_spec.

    `note` carries a free-text correction from the spec approval gate ("edit"): the
    spec is re-normalised from the same answers with the change applied."""
    system = SPEC_SYSTEM + SPEC_JSON_NOTE
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _render_answers(answers, original_goal, note)},
    ]
    obj = _spec_via_json(config, messages)
    if obj is None:
        reply = chat(config, messages, temperature=config.temperature)
        obj = _loads(reply)
    if obj is None:
        return _fallback_spec(answers, original_goal)
    return _spec_from_obj(obj, original_goal)


# ---- consumers: planner and Definition of Done ----------------------------

def en_goal(spec: Spec) -> str:
    """The English view handed to make_plan / make_acceptance as one goal string.
    Includes the verbatim literals (in their original language, quoted) with an
    explicit DO-NOT-TRANSLATE note, so the planner places the exact strings and the
    judge can reason about them — while the reasoning around them stays English."""
    parts = [f"Purpose: {spec.purpose}", f"Audience: {spec.audience}"]
    if spec.features:
        parts.append("Core features:\n" + "\n".join(f"- {f}" for f in spec.features))
    parts.append(f"Prototype scope: {spec.scope}")
    if spec.ui_ux:
        parts.append(f"UI/UX: {spec.ui_ux}")
    if spec.verbatim:
        parts.append("Verbatim content — use these EXACT strings, do NOT translate:\n"
                     + "\n".join(f'- "{v}"' for v in spec.verbatim))
    return "\n\n".join(parts)


def verbatim_criteria(spec: Spec) -> list:
    """The verbatim content strings as deterministic, PATHLESS acceptance anchors.
    contains_any means "must appear verbatim in SOME project file" — the file is
    unknown at spec time (the planner runs later), and these needles come from the
    same field the executor treats as content truth, so they match the output
    language by construction."""
    return [Criterion("contains_any", v) for v in spec.verbatim if v.strip()]


# ---- persistence: .achilles/spec.md ---------------------------------------

_META_RE = re.compile(r"^>\s*([\w-]+):\s*(.*)$")


def _fence_for(text: str) -> str:
    """A backtick fence at least one tick longer than any run inside the text, so a
    goal that itself contains ``` code fences round-trips (CommonMark rule)."""
    longest = max((len(m.group()) for m in re.finditer(r"`+", text or "")), default=0)
    return "`" * max(3, longest + 1)


def render_spec(spec: Spec) -> str:
    fence = _fence_for(spec.original_goal)
    out = [
        "# Achilles — Spec", "",
        "> Spec-Version: 1",
        f"> Source-Language: {spec.source_language}",
        "> Mode: interview", "",
        "## Original goal", f"{fence}text", spec.original_goal, fence, "",
        "## Purpose", spec.purpose, "",
        "## Audience", spec.audience, "",
        "## Core features", *[f"- {f}" for f in spec.features], "",
        "## Prototype scope", spec.scope, "",
        "## UI / UX", spec.ui_ux, "",
        "## Verbatim content", *[f'- "{v}"' for v in spec.verbatim], "",
    ]
    return "\n".join(out) + "\n"


def parse_spec(text: str) -> Spec:
    """Read spec.md back (for resume). Header-driven, tolerant of blank lines. The
    Original goal is stored in an adaptive fence and read out verbatim."""
    lines = (text or "").splitlines()
    meta: dict = {}
    sections: dict = {}
    goal_lines: list = []
    cur = None
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if cur is None:
            mm = _META_RE.match(line)
            if mm:
                meta[mm.group(1).lower()] = mm.group(2).strip()
                i += 1
                continue
        if line.startswith("## "):
            cur = line[3:].strip().lower()
            sections.setdefault(cur, [])
            i += 1
            if cur.startswith("original goal"):
                while i < n and not lines[i].strip():        # skip blanks before fence
                    i += 1
                if i < n and lines[i].strip().startswith("```"):
                    ticks = len(re.match(r"`+", lines[i].strip()).group())
                    i += 1
                    while i < n:
                        s = lines[i].strip()
                        if s and set(s) == {"`"} and len(s) == ticks:
                            i += 1                            # consume closing fence
                            break
                        goal_lines.append(lines[i])
                        i += 1
            continue
        if cur is not None:
            sections[cur].append(line)
        i += 1

    def sect(name):
        return "\n".join(sections.get(name, [])).strip()

    def bullets(name):
        out = []
        for l in sections.get(name, []):
            s = l.strip()
            if s.startswith(("- ", "* ")):
                out.append(s[2:].strip())
        return out

    verbatim = [b[1:-1] if len(b) >= 2 and b[0] == b[-1] == '"' else b
                for b in bullets("verbatim content")]
    return Spec(
        source_language=meta.get("source-language", "unknown"),
        original_goal="\n".join(goal_lines).strip(),
        purpose=sect("purpose"),
        audience=sect("audience"),
        features=bullets("core features"),
        scope=sect("prototype scope"),
        ui_ux=sect("ui / ux") or sect("ui/ux"),
        verbatim=verbatim,
    )
