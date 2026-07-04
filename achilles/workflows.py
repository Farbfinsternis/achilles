"""
workflows.py — the named-workflow store and its one-time setup brain.

A workflow is a ComfyUI "API format" export: a JSON dict keyed by node id, each
value `{class_type, inputs, _meta}`. You `register` one under a name (once, it
persists), mark one `default`, and thereafter the model just asks for an image
and the store injects the prompt + aspect into the right nodes.

We do NOT guess which nodes hold the prompt and the resolution — guessing was a
source of *silent* wrong renders (a mis-detected slot only shows up in the image).
Instead the human MARKS the nodes in ComfyUI, by editing a node's title:

  * the PROMPT node   → title `achilles:prompt`   (the text field to fill)
  * the RESOLUTION    → title `achilles:aspect`   (optional; the aspect/size control)

The title travels in the API export as `_meta.title`, so no JSON editing is
needed. `register` becomes a VALIDATOR, not a detector: it reads the markers,
resolves each to a concrete node+field, and either accepts the workflow or
rejects it with a teaching error naming exactly what is missing. It never falls
back to guessing — a workflow the human hasn't marked simply fails to register.

Mark the node that owns the *literal* value we overwrite. If the aspect is wired
in from an upstream node, mark that upstream node (the one holding the literal) —
writing into a wired input does nothing, ComfyUI takes the wire.

What we still READ (not guess) is *how* to drive a marked node: the aspect field's
real options come from `/object_info`, and we map the model's three words
(square/landscape/portrait) onto them. Nothing here talks to LM Studio or renders;
it only reads/writes JSON and asks a ComfyClient for `/object_info`.

The one sanctioned exception to "the human marks it" is `register_adhoc`: for a
throwaway workflow handed in at run time, a MODEL nominates the two node ids from a
value-free digest of the graph, and we write the SAME markers it would have. Crucially
this changes only WHERE the markers come from — the model's guess is then fed through
the identical `validate_workflow`, so a wrong nomination is rejected with the same
teaching error, never trusted blind. That is the whole safety of the ad-hoc path.
"""

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import llm


# The three shapes the model can ask for. Each workflow maps them to its own
# dialect (an enum string for a Resolution-Selector, a width/height pair for a
# raw EmptyLatentImage) — but the model only ever says one of these words.
ASPECTS = ("square", "landscape", "portrait")

# How we recognise each aspect inside a selector's real option strings. First
# hit wins; matched case-insensitively as a substring.
_ASPECT_HINTS = {
    "square":    ("1:1", "square"),
    "landscape": ("16:9", "landscape", "widescreen"),
    "portrait":  ("9:16", "portrait"),
}

# Fallback pixel maps for a raw width/height selector, scaled from the workflow's
# own native megapixels (kept near native = VRAM- AND quality-safe).
_PIXEL_RATIOS = {"square": (1, 1), "landscape": (16, 9), "portrait": (9, 16)}


class WorkflowError(RuntimeError):
    pass


@dataclass
class ResolutionSlot:
    node: str
    field: str
    kind: str                       # "enum" | "pixels"
    mapping: dict = field(default_factory=dict)   # aspect -> value(s)


@dataclass
class Workflow:
    name: str
    prompt: Optional[list] = None            # [node_id, field] or None
    resolution: Optional[ResolutionSlot] = None


# ---- graph reading helpers ------------------------------------------------

def _is_link(v) -> bool:
    """A ComfyUI connection is `[node_id, output_index]`. Everything else in an
    inputs dict is a literal value we may set."""
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def _class(graph: dict, node_id: str) -> str:
    return graph.get(node_id, {}).get("class_type", "")


def _inputs(graph: dict, node_id: str) -> dict:
    return graph.get(node_id, {}).get("inputs", {}) or {}


# ---- the marked-node reader (aspect field + options) ----------------------

# Input names that plausibly hold an aspect/format enum, when /object_info is
# unavailable to confirm the type from the server.
_ENUM_NAME_HINTS = ("aspect_ratio", "aspect", "ratio", "format", "orientation",
                    "resolution", "size")


def _pick_enum_field(inputs: dict, cls: str, object_info: Optional[dict]):
    """Which literal input selects the aspect? Trust /object_info's type (a combo
    is a list of options) when we have it; otherwise fall back to the name hint."""
    for name, val in inputs.items():
        if _is_link(val):
            continue
        if object_info and _combo_options(object_info, cls, name):
            return name
    for name in _ENUM_NAME_HINTS:
        v = inputs.get(name)
        if v is not None and not _is_link(v) and isinstance(v, str):
            return name
    return None


def _combo_options(object_info: dict, cls: str, field_name: str) -> list:
    """Return a combo input's option list, or [] if this field isn't a combo.
    ComfyUI has TWO dialects for declaring a combo, and we accept both:
      * old: `[[opt, opt, …], {meta}]`     — the options ARE the type (entry[0]).
      * new: `["COMBO", {"options": […]}]` — the type is a name; options live in
             the meta dict (newer ComfyUI). We don't hard-require the "COMBO"
             string, just options-in-meta, to stay tolerant of type aliases."""
    spec = (object_info or {}).get(cls, {}).get("input", {})
    for group in ("required", "optional"):
        entry = spec.get(group, {}).get(field_name)
        if not entry:
            continue
        if isinstance(entry[0], list):
            return entry[0]
        if len(entry) > 1 and isinstance(entry[1], dict) \
                and isinstance(entry[1].get("options"), list):
            return entry[1]["options"]
    return []


def _map_enum(options: list, current) -> dict:
    """Map square/landscape/portrait onto the selector's real option strings by
    substring hint. Unmatched aspects are left out (the REPL can fill them in).

    Hints are tried in priority order (the numeric ratio FIRST, word hints after),
    and each hint is scanned across ALL options before moving to the next. This
    matters because a word hint can collide: the portrait label
    "9:16 (Portrait Widescreen)" contains "widescreen" (a landscape hint), so an
    option-outer/hint-inner scan would mis-map landscape onto it. Ratio-first,
    option-inner lets the unambiguous "16:9" win before "widescreen" is tried."""
    mapping = {}
    for aspect, hints in _ASPECT_HINTS.items():
        for h in hints:
            hit = next((o for o in options if h in str(o).lower()), None)
            if hit is not None:
                mapping[aspect] = hit
                break
    # WxH dimension strings (e.g. "1344x768") carry no ratio token or word, so the
    # hint pass above misses them. For any aspect still unmapped, classify options
    # by their parsed dimensions (first match per aspect wins).
    if options and len(mapping) < len(ASPECTS):
        for o in options:
            asp = _wxh_aspect(o)
            if asp and asp not in mapping:
                mapping[asp] = o
    # No option list (server was down at register time): keep the current value
    # for the single aspect it best looks like, so at least the native shape still
    # works. Ratio-first again, and assign to ONE aspect only (the strongest hit).
    if not options and isinstance(current, str):
        cur = current.lower()
        for aspect, hints in _ASPECT_HINTS.items():
            if hints[0] in cur:            # the numeric ratio is the sure signal
                mapping[aspect] = current
                break
        else:                             # no ratio token — fall back to any word
            for aspect, hints in _ASPECT_HINTS.items():
                if any(h in cur for h in hints):
                    mapping[aspect] = current
                    break
    return mapping


def _map_pixels(width: int, height: int) -> dict:
    """Reshape at constant megapixels, rounded to /64 (latent requirement)."""
    mp = width * height
    out = {}
    for aspect, (rw, rh) in _PIXEL_RATIOS.items():
        scale = (mp / (rw * rh)) ** 0.5
        out[aspect] = [_round64(rw * scale), _round64(rh * scale)]
    return out


def _round64(x: float) -> int:
    return max(64, int(round(x / 64.0)) * 64)


_WXH_RE = re.compile(r"(\d+)\s*[x×]\s*(\d+)")


def _wxh_aspect(option) -> Optional[str]:
    """Classify a "WIDTHxHEIGHT" option string (e.g. "1344x768") into one of the
    three aspects by its parsed dimensions. None if it isn't a WxH string. Ratio
    tokens like "16:9" use a colon and are handled by the hint pass, not here."""
    m = _WXH_RE.search(str(option).lower())
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    if w == h:
        return "square"
    return "landscape" if w > h else "portrait"


# ---- the persistent store -------------------------------------------------

class Store:
    """The on-disk workflow registry: `<name>.json` (the graph) + `<name>.meta.json`
    (detected slots) + a `_default` pointer. Self-provisions its directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _graph_path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def _meta_path(self, name: str) -> Path:
        return self.root / f"{name}.meta.json"

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json")
                      if not p.name.endswith(".meta.json"))

    def exists(self, name: str) -> bool:
        return self._graph_path(name).is_file()

    def load_graph(self, name: str) -> dict:
        if not self.exists(name):
            raise WorkflowError(f"no workflow named '{name}'. "
                                f"Registered: {', '.join(self.names()) or '(none)'}.")
        return json.loads(self._graph_path(name).read_text(encoding="utf-8"))

    def load_meta(self, name: str) -> Workflow:
        raw = json.loads(self._meta_path(name).read_text(encoding="utf-8"))
        res = raw.get("resolution")
        return Workflow(
            name=name,
            prompt=raw.get("prompt"),
            resolution=ResolutionSlot(**res) if res else None,
        )

    def save(self, name: str, graph: dict, wf: Workflow) -> None:
        self._graph_path(name).write_text(
            json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        meta = {"prompt": wf.prompt}
        if wf.resolution:
            meta["resolution"] = wf.resolution.__dict__
        self._meta_path(name).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def remove(self, name: str) -> None:
        for p in (self._graph_path(name), self._meta_path(name)):
            p.unlink(missing_ok=True)
        if self.get_default() == name:
            (self.root / "_default").unlink(missing_ok=True)

    def get_default(self) -> Optional[str]:
        f = self.root / "_default"
        if f.is_file():
            name = f.read_text(encoding="utf-8").strip()
            if self.exists(name):
                return name
        return None

    def set_default(self, name: str) -> None:
        if not self.exists(name):
            raise WorkflowError(f"cannot default to unknown workflow '{name}'.")
        (self.root / "_default").write_text(name, encoding="utf-8")


# ---- markers: the human's contract, read from node titles -----------------

_MARKER_PREFIX = "achilles:"

# The user writes one canonical role; we also accept a few natural synonyms and
# normalise them. An unknown role is a teaching error, never silently ignored.
_ROLE_ALIASES = {
    "prompt": "prompt",
    "aspect": "aspect",
    "aspect-ratio": "aspect",
    "aspect_ratio": "aspect",
    "resolution": "aspect",
    "size": "aspect",
}


@dataclass
class Marker:
    role: str                    # normalised: "prompt" | "aspect"
    raw_role: str                # what the user actually typed (for messages)
    node: str
    field: Optional[str]         # explicit "=field" suffix, else None


def _collect_markers(graph: dict):
    """Read every node title starting with `achilles:`. Returns (by_role, errors):
    by_role maps a normalised role to the markers found for it; errors holds
    teaching messages for unknown roles."""
    by_role: dict = {}
    errors: list = []
    for nid, node in graph.items():
        title = ((node.get("_meta") or {}).get("title") or "").strip()
        if not title.lower().startswith(_MARKER_PREFIX):
            continue
        spec = title[len(_MARKER_PREFIX):]
        raw_role, _, fld = spec.partition("=")
        raw_role = raw_role.strip().lower()
        role = _ROLE_ALIASES.get(raw_role)
        if role is None:
            errors.append(f"node {nid}: unknown marker 'achilles:{raw_role}'. "
                          "Valid markers: achilles:prompt, achilles:aspect.")
            continue
        by_role.setdefault(role, []).append(
            Marker(role, raw_role, nid, fld.strip() or None))
    return by_role, errors


# ---- validate & register (the one-time setup) & apply (per image) ----------

@dataclass
class RegisterReport:
    name: str
    prompt: Optional[list] = None
    resolution: Optional[ResolutionSlot] = None
    echo: list = field(default_factory=list)        # human-readable resolutions
    errors: list = field(default_factory=list)      # teaching rejections
    notes: list = field(default_factory=list)       # non-fatal warnings (ad-hoc)
    unsupported_aspects: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_workflow(graph: dict, object_info: Optional[dict],
                      name: str = "workflow") -> RegisterReport:
    """Resolve the workflow's markers into concrete slots, or collect teaching
    errors. Pure — no I/O, no saving. `achilles:prompt` is required;
    `achilles:aspect` is optional (its absence means the workflow renders at its
    built-in resolution and the model's aspect is ignored)."""
    rep = RegisterReport(name=name)
    by_role, marker_errs = _collect_markers(graph)
    rep.errors.extend(marker_errs)

    prompts = by_role.get("prompt", [])
    if not prompts:
        rep.errors.append(
            "required marker 'achilles:prompt' not found. In ComfyUI, rename your "
            "prompt node's title to 'achilles:prompt' and re-export (API format).")
    elif len(prompts) > 1:
        rep.errors.append("marker 'achilles:prompt' is on several nodes "
                          f"({', '.join(m.node for m in prompts)}); it must be unique.")
    else:
        _resolve_prompt(graph, prompts[0], rep)

    aspects = by_role.get("aspect", [])
    if len(aspects) > 1:
        rep.errors.append("marker 'achilles:aspect' is on several nodes "
                          f"({', '.join(m.node for m in aspects)}); it must be unique.")
    elif aspects:
        _resolve_aspect(graph, aspects[0], object_info, rep)
    else:
        rep.echo.append("achilles:aspect → (none) — renders at the workflow's "
                        "built-in resolution; the model's aspect is ignored.")

    if object_info:
        missing = sorted({_class(graph, nid) for nid in graph} - set(object_info))
        if missing:
            rep.errors.append("this ComfyUI is missing node type(s): "
                              + ", ".join(missing) + ". Install the custom nodes "
                              "or pick another workflow.")
    return rep


def _resolve_prompt(graph: dict, marker: Marker, rep: RegisterReport) -> None:
    nid, cls, ins = marker.node, _class(graph, marker.node), _inputs(graph, marker.node)
    if marker.field:
        v = ins.get(marker.field)
        if not isinstance(v, str) or _is_link(v):
            rep.errors.append(f"achilles:prompt=({marker.field}) on node {nid}: no "
                              "such literal text field.")
            return
        fld = marker.field
    else:
        text_fields = [f for f, v in ins.items()
                       if isinstance(v, str) and not _is_link(v)]
        if len(text_fields) == 1:
            fld = text_fields[0]
        elif not text_fields:
            rep.errors.append(f"achilles:prompt on node {nid} ('{cls}') has no "
                              "literal text field to hold the prompt.")
            return
        else:
            rep.errors.append(f"achilles:prompt on node {nid} ('{cls}') has several "
                              f"text fields ({', '.join(text_fields)}); pick one with "
                              "achilles:prompt=<field>.")
            return
    rep.prompt = [nid, fld]
    rep.echo.append(f"achilles:prompt → node {nid} '{cls}', field '{fld}'")


def _resolve_aspect(graph: dict, marker: Marker, object_info: Optional[dict],
                    rep: RegisterReport) -> None:
    nid, cls, ins = marker.node, _class(graph, marker.node), _inputs(graph, marker.node)
    if marker.field:
        if marker.field not in ins or _is_link(ins[marker.field]):
            rep.errors.append(f"achilles:aspect=({marker.field}) on node {nid}: no "
                              "such literal field.")
            return
        fld = marker.field
    else:
        fld = _pick_enum_field(ins, cls, object_info)

    if fld is not None and isinstance(ins.get(fld), str):
        options = _combo_options(object_info, cls, fld) if object_info else []
        mapping = _map_enum(options, ins.get(fld))
        if not mapping:
            rep.errors.append(f"achilles:aspect on node {nid}.{fld}: could not map "
                              "square/landscape/portrait onto its options "
                              f"({options or 'ComfyUI unavailable at register'}).")
            return
        slot = ResolutionSlot(nid, fld, "enum", mapping)
    elif isinstance(ins.get("width"), int) and isinstance(ins.get("height"), int):
        slot = ResolutionSlot(nid, "width", "pixels",
                              _map_pixels(ins["width"], ins["height"]))
    else:
        rep.errors.append(f"achilles:aspect on node {nid} ('{cls}'): no aspect enum "
                          "and no literal width/height ints found. If the value is "
                          "wired in, mark the upstream node that holds the literal.")
        return

    rep.resolution = slot
    rep.unsupported_aspects = [a for a in ASPECTS if a not in slot.mapping]
    rep.echo.append(f"achilles:aspect → node {nid} '{cls}', field '{slot.field}' "
                    f"({slot.kind})")
    for a in ASPECTS:
        v = slot.mapping.get(a)
        rep.echo.append(f"     {a:<9} → {v if v is not None else '— not derivable'}")


def _load_export(path: Path) -> dict:
    """Read a ComfyUI API-format export and confirm its shape (a dict of nodes,
    each carrying a class_type). Raises WorkflowError on anything else."""
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(graph, dict) or not all(
            isinstance(v, dict) and "class_type" in v for v in graph.values()):
        raise WorkflowError(f"{path} is not a ComfyUI API-format export.")
    return graph


def register(store: Store, path: Path, name: str,
             object_info: Optional[dict] = None) -> RegisterReport:
    """Validate a marked workflow and, only if it is clean, persist it. Returns a
    RegisterReport (check `.ok`): on failure nothing is saved and `.errors` carry
    teaching messages. Raises only when the file itself isn't a ComfyUI export."""
    graph = _load_export(Path(path))
    rep = validate_workflow(graph, object_info, name)
    if rep.ok:
        store.save(name, graph,
                   Workflow(name=name, prompt=rep.prompt, resolution=rep.resolution))
    return rep


# ---- ad-hoc annotation: let a model place the markers a human didn't -------
#
# The curated path needs a human to title two nodes. The ad-hoc path asks the
# MODEL to nominate those same two node ids from a value-free digest of the graph,
# then writes synthetic `achilles:` markers so the SAME validate_workflow runs
# afterwards. The model's whole job is two ids — no field resolution, no enum
# mapping (validate_workflow still owns all of that) — so a bad nomination is
# rejected by the validator exactly as a bad human marker would be.

# class_type / title fragments that mark a workflow-internal prompt EXPANDER (a
# second LLM the graph runs on the prompt). Used only to WARN about double
# expansion when the marked prompt node feeds one of these before the encoder.
_EXPANDER_HINTS = ("lmstudio", "ollama", "llm", "gpt", "qwen", "promptgen",
                   "expand", "enhance")

# Reasoning models (e.g. Ornith) emit a long <think>…</think> preamble before the
# two answer lines — a full 41-node digest already cost ~4.6k completion tokens in
# testing, and a bigger graph reasons longer. This is a CEILING, not a target: the
# model stops at its own stop token, so a high value never forces long output, it
# only stops a mid-thought truncation (which llm.chat rejects as unsafe to parse).
# Keep it well under the server's context length (prompt + reasoning + answer must
# all fit); raise it further for very large workflows.
_ANNOTATE_MAX_TOKENS = 32768

# Positive framing only. Telling a small local model what NOT to pick (the
# negative prompt, a wired node) reliably draws it toward exactly that — the white-
# elephant effect. In live testing the plain "find the node where the prompt is
# entered" landed on the right node; a "beware the negative / not a wire" variant
# drifted straight onto the negative-prompt node. So we describe the target, never
# the traps.
_ANNOTATE_SYSTEM = (
    "Analyze this ComfyUI workflow from the end to the beginning. Find the node "
    "where the prompt must be entered, and the node that configures the image "
    "format.\n"
    "Each line is one node: `id  class_type  \"title\"  input=<type|[->id]>`, where "
    "[->id] is a wire from another node and <type> is a literal value.\n"
    "Reply with exactly two lines:\n"
    "prompt: <id>\naspect: <id>\n"
    "If there is no image-format node, write `aspect: none`."
)


@dataclass
class AnnotateReport:
    graph: dict                                     # a COPY, markers written in
    prompt_id: Optional[str] = None
    aspect_id: Optional[str] = None
    notes: list = field(default_factory=list)       # warnings (e.g. double expand)
    errors: list = field(default_factory=list)      # hard failures (no usable id)

    @property
    def ok(self) -> bool:
        return not self.errors


def _node_order(graph: dict):
    """Node ids in numeric order when they are all integers, else lexical."""
    ids = list(graph)
    if ids and all(i.isdigit() for i in ids):
        return sorted(ids, key=int)
    return sorted(ids)


def _digest(graph: dict) -> str:
    """A value-free, line-per-node abridgement for the nominating model: id,
    class_type, title, and for each input ONLY whether it is a link ([->id]) or a
    literal (<type>). Prompt texts, seeds and model paths are withheld on purpose —
    noise for the "which node" question, and they blow up the context on big graphs."""
    lines = []
    for nid in _node_order(graph):
        node = graph[nid]
        cls = node.get("class_type", "")
        title = ((node.get("_meta") or {}).get("title") or "").strip()
        parts = []
        for name, v in (node.get("inputs") or {}).items():
            parts.append(f"{name}=[->{v[0]}]" if _is_link(v)
                         else f"{name}=<{type(v).__name__}>")
        head = f'{nid}  {cls}  "{title}"' if title else f"{nid}  {cls}"
        lines.append(head + ("  " + " ".join(parts) if parts else ""))
    return "\n".join(lines)


def _strip_reasoning(text: str) -> str:
    """Drop a reasoning model's <think>…</think> preamble so we parse only its
    final answer. Closed blocks only — an unclosed one means the reply was
    truncated mid-thought, which llm.chat already rejects upstream."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)


def _parse_nomination(text: str):
    """Pull `prompt: <id>` and `aspect: <id>` from the model's reply. Returns
    (prompt_id, aspect_id, ok). Tolerant of quotes and a `node ` prefix; 'none'/
    '-'/'null' for aspect means "no aspect node". ok is False when the prompt line
    is missing — we never fall back to guessing.

    Takes the LAST match per role: a reasoning model may float candidates while
    thinking ("could be node 3… no, 41"), and the final line is its verdict."""
    text = _strip_reasoning(text)
    def grab(role):
        hits = re.findall(rf'{role}\s*[:=]\s*"?(?:node\s*)?([\w-]+)"?', text or "", re.I)
        return hits[-1] if hits else None
    p, a = grab("prompt"), grab("aspect")
    if a and a.lower() in ("none", "null", "na", "-"):
        a = None
    return p, a, p is not None


def _consumers(graph: dict) -> dict:
    """node id -> list of node ids that wire in one of its outputs."""
    cons: dict = {}
    for nid, node in graph.items():
        for v in (node.get("inputs") or {}).values():
            if _is_link(v):
                cons.setdefault(v[0], []).append(nid)
    return cons


def _expander_between(graph: dict, prompt_id: str) -> Optional[str]:
    """Walking FORWARD from the marked prompt node toward its CLIPTextEncode sink,
    is a prompt-expander node in the way? Returns that node's id (for the warning)
    or None. Stops at encoders — expansion past the encoder is irrelevant."""
    cons = _consumers(graph)
    seen, stack = set(), list(cons.get(prompt_id, []))
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        cls = _class(graph, nid).lower()
        title = ((graph.get(nid, {}).get("_meta") or {}).get("title") or "").lower()
        if "cliptextencode" in cls:
            continue                       # reached the sink on this branch
        if any(h in cls or h in title for h in _EXPANDER_HINTS):
            return nid
        stack.extend(cons.get(nid, []))
    return None


def annotate(graph: dict, config, chat=llm.chat) -> AnnotateReport:
    """Ask a model to nominate the prompt and aspect node ids, then write them as
    synthetic `achilles:` title markers into a COPY of the graph. Injects nothing
    else — validate_workflow still resolves fields, options and errors. `chat` is
    injectable so tests need no server."""
    import copy
    try:
        reply = chat(
            config,
            [{"role": "system", "content": _ANNOTATE_SYSTEM},
             {"role": "user", "content": _digest(graph)}],
            temperature=0.0, max_tokens=_ANNOTATE_MAX_TOKENS)
    except llm.LLMError as e:
        return AnnotateReport(graph, errors=[f"could not analyse workflow: {e}"])

    p, a, ok = _parse_nomination(reply)
    if not ok:
        return AnnotateReport(
            graph, errors=["the model did not name a prompt node "
                           f"(reply: {(reply or '').strip()[:200]!r})."])

    out = copy.deepcopy(graph)
    rep = AnnotateReport(out, prompt_id=p, aspect_id=a)
    if p not in out:
        rep.errors.append(f"model named prompt node '{p}', not in the workflow.")
        return rep
    if a is not None and a not in out:
        rep.notes.append(f"model named aspect node '{a}', not in the workflow — "
                         "ignored (rendering at the built-in resolution).")
        a = rep.aspect_id = None

    out[p].setdefault("_meta", {})["title"] = _MARKER_PREFIX + "prompt"
    if a is not None:
        out[a].setdefault("_meta", {})["title"] = _MARKER_PREFIX + "aspect"

    exp = _expander_between(out, p)
    if exp:
        rep.notes.append(
            f"node {exp} ('{_class(out, exp)}') sits between the prompt node {p} and "
            "the encoder — your prompt gets re-expanded inside the workflow. If that "
            "is not intended, mark the node closer to the encoder instead.")
    return rep


def register_adhoc(store: Store, path: Path, name: str, config,
                   object_info: Optional[dict] = None,
                   chat=llm.chat) -> RegisterReport:
    """Like register(), but a MODEL places the markers instead of a human. Loads
    the export, annotates a copy, then runs the exact same validate_workflow — so a
    mis-nomination is rejected with the same teaching errors, never trusted blind.
    Persists only when clean. Annotation warnings ride along in `.notes`."""
    graph = _load_export(Path(path))
    ann = annotate(graph, config, chat=chat)
    if not ann.ok:
        return RegisterReport(name=name, errors=list(ann.errors), notes=list(ann.notes))
    rep = validate_workflow(ann.graph, object_info, name)
    rep.notes[0:0] = ann.notes
    if rep.ok:
        store.save(name, ann.graph,
                   Workflow(name=name, prompt=rep.prompt, resolution=rep.resolution))
    return rep


def apply(store: Store, name: str, prompt: str, aspect: str) -> dict:
    """Return a ready-to-queue graph with the prompt and aspect injected. Called
    BEFORE any VRAM is touched, so a bad aspect fails cheap while the LM is still
    loaded."""
    aspect = (aspect or "").strip().lower()
    if aspect and aspect not in ASPECTS:
        raise WorkflowError(f"unknown aspect '{aspect}'. "
                            f"Use one of: {', '.join(ASPECTS)}.")
    graph = store.load_graph(name)
    wf = store.load_meta(name)

    if wf.prompt:
        node, fld = wf.prompt
        graph[node]["inputs"][fld] = prompt

    if aspect and wf.resolution:
        r = wf.resolution
        value = r.mapping.get(aspect)
        if value is None:
            raise WorkflowError(
                f"workflow '{name}' has no '{aspect}' mapping "
                f"(known: {', '.join(r.mapping) or 'none'}).")
        if r.kind == "enum":
            graph[r.node]["inputs"][r.field] = value
        else:  # pixels: value is [w, h]
            graph[r.node]["inputs"]["width"] = value[0]
            graph[r.node]["inputs"]["height"] = value[1]
    return graph
