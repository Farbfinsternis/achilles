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
"""

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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


def register(store: Store, path: Path, name: str,
             object_info: Optional[dict] = None) -> RegisterReport:
    """Validate a marked workflow and, only if it is clean, persist it. Returns a
    RegisterReport (check `.ok`): on failure nothing is saved and `.errors` carry
    teaching messages. Raises only when the file itself isn't a ComfyUI export."""
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(graph, dict) or not all(
            isinstance(v, dict) and "class_type" in v for v in graph.values()):
        raise WorkflowError(f"{path} is not a ComfyUI API-format export.")

    rep = validate_workflow(graph, object_info, name)
    if rep.ok:
        store.save(name, graph,
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
