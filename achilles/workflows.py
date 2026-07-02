"""
workflows.py — the named-workflow store and its one-time setup brain.

A workflow is a ComfyUI "API format" export: a JSON dict keyed by node id, each
value `{class_type, inputs, _meta}`. You `register` one under a name (once, it
persists), mark one `default`, and thereafter the model just asks for an image
and the store injects the prompt + aspect into the right nodes.

The hard part is *finding* the right nodes, and the insight is that we do NOT
read the graph semantically the way a person does — we follow its WIRING, because
ComfyUI encodes meaning in the connections:

  * the PROMPT node  = trace back from the sampler's `positive` input until we
                       reach a node with a *literal* `text` field.
  * the RESOLUTION   = trace back from the latent's width/height until we reach
                       the node that holds the *literal* value (a combo like
                       "16:9 (Widescreen)", or raw width/height ints).

A slot is only ever a *literal* input. A wired input (`["12", 0]`) is a
connection, so we follow it to the node upstream that owns the real value. That
one rule is what lets a dumb algorithm land where a human would.

Nothing here talks to LM Studio or renders; it only reads/writes JSON and asks a
ComfyClient for `/object_info` (to verify custom nodes and read a combo's real
options). Human confirmation of the detected slots happens in the REPL, not here.
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


def _trace_back(graph: dict, start_node: str, predicate) -> Optional[str]:
    """Walk backward from start_node over wired inputs until a node satisfies
    `predicate(node_id)`. Breadth-first, cycle-guarded. Returns the node id."""
    seen, queue = set(), [start_node]
    while queue:
        nid = queue.pop(0)
        if nid in seen or nid not in graph:
            continue
        seen.add(nid)
        if predicate(nid):
            return nid
        for v in _inputs(graph, nid).values():
            if _is_link(v):
                queue.append(v[0])
    return None


# ---- slot detection (the topology brain) ----------------------------------

def find_prompt_slot(graph: dict) -> Optional[list]:
    """The positive-prompt text field. Find the sampler (a node with a `positive`
    input), then trace back from that connection to the first node carrying a
    literal `text` — that is the CLIPTextEncode the prompt lives in."""
    sampler = next((nid for nid in graph if "positive" in _inputs(graph, nid)), None)
    if not sampler:
        return None
    pos = _inputs(graph, sampler).get("positive")
    if not _is_link(pos):
        return None

    def has_literal_text(nid: str) -> bool:
        t = _inputs(graph, nid).get("text")
        return t is not None and not _is_link(t) and isinstance(t, str)

    node = _trace_back(graph, pos[0], has_literal_text)
    return [node, "text"] if node else None


def find_resolution_slot(graph: dict, object_info: Optional[dict] = None) -> Optional[ResolutionSlot]:
    """The width/height control. Find the latent node (…LatentImage, or any node
    with width+height inputs); if its dimensions are wired, follow them to the
    upstream node that holds the literal — a combo (enum) selector or raw ints."""
    latent = next(
        (nid for nid in graph
         if _class(graph, nid).endswith("LatentImage")
         or {"width", "height"} <= set(_inputs(graph, nid))),
        None)
    if not latent:
        return None

    # The controllable node is wherever the literal dimensions actually live.
    w = _inputs(graph, latent).get("width")
    control = w[0] if _is_link(w) else latent
    if control not in graph:
        return None

    ins = _inputs(graph, control)
    cls = _class(graph, control)

    # Preferred: a combo/enum input whose literal string picks the aspect.
    enum_field = _pick_enum_field(ins, cls, object_info)
    if enum_field:
        options = _combo_options(object_info, cls, enum_field) if object_info else []
        return ResolutionSlot(control, enum_field, "enum",
                              _map_enum(options, ins.get(enum_field)))

    # Fallback: raw literal width+height ints on the control node.
    if isinstance(ins.get("width"), int) and isinstance(ins.get("height"), int):
        return ResolutionSlot(control, "width", "pixels",
                              _map_pixels(ins["width"], ins["height"]))
    return None


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
    """A combo input in /object_info is declared as `[[opt, opt, …], {meta}]`.
    Return the option list, or [] if this field isn't a combo."""
    spec = (object_info or {}).get(cls, {}).get("input", {})
    for group in ("required", "optional"):
        entry = spec.get(group, {}).get(field_name)
        if entry and isinstance(entry[0], list):
            return entry[0]
    return []


def _map_enum(options: list, current) -> dict:
    """Map square/landscape/portrait onto the selector's real option strings by
    substring hint. Unmatched aspects are left out (the REPL can fill them in)."""
    mapping = {}
    for aspect, hints in _ASPECT_HINTS.items():
        for opt in options:
            if any(h in str(opt).lower() for h in hints):
                mapping[aspect] = opt
                break
    # No option list (server was down at register time): keep the current value
    # for whatever aspect it looks like, so at least the native shape still works.
    if not options and isinstance(current, str):
        for aspect, hints in _ASPECT_HINTS.items():
            if any(h in current.lower() for h in hints):
                mapping[aspect] = current
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


# ---- register (the one-time setup) & apply (per image) ---------------------

@dataclass
class RegisterReport:
    name: str
    prompt: Optional[list]
    resolution: Optional[ResolutionSlot]
    missing_nodes: list = field(default_factory=list)


def register(store: Store, path: Path, name: str,
             object_info: Optional[dict] = None) -> RegisterReport:
    """Parse a workflow, verify its custom nodes exist, detect the prompt and
    resolution slots, and persist it. Raises if custom nodes are missing (abort
    before saving) — that is the portability check. Missing a resolution slot is
    NOT fatal: the workflow still registers and just ignores `aspect`."""
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(graph, dict) or not all(
            isinstance(v, dict) and "class_type" in v for v in graph.values()):
        raise WorkflowError(f"{path} is not a ComfyUI API-format export.")

    missing = []
    if object_info:
        present = set(object_info.keys())
        missing = sorted({_class(graph, nid) for nid in graph} - present)
        if missing:
            raise WorkflowError(
                "workflow needs node type(s) not installed on this ComfyUI: "
                + ", ".join(missing) + ". Install the custom nodes or pick "
                "another workflow. Not saved.")

    wf = Workflow(
        name=name,
        prompt=find_prompt_slot(graph),
        resolution=find_resolution_slot(graph, object_info),
    )
    store.save(name, graph, wf)
    return RegisterReport(name, wf.prompt, wf.resolution, missing)


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
