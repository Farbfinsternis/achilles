"""Tests for workflows.py — the topology brain that finds a workflow's prompt and
resolution slots and maps square/landscape/portrait onto a selector's real options.

These pin two bugs found against a real Krea-2 export + a live ComfyUI:
  * /object_info now declares a combo as ["COMBO", {"options": [...]}] (new format),
    not the old [[opt, ...], {meta}] — _combo_options must read both.
  * a word hint can collide: the portrait label "9:16 (Portrait Widescreen)" contains
    "widescreen" (a landscape hint), so the ratio token must win first.
"""

from achilles import workflows as wf


# The aspect_ratio options a live ComfyUI ResolutionSelector reports, in order.
KREA_ASPECTS = [
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",   # contains "widescreen" — the collision trap
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
]

# A ResolutionSelector node type in the NEW /object_info combo dialect.
OBJECT_INFO_NEW = {
    "ResolutionSelector": {
        "input": {"required": {
            "aspect_ratio": ["COMBO", {"default": "1:1 (Square)",
                                       "options": KREA_ASPECTS}],
            "megapixels": ["FLOAT", {"default": 1.0}],
        }}
    }
}

# Same node in the OLD dialect (options ARE the type).
OBJECT_INFO_OLD = {
    "ResolutionSelector": {
        "input": {"required": {
            "aspect_ratio": [KREA_ASPECTS, {"default": "1:1 (Square)"}],
        }}
    }
}

# A Krea-2-shaped graph: the positive prompt reaches the CLIPTextEncode THROUGH an
# intermediate conditioning node (56), and width/height are wired to a selector (22).
KREA_GRAPH = {
    "4":  {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
    "8":  {"class_type": "EmptyLatentImage",
           "inputs": {"width": ["22", 0], "height": ["22", 1], "batch_size": 1}},
    "11": {"class_type": "KSampler",
           "inputs": {"positive": ["56", 0], "negative": ["16", 0],
                      "latent_image": ["8", 0]}},
    "16": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
    "22": {"class_type": "ResolutionSelector",
           "inputs": {"aspect_ratio": "9:16 (Portrait Widescreen)", "megapixels": 1}},
    "56": {"class_type": "ConditioningKrea2Rebalance",
           "inputs": {"conditioning": ["4", 0]}},
}


def test_combo_options_new_format():
    assert wf._combo_options(OBJECT_INFO_NEW, "ResolutionSelector",
                             "aspect_ratio") == KREA_ASPECTS


def test_combo_options_old_format():
    assert wf._combo_options(OBJECT_INFO_OLD, "ResolutionSelector",
                             "aspect_ratio") == KREA_ASPECTS


def test_combo_options_absent_field():
    assert wf._combo_options(OBJECT_INFO_NEW, "ResolutionSelector", "megapixels") == []
    assert wf._combo_options(OBJECT_INFO_NEW, "Nope", "aspect_ratio") == []


def test_map_enum_ratio_beats_widescreen_collision():
    m = wf._map_enum(KREA_ASPECTS, "9:16 (Portrait Widescreen)")
    assert m["square"] == "1:1 (Square)"
    assert m["landscape"] == "16:9 (Widescreen)"        # NOT the portrait label
    assert m["portrait"] == "9:16 (Portrait Widescreen)"


def test_map_enum_fallback_single_aspect_no_collision():
    # Options empty (server down at register): current is a portrait value, so ONLY
    # portrait maps — landscape must not get it via the "widescreen" word.
    m = wf._map_enum([], "9:16 (Portrait Widescreen)")
    assert m == {"portrait": "9:16 (Portrait Widescreen)"}


def test_find_prompt_slot_through_intermediate_node():
    # positive → 56 → 4; must trace past the rebalance node to the literal text.
    assert wf.find_prompt_slot(KREA_GRAPH) == ["4", "text"]


def test_find_resolution_slot_end_to_end():
    slot = wf.find_resolution_slot(KREA_GRAPH, OBJECT_INFO_NEW)
    assert slot is not None
    assert slot.node == "22"
    assert slot.field == "aspect_ratio"
    assert slot.kind == "enum"
    assert slot.mapping["landscape"] == "16:9 (Widescreen)"
    assert slot.mapping["portrait"] == "9:16 (Portrait Widescreen)"
    assert slot.mapping["square"] == "1:1 (Square)"


def test_apply_injects_prompt_and_aspect(tmp_path):
    store = wf.Store(tmp_path)
    store.save("krea", KREA_GRAPH, wf.Workflow(
        name="krea", prompt=["4", "text"],
        resolution=wf.find_resolution_slot(KREA_GRAPH, OBJECT_INFO_NEW)))
    graph = wf.apply(store, "krea", "a warm bakery", "landscape")
    assert graph["4"]["inputs"]["text"] == "a warm bakery"
    assert graph["22"]["inputs"]["aspect_ratio"] == "16:9 (Widescreen)"
