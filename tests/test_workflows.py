"""Tests for workflows.py — the marked-node contract and the aspect mapping.

Achilles no longer *guesses* which nodes hold the prompt and resolution; the human
marks them by node title (achilles:prompt / achilles:aspect) and register()
VALIDATES. These tests pin the validator (accept/reject + teaching errors) and the
mapping logic that turns square/landscape/portrait into a selector's real options.
"""

import json

import pytest

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

# WxH option strings (e.g. CM_SDXLResolution) — no ratio token, no words.
WXH_OPTIONS = ["1024x1024", "1152x896", "896x1152", "1344x768", "768x1344"]


def _combo(options):
    return ["COMBO", {"default": options[0], "options": options}]


# A Krea-2-shaped graph, MARKED: node 4 is the prompt, node 22 is the aspect.
KREA_GRAPH = {
    "4":  {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]},
           "_meta": {"title": "achilles:prompt"}},
    "8":  {"class_type": "EmptyLatentImage",
           "inputs": {"width": ["22", 0], "height": ["22", 1], "batch_size": 1},
           "_meta": {"title": "Empty Latent Image"}},
    "11": {"class_type": "KSampler",
           "inputs": {"positive": ["56", 0], "negative": ["16", 0],
                      "latent_image": ["8", 0]},
           "_meta": {"title": "KSampler"}},
    "16": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]},
           "_meta": {"title": "ConditioningZeroOut"}},
    "22": {"class_type": "ResolutionSelector",
           "inputs": {"aspect_ratio": "16:9 (Widescreen)", "megapixels": 2,
                      "multiple": 8},
           "_meta": {"title": "achilles:aspect"}},
    "56": {"class_type": "ConditioningKrea2Rebalance",
           "inputs": {"conditioning": ["4", 0]},
           "_meta": {"title": "Krea 2 Control"}},
}


def _object_info_for(graph, resolution_options=KREA_ASPECTS):
    """Build an /object_info that declares every class in `graph` (so the missing-
    node check passes) and gives ResolutionSelector its aspect_ratio combo."""
    oi = {n["class_type"]: {"input": {"required": {}}} for n in graph.values()}
    if "ResolutionSelector" in oi:
        oi["ResolutionSelector"]["input"]["required"]["aspect_ratio"] = \
            _combo(resolution_options)
    return oi


KREA_OI = _object_info_for(KREA_GRAPH)


# ---- _combo_options -------------------------------------------------------

def test_combo_options_new_format():
    oi = {"R": {"input": {"required": {"aspect_ratio": _combo(KREA_ASPECTS)}}}}
    assert wf._combo_options(oi, "R", "aspect_ratio") == KREA_ASPECTS


def test_combo_options_old_format():
    oi = {"R": {"input": {"required": {"aspect_ratio": [KREA_ASPECTS, {}]}}}}
    assert wf._combo_options(oi, "R", "aspect_ratio") == KREA_ASPECTS


def test_combo_options_absent():
    assert wf._combo_options(KREA_OI, "ResolutionSelector", "megapixels") == []
    assert wf._combo_options(KREA_OI, "Nope", "aspect_ratio") == []


# ---- _map_enum ------------------------------------------------------------

def test_map_enum_ratio_beats_widescreen_collision():
    m = wf._map_enum(KREA_ASPECTS, "9:16 (Portrait Widescreen)")
    assert m["square"] == "1:1 (Square)"
    assert m["landscape"] == "16:9 (Widescreen)"        # NOT the portrait label
    assert m["portrait"] == "9:16 (Portrait Widescreen)"


def test_map_enum_wxh_options():
    m = wf._map_enum(WXH_OPTIONS, "1024x1024")
    assert m["square"] == "1024x1024"
    assert m["landscape"] in ("1152x896", "1344x768")   # first W>H wins
    assert m["portrait"] in ("896x1152", "768x1344")    # first W<H wins
    assert m["landscape"] == "1152x896"                 # deterministic: first match
    assert m["portrait"] == "896x1152"


def test_map_enum_fallback_single_aspect_no_collision():
    m = wf._map_enum([], "9:16 (Portrait Widescreen)")
    assert m == {"portrait": "9:16 (Portrait Widescreen)"}


# ---- validate_workflow: happy path ----------------------------------------

def test_validate_accepts_marked_krea():
    rep = wf.validate_workflow(KREA_GRAPH, KREA_OI, "krea")
    assert rep.ok, rep.errors
    assert rep.prompt == ["4", "text"]
    assert rep.resolution.node == "22"
    assert rep.resolution.field == "aspect_ratio"
    assert rep.resolution.kind == "enum"
    assert rep.resolution.mapping["landscape"] == "16:9 (Widescreen)"
    assert rep.resolution.mapping["portrait"] == "9:16 (Portrait Widescreen)"
    assert rep.resolution.mapping["square"] == "1:1 (Square)"
    assert rep.unsupported_aspects == []


# ---- validate_workflow: rejections ----------------------------------------

def _unmark(graph, nid, title):
    g = json.loads(json.dumps(graph))       # deep copy
    g[nid]["_meta"]["title"] = title
    return g


def test_validate_missing_prompt_marker():
    g = _unmark(KREA_GRAPH, "4", "CLIP Text Encode")
    rep = wf.validate_workflow(g, _object_info_for(KREA_GRAPH), "krea")
    assert not rep.ok
    assert any("achilles:prompt" in e for e in rep.errors)


def test_validate_duplicate_prompt_marker():
    g = _unmark(KREA_GRAPH, "16", "achilles:prompt")   # a second prompt marker
    rep = wf.validate_workflow(g, KREA_OI, "krea")
    assert not rep.ok
    assert any("several nodes" in e and "prompt" in e for e in rep.errors)


def test_validate_unknown_marker():
    g = _unmark(KREA_GRAPH, "8", "achilles:megapixel")
    rep = wf.validate_workflow(g, KREA_OI, "krea")
    assert not rep.ok
    assert any("unknown marker 'achilles:megapixel'" in e for e in rep.errors)


def test_validate_missing_aspect_is_allowed():
    g = _unmark(KREA_GRAPH, "22", "Resolution Selector")
    rep = wf.validate_workflow(g, _object_info_for(KREA_GRAPH), "krea")
    assert rep.ok, rep.errors
    assert rep.resolution is None
    assert any("built-in resolution" in line for line in rep.echo)


def test_validate_ambiguous_prompt_field_needs_suffix():
    g = _unmark(KREA_GRAPH, "4", "achilles:prompt")
    g["4"]["inputs"] = {"text": "", "text_g": "", "clip": ["2", 0]}   # two text fields
    rep = wf.validate_workflow(g, KREA_OI, "krea")
    assert not rep.ok
    assert any("pick one with achilles:prompt=<field>" in e for e in rep.errors)
    # ...and the suffix resolves it:
    g["4"]["_meta"]["title"] = "achilles:prompt=text_g"
    rep2 = wf.validate_workflow(g, KREA_OI, "krea")
    assert rep2.ok, rep2.errors
    assert rep2.prompt == ["4", "text_g"]


def test_validate_aspect_pixels_from_raw_ints():
    graph = {
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": ""},
              "_meta": {"title": "achilles:prompt"}},
        "7": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": 1920, "height": 1088, "batch_size": 1},
              "_meta": {"title": "achilles:aspect"}},
    }
    rep = wf.validate_workflow(graph, _object_info_for(graph), "z")
    assert rep.ok, rep.errors
    assert rep.resolution.kind == "pixels"
    assert rep.resolution.mapping["landscape"] == [1920, 1088]
    assert rep.resolution.mapping["portrait"] == [1088, 1920]


def test_validate_aspect_wired_reports_upstream():
    graph = {
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": ""},
              "_meta": {"title": "achilles:prompt"}},
        "7": {"class_type": "EmptyLatentImage",
              "inputs": {"width": ["9", 0], "height": ["9", 1]},
              "_meta": {"title": "achilles:aspect"}},   # marked on the wrong (wired) node
    }
    rep = wf.validate_workflow(graph, _object_info_for(graph), "bad")
    assert not rep.ok
    assert any("upstream node" in e for e in rep.errors)


# ---- register: persistence gate -------------------------------------------

def test_register_saves_only_when_valid(tmp_path):
    store = wf.Store(tmp_path / "store")     # keep source files out of the store dir
    src = tmp_path / "src"
    src.mkdir()
    good = src / "good.json"
    good.write_text(json.dumps(KREA_GRAPH), encoding="utf-8")
    rep = wf.register(store, good, "krea", KREA_OI)
    assert rep.ok
    assert store.names() == ["krea"]

    bad = src / "bad.json"
    bad.write_text(json.dumps(_unmark(KREA_GRAPH, "4", "no marker")), encoding="utf-8")
    rep2 = wf.register(store, bad, "nope", _object_info_for(KREA_GRAPH))
    assert not rep2.ok
    assert "nope" not in store.names()          # not persisted


def test_register_rejects_non_api_file(tmp_path):
    store = wf.Store(tmp_path / "store")
    junk = tmp_path / "junk.json"
    junk.write_text('["not", "a", "graph"]', encoding="utf-8")
    with pytest.raises(wf.WorkflowError):
        wf.register(store, junk, "junk", None)


# ---- apply ----------------------------------------------------------------

def test_apply_injects_prompt_and_aspect(tmp_path):
    store = wf.Store(tmp_path)
    rep = wf.validate_workflow(KREA_GRAPH, KREA_OI, "krea")
    store.save("krea", KREA_GRAPH,
               wf.Workflow(name="krea", prompt=rep.prompt, resolution=rep.resolution))
    graph = wf.apply(store, "krea", "a warm bakery", "landscape")
    assert graph["4"]["inputs"]["text"] == "a warm bakery"
    assert graph["22"]["inputs"]["aspect_ratio"] == "16:9 (Widescreen)"


# ---- ad-hoc annotation ----------------------------------------------------
#
# A faithful mini-Standard.json: the raw prompt in 41 flows through the LMStudio
# expander 40, a PreviewAny pass-through 39, and only THEN into the CLIPTextEncode
# 3 (whose `text` is WIRED, not a literal). This is the exact shape where Ornith
# once claimed a direct 40->3 link; the validator must reject a nomination of 3.
STD_GRAPH = {
    "3":  {"class_type": "CLIPTextEncode",
           "inputs": {"text": ["39", 0], "clip": ["4", 1]},
           "_meta": {"title": "CLIP Text Encode (Prompt)"}},
    "4":  {"class_type": "Power Lora Loader (rgthree)",
           "inputs": {"model": ["1", 0], "clip": ["2", 0]},
           "_meta": {"title": "Power Lora Loader"}},
    "39": {"class_type": "PreviewAny", "inputs": {"source": ["40", 0]},
           "_meta": {"title": "Preview as Text"}},
    "40": {"class_type": "> LMStudio",
           "inputs": {"prompt": ["41", 0], "system_message": "expand it"},
           "_meta": {"title": "LMStudio"}},
    "41": {"class_type": "PrimitiveStringMultiline",
           "inputs": {"value": "a woman in a batman costume"},
           "_meta": {"title": "Text String (Multiline)"}},
    "21": {"class_type": "CM_SDXLResolution", "inputs": {"resolution": "768x1344"},
           "_meta": {"title": "SDXLResolution"}},
    "24": {"class_type": "EmptyLatentImage",
           "inputs": {"width": ["21", 0], "height": ["21", 1], "batch_size": 1},
           "_meta": {"title": "Empty Latent Image"}},
}


def _std_oi():
    oi = {n["class_type"]: {"input": {"required": {}}} for n in STD_GRAPH.values()}
    oi["CM_SDXLResolution"]["input"]["required"]["resolution"] = _combo(WXH_OPTIONS)
    return oi


def _chat(reply):
    """A stub llm.chat that always returns `reply` (no server, no config needed)."""
    def f(config, messages, temperature=0.0, max_tokens=200):
        return reply
    return f


def test_digest_withholds_values_but_keeps_wiring():
    d = wf._digest(STD_GRAPH)
    assert "41  PrimitiveStringMultiline" in d
    assert "value=<str>" in d                       # the field, as a type…
    assert "batman" not in d                         # …never the literal text
    assert "768x1344" not in d                       # nor the resolution value
    assert "text=[->39]" in d                        # links ARE shown (the chain)


def test_parse_nomination_variants():
    assert wf._parse_nomination("prompt: 41\naspect: 21") == ("41", "21", True)
    assert wf._parse_nomination("prompt: node 41\naspect: none") == ("41", None, True)
    assert wf._parse_nomination("no idea") == (None, None, False)


def test_parse_nomination_strips_reasoning_and_takes_verdict():
    # A reasoning model floats a wrong candidate while thinking, then commits in the
    # final lines. The <think> block is stripped and the LAST match wins.
    reply = ("<think>the title says Prompt on node 3, maybe prompt: 3? no, its text "
             "is wired, follow it back to 41</think>\nprompt: 41\naspect: 21")
    assert wf._parse_nomination(reply) == ("41", "21", True)


def test_annotate_marks_nominated_nodes_and_copies():
    rep = wf.annotate(STD_GRAPH, None, chat=_chat("prompt: 41\naspect: 21"))
    assert rep.ok
    assert rep.graph["41"]["_meta"]["title"] == "achilles:prompt"
    assert rep.graph["21"]["_meta"]["title"] == "achilles:aspect"
    # the input graph is untouched (annotate works on a copy)
    assert STD_GRAPH["41"]["_meta"]["title"] == "Text String (Multiline)"


def test_annotate_warns_about_double_expansion():
    rep = wf.annotate(STD_GRAPH, None, chat=_chat("prompt: 41\naspect: 21"))
    assert any("re-expanded" in n and "40" in n for n in rep.notes)


def test_annotate_rejects_id_not_in_graph():
    rep = wf.annotate(STD_GRAPH, None, chat=_chat("prompt: 999\naspect: none"))
    assert not rep.ok
    assert any("not in the workflow" in e for e in rep.errors)


def _write(tmp_path, graph):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    p = src / "wf.json"
    p.write_text(json.dumps(graph), encoding="utf-8")
    return p


def test_register_adhoc_end_to_end(tmp_path):
    store = wf.Store(tmp_path / "store")
    path = _write(tmp_path, STD_GRAPH)
    rep = wf.register_adhoc(store, path, "_adhoc", None, _std_oi(),
                            chat=_chat("prompt: 41\naspect: 21"))
    assert rep.ok, rep.errors
    assert "_adhoc" in store.names()
    graph = wf.apply(store, "_adhoc", "hello world", "portrait")
    assert graph["41"]["inputs"]["value"] == "hello world"     # raw prompt node
    assert graph["21"]["inputs"]["resolution"] == "896x1152"   # portrait WxH


def test_register_adhoc_rejects_wired_encoder(tmp_path):
    """The 40->3 regression: nominating the CLIPTextEncode (whose text is wired)
    must be rejected by the SAME validator, not silently accepted."""
    store = wf.Store(tmp_path / "store")
    path = _write(tmp_path, STD_GRAPH)
    rep = wf.register_adhoc(store, path, "_adhoc", None, _std_oi(),
                            chat=_chat("prompt: 3\naspect: 21"))
    assert not rep.ok
    assert any("no literal text field" in e for e in rep.errors)
    assert "_adhoc" not in store.names()                       # nothing persisted


def test_register_adhoc_rejects_unparseable_reply(tmp_path):
    store = wf.Store(tmp_path / "store")
    path = _write(tmp_path, STD_GRAPH)
    rep = wf.register_adhoc(store, path, "_adhoc", None, _std_oi(),
                            chat=_chat("I think it lives somewhere in the graph"))
    assert not rep.ok
    assert any("did not name a prompt node" in e for e in rep.errors)


# ---- find_workflow_path / strip_workflow_path (dropped-into-goal) ----------

def test_find_workflow_path_detects_export(tmp_path):
    p = _write(tmp_path, STD_GRAPH)          # a real ComfyUI export
    goal = f'baue eine landingpage, verwende diesen workflow "{p}" für die bilder'
    raw, resolved = wf.find_workflow_path(goal)
    assert resolved == str(p)
    assert raw == f'"{p}"'                    # the exact substring, quotes included
    # bare (unquoted) path is found too
    raw2, resolved2 = wf.find_workflow_path(f"nutze {p} bitte")
    assert resolved2 == str(p)


def test_find_workflow_path_ignores_non_workflow_json(tmp_path):
    junk = tmp_path / "config.json"
    junk.write_text('{"theme": "dark", "port": 8080}', encoding="utf-8")
    assert wf.find_workflow_path(f"read the settings in {junk} first") is None


def test_find_workflow_path_none_when_absent():
    assert wf.find_workflow_path("just build me a website, no files here") is None
    assert wf.find_workflow_path("use /does/not/exist/wf.json please") is None


def test_strip_workflow_path_tidies_goal(tmp_path):
    p = _write(tmp_path, STD_GRAPH)
    goal = f'landingpage bauen, verwende diesen workflow "{p}" für die bilder'
    raw, _ = wf.find_workflow_path(goal)
    cleaned = wf.strip_workflow_path(goal, raw)
    assert str(p) not in cleaned
    assert '"' not in cleaned                 # the quotes went with it
    assert cleaned == "landingpage bauen, verwende diesen workflow für die bilder"
