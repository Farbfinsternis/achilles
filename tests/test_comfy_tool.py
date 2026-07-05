"""
The generate_image tool renders RASTER pixels (JPG/PNG) via ComfyUI. A weak model
sometimes routes an SVG here because it "is an image" — but writing raster bytes to
a .svg would corrupt it. The tool must redirect such a call to write_file, cheaply,
before any VRAM swap or ComfyUI contact.
"""
import types

from achilles import comfy as C
from achilles.comfy import build_tool
from achilles.tools import ToolContext


def _cfg(tmp_path):
    return types.SimpleNamespace(
        comfy_url="http://127.0.0.1:8188", model="m", lms_command="lms",
        workflows_dir=str(tmp_path))


def test_svg_path_is_redirected_to_write_file(tmp_path):
    tool = build_tool(_cfg(tmp_path))
    res = tool.run({"prompt": "a minimalist logo", "path": "assets/logo.svg"},
                   None, ToolContext(tmp_path))
    assert res.startswith("ERROR")
    assert "write_file" in res
    # It must bail BEFORE trying ComfyUI (no "not reachable" noise leaked out).
    assert "not reachable" not in res


def test_svg_guard_is_case_insensitive(tmp_path):
    tool = build_tool(_cfg(tmp_path))
    res = tool.run({"prompt": "icon", "path": "ICON.SVG"}, None, ToolContext(tmp_path))
    assert res.startswith("ERROR") and "write_file" in res


def test_raster_path_does_not_hit_the_svg_guard(tmp_path):
    # A .jpg path sails past the SVG guard (it fails later for other reasons — no
    # workflow registered — but NOT with the SVG redirect).
    tool = build_tool(_cfg(tmp_path))
    res = tool.run({"prompt": "a hero photo", "path": "assets/hero.jpg"},
                   None, ToolContext(tmp_path))
    assert "SVG" not in res


# ---- prompt-engineer persona ----------------------------------------------

def _pcfg():
    return types.SimpleNamespace(temperature=0.2)


def test_engineer_prompt_enriches(monkeypatch):
    monkeypatch.setattr(C, "chat",
                        lambda cfg, msgs, **k: "warm rustic bakery, morning light, "
                                               "cinematic, highly detailed")
    out = C._engineer_prompt(_pcfg(), "a bakery", "landscape", lambda *_: None)
    assert "cinematic" in out and "bakery" in out


def test_engineer_prompt_falls_back_on_llm_error(monkeypatch):
    def boom(*a, **k):
        raise C.LLMError("model down")
    monkeypatch.setattr(C, "chat", boom)
    out = C._engineer_prompt(_pcfg(), "a bakery", "square", lambda *_: None)
    assert out == "a bakery"                       # never blocks image gen


def test_engineer_prompt_falls_back_on_empty(monkeypatch):
    monkeypatch.setattr(C, "chat", lambda *a, **k: "   ")
    out = C._engineer_prompt(_pcfg(), "raw brief", "landscape", lambda *_: None)
    assert out == "raw brief"


def test_engineer_prompt_strips_wrapping_quotes(monkeypatch):
    monkeypatch.setattr(C, "chat", lambda *a, **k: '"a quoted prompt"')
    out = C._engineer_prompt(_pcfg(), "x", "landscape", lambda *_: None)
    assert out == "a quoted prompt"


def test_run_feeds_engineered_prompt_to_workflow(tmp_path, monkeypatch):
    # The refined prompt (not the raw brief) is what gets baked into the workflow.
    tool = build_tool(_cfg(tmp_path))
    captured = {}
    monkeypatch.setattr(C, "chat", lambda *a, **k: "enriched cinematic hero, dramatic light")
    monkeypatch.setattr(C.wf, "apply",
                        lambda store, name, prompt, aspect: captured.__setitem__("prompt", prompt))
    monkeypatch.setattr(C.lmstudio, "available", lambda cmd="lms": False)  # bail after apply
    res = tool.run({"prompt": "a hero", "path": "assets/hero.jpg", "workflow": "wf1"},
                   None, ToolContext(tmp_path))
    assert captured["prompt"] == "enriched cinematic hero, dramatic light"
    assert res.startswith("ERROR")                 # stopped at the lms-availability check
