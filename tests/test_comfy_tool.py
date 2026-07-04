"""
The generate_image tool renders RASTER pixels (JPG/PNG) via ComfyUI. A weak model
sometimes routes an SVG here because it "is an image" — but writing raster bytes to
a .svg would corrupt it. The tool must redirect such a call to write_file, cheaply,
before any VRAM swap or ComfyUI contact.
"""
import types

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
