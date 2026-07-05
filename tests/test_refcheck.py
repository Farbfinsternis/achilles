"""
oracles/refcheck.py — the deterministic cross-file reference oracle. It flags a JS
lookup (getElementById / getElementsByClassName / a simple querySelector) whose id
or class is defined NOWHERE. The overriding requirement is NO FALSE REDS: Tailwind
utilities, runtime-added classes and compound/dynamic selectors must never flag.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "refcheck", Path(__file__).resolve().parent.parent / "oracles" / "refcheck.py")
refcheck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refcheck)


def _site(tmp_path, html="", css="", js=""):
    (tmp_path / "index.html").write_text(html, encoding="utf-8")
    if css:
        (tmp_path / "styles.css").write_text(css, encoding="utf-8")
    if js:
        (tmp_path / "script.js").write_text(js, encoding="utf-8")
    return refcheck.check_refs(tmp_path)


# ---- true positives (real dangling references) ----------------------------

def test_flags_dangling_class_query(tmp_path):
    p = _site(tmp_path,
              html='<div class="card"></div>',
              js="document.querySelectorAll('.step-item').forEach(x=>{});")
    assert len(p) == 1 and "step-item" in p[0]


def test_flags_missing_get_by_id(tmp_path):
    p = _site(tmp_path, html='<div id="hero"></div>',
              js="const el = document.getElementById('heroo');")
    assert len(p) == 1 and "heroo" in p[0]


# ---- no false reds (the property that matters) ----------------------------

def test_valid_class_query_is_clean(tmp_path):
    p = _site(tmp_path, html='<div class="card"></div>',
              js="document.querySelector('.card');")
    assert p == []


def test_class_defined_only_in_css_is_clean(tmp_path):
    p = _site(tmp_path, html="<div></div>", css=".badge { color: red; }",
              js="document.querySelector('.badge');")
    assert p == []


def test_runtime_added_class_is_clean(tmp_path):
    # A class the JS adds itself (never in the static HTML) must not flag.
    p = _site(tmp_path, html='<section class="reveal"></section>',
              js="el.classList.add('is-visible'); document.querySelectorAll('.is-visible');")
    assert p == []


def test_tailwind_utility_class_is_clean(tmp_path):
    # A utility class present in the HTML class list counts as defined.
    p = _site(tmp_path, html='<nav class="hidden md:flex"></nav>',
              js="document.querySelector('.hidden').classList.toggle('hidden');")
    assert p == []


def test_compound_and_dynamic_selectors_are_skipped(tmp_path):
    # Compound selectors and non-literal args are never flagged (avoids false reds).
    js = ("document.querySelector('.card .title');"
          "document.querySelector('.btn:hover');"
          "document.querySelector('[data-role]');"
          "document.querySelector(theSelector);")
    p = _site(tmp_path, html='<div class="card"></div>', js=js)
    assert p == []


def test_id_assigned_in_js_is_clean(tmp_path):
    p = _site(tmp_path, html="<div></div>",
              js="const d=document.createElement('div'); d.id='panel'; "
                 "document.getElementById('panel');")
    assert p == []


def test_inline_script_and_style_are_read(tmp_path):
    # A class defined in an inline <style> and queried from an inline <script>.
    html = ('<style>.glow{filter:blur(2px);}</style>'
            '<div class="glow"></div>'
            '<script>document.querySelector(".glow");</script>')
    assert _site(tmp_path, html=html) == []


def test_non_web_project_is_green(tmp_path):
    # No HTML at all → nothing to check, never a red.
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    assert refcheck.check_refs(tmp_path) == []


def test_main_returns_exit_codes(tmp_path):
    (tmp_path / "index.html").write_text('<div class="card"></div>', encoding="utf-8")
    (tmp_path / "script.js").write_text("document.querySelector('.ghost');", encoding="utf-8")
    assert refcheck.main(["refcheck", str(tmp_path)]) == 1        # dangling → red
    (tmp_path / "script.js").write_text("document.querySelector('.card');", encoding="utf-8")
    assert refcheck.main(["refcheck", str(tmp_path)]) == 0        # resolves → green
