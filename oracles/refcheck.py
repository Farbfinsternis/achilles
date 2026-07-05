#!/usr/bin/env python3
"""
refcheck.py — a REUSABLE oracle for CROSS-FILE identifier consistency (JS ↔ HTML/CSS).

Point Achilles at it as the verify_command for a web task:

    --verify "python /path/to/achilles/oracles/refcheck.py ."

`static_site.py` already checks the FILE layer (every href/src resolves, every
#anchor has an id). This one checks the IDENTIFIER layer that a weak model keeps
drifting on: a script that queries a class or id which no HTML/CSS/JS ever
defines — the ".step-item" typo, the `getElementById('hero')` for an element that
was renamed. Those "look fine per file" and only break when the files must agree.

It is deliberately CONSERVATIVE — a false red is worse than a miss, because it
fails correct work and spins the fix loop. So it only flags a JS lookup whose
identifier appears NOWHERE as a definition, and it counts a very liberal set of
definitions:

  * ids      — HTML id="…", CSS #id, JS `.id = "…"` / setAttribute('id',…), and
               ids embedded in JS-built HTML strings.
  * classes  — HTML class="…", CSS .class, JS classList.add/toggle/replace,
               className="…", and classes in JS-built HTML strings.

Only clear string-literal lookups are checked — getElementById, getElementsByClassName,
and querySelector/All with a SINGLE simple `#id` or `.class` selector. Anything
dynamic (a variable, a template literal) or compound (`.a .b`, `.x:hover`,
`[data-y]`) is skipped, so Tailwind utilities, runtime-added classes and complex
selectors never produce a false red.

stdlib only, in keeping with Achilles' zero-dependency rule.
Exit 0 = green (every JS lookup resolves); exit 1 = red (dangling lookups listed).

Usage:
    python refcheck.py [directory]     # default: current directory
"""

import re
import sys
from pathlib import Path


# ---- JS lookups we hold to account (string literals only) -----------------
_GET_BY_ID = re.compile(r"getElementById\(\s*['\"]([\w-]+)['\"]")
_GET_BY_CLASS = re.compile(r"getElementsByClassName\(\s*['\"]([\w-]+)['\"]")
_QUERY = re.compile(r"querySelector(?:All)?\(\s*['\"]([^'\"]+)['\"]")
_SIMPLE_SEL = re.compile(r"^[#.][\w-]+$")     # ONLY a lone #id or .class

# ---- definitions (gathered liberally: over-counting only avoids false reds) --
_HTML_ID = re.compile(r"""\bid\s*=\s*['"]([\w-]+)['"]""")
_HTML_CLASS = re.compile(r"""\bclass(?:Name)?\s*=\s*['"]([^'"]+)['"]""")
_CSS_ID = re.compile(r"#([\w-]+)")
_CSS_CLASS = re.compile(r"\.([\w-]+)")
_JS_CLASSLIST = re.compile(r"classList\.(?:add|toggle|replace|remove|contains)\(\s*['\"]([\w-]+)['\"]")
_JS_ID_ASSIGN = re.compile(r"\.id\s*=\s*['\"]([\w-]+)['\"]")
_JS_SETATTR_ID = re.compile(r"setAttribute\(\s*['\"]id['\"]\s*,\s*['\"]([\w-]+)['\"]")

_STYLE_BLOCK = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)
# An inline <script> WITHOUT a src attribute carries JS we should read.
_SCRIPT_BLOCK = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _classes(text: str) -> set[str]:
    out: set[str] = set()
    for value in _HTML_CLASS.findall(text):
        out.update(value.split())
    return out


def check_refs(root: Path) -> list[str]:
    html_files = sorted(root.rglob("*.html"))
    if not html_files:
        return []  # not a web project (from refcheck's view) — nothing to check

    html_text = "\n".join(_read(f) for f in html_files)
    css_text = "\n".join(_read(f) for f in sorted(root.rglob("*.css")))
    js_text = "\n".join(_read(f) for f in sorted(root.rglob("*.js")))
    for f in html_files:                       # inline <style>/<script> blocks
        t = _read(f)
        css_text += "\n" + "\n".join(_STYLE_BLOCK.findall(t))
        js_text += "\n" + "\n".join(_SCRIPT_BLOCK.findall(t))

    # The universe of DEFINED ids/classes — counted from every plausible source,
    # including ids/classes embedded in JS-built HTML strings, so a real
    # definition is never mistaken for a dangling lookup.
    ids = set(_HTML_ID.findall(html_text))
    ids |= set(_CSS_ID.findall(css_text))
    ids |= set(_JS_ID_ASSIGN.findall(js_text)) | set(_JS_SETATTR_ID.findall(js_text))
    ids |= set(_HTML_ID.findall(js_text))      # ids inside JS template/HTML strings

    classes = _classes(html_text) | _classes(js_text)
    classes |= set(_CSS_CLASS.findall(css_text))
    classes |= set(_JS_CLASSLIST.findall(js_text))

    problems: list[str] = []

    def flag(msg: str) -> None:
        if msg not in problems:
            problems.append(msg)

    for x in _GET_BY_ID.findall(js_text):
        if x not in ids:
            flag(f"JS getElementById('{x}') — no element defines id=\"{x}\"")
    for x in _GET_BY_CLASS.findall(js_text):
        if x not in classes:
            flag(f"JS getElementsByClassName('{x}') — no '{x}' class defined anywhere")
    for sel in _QUERY.findall(js_text):
        sel = sel.strip()
        if not _SIMPLE_SEL.match(sel):
            continue                            # compound/dynamic — skip (no false reds)
        name = sel[1:]
        if sel[0] == "#" and name not in ids:
            flag(f"JS querySelector('{sel}') — no element defines id=\"{name}\"")
        elif sel[0] == "." and name not in classes:
            flag(f"JS querySelector('{sel}') — no '{name}' class defined anywhere")

    return problems


def main(argv: list[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else Path.cwd()
    problems = check_refs(root)
    if problems:
        print(f"refcheck: {len(problems)} dangling reference(s) under {root}:")
        for p in problems:
            print(f"   - {p}")
        return 1
    print(f"refcheck: OK — every JS lookup resolves under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
