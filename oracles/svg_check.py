#!/usr/bin/env python3
"""
svg_check.py — a REUSABLE oracle for hand/agent-authored SVG files.

Point Achilles at it as the verify_command for an SVG task:

    --verify "python /path/to/achilles/oracles/svg_check.py ."

It knows nothing about WHAT the SVG depicts (no oracle can) — it checks that the
file is a STRUCTURALLY VALID, renderable SVG, which is exactly the floor a weak
model needs so it can't commit broken markup overnight:

  1. the file parses as XML
  2. the root element is <svg> and declares a viewBox (so it scales)
  3. it contains at least one drawing element (path/rect/circle/…)
  4. it has no external references (href/src to another file) and no <script>

stdlib only. Exit 0 = green (all *.svg under the path are valid); exit 1 = red.

Usage:
    python svg_check.py [path]     # path = a .svg file or a dir (default: cwd)
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_DRAW = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()  # strip the {namespace}


def check_svg(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8", errors="replace"))
    except ET.ParseError as e:
        return [f"{path.name}: not valid XML ({e})"]

    if _local(root.tag) != "svg":
        problems.append(f"{path.name}: root element is <{_local(root.tag)}>, not <svg>")
    if "viewBox" not in root.attrib and not (root.get("width") and root.get("height")):
        problems.append(f"{path.name}: <svg> has no viewBox (nor width+height)")

    tags = [_local(el.tag) for el in root.iter()]
    if not (_DRAW & set(tags)):
        problems.append(f"{path.name}: no drawing elements (path/rect/circle/…) — empty SVG")
    if "script" in tags:
        problems.append(f"{path.name}: contains a <script> element")

    for el in root.iter():
        for attr in ("href", "{http://www.w3.org/1999/xlink}href", "src"):
            v = el.get(attr)
            if v and not v.startswith("#"):        # in-document refs (#id) are fine
                problems.append(f"{path.name}: external reference {attr}='{v}'")
    return problems


def main(argv: list[str]) -> int:
    target = Path(argv[1]).resolve() if len(argv) > 1 else Path.cwd()
    files = [target] if target.is_file() else sorted(target.rglob("*.svg"))
    if not files:
        print(f"svg-check: no .svg file found under {target}")
        return 1

    problems: list[str] = []
    for f in files:
        problems.extend(check_svg(f))

    if problems:
        print(f"svg-check: {len(problems)} problem(s) in {len(files)} file(s):")
        for p in problems:
            print(f"   - {p}")
        return 1
    print(f"svg-check: OK — {len(files)} valid SVG file(s) under {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
