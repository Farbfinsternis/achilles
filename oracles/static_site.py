#!/usr/bin/env python3
"""
static_site.py — a REUSABLE oracle for ANY static website.

Point Achilles at it as the verify_command (works for every static site, with
zero per-project authoring):

    verify_command = "python /path/to/achilles/oracles/static_site.py"

It knows NOTHING about your project's domain. It checks *internal consistency* —
the things that must hold for any correct static site regardless of content:

  1. there is at least one .html file
  2. every locally-referenced file exists       (href/src pointing at a local path)
  3. every in-page anchor (#foo) has a matching id="foo"

That catches the bugs we keep hitting — a linked `styles.css` that was never
created, a nav link to `#kontakt` with no such section — WITHOUT you writing a
single project-specific assertion.

What it deliberately CANNOT check: domain truth — e.g. that the page actually
mentions "Peter Meyer". That is the one thing only a per-project test can
assert. This generic oracle handles everything else, for free, on every project.

stdlib only, in keeping with Achilles' zero-dependency rule.
Exit 0 = green (all references resolve); exit 1 = red (problems listed).

Usage:
    python static_site.py [directory]     # default: current directory
"""

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, unquote


# Attributes that can point at a local resource we should be able to find.
REF_ATTRS = ("href", "src")
# URL schemes that are NOT local files — we don't try to resolve these.
EXTERNAL_SCHEMES = ("http", "https", "mailto", "tel", "data", "javascript", "ftp")


class _SiteParser(HTMLParser):
    """Collects, for one HTML file: the ids it defines, the in-page anchors it
    links to, and the local files it references."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.anchors: list[str] = []      # targets of href="#foo"
        self.local_refs: list[str] = []   # local file paths referenced

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if d.get("id"):
            self.ids.add(d["id"])
        for attr in REF_ATTRS:
            value = d.get(attr)
            if value:
                self._classify(value.strip())

    def _classify(self, value: str) -> None:
        # Pure in-page anchor: "#section"
        if value.startswith("#"):
            if len(value) > 1:
                self.anchors.append(value[1:])
            return
        parsed = urlparse(value)
        if parsed.scheme in EXTERNAL_SCHEMES or parsed.netloc:
            return  # external link or protocol-relative //cdn… — not our file
        if parsed.path:
            self.local_refs.append(unquote(parsed.path))  # drop ?query and #frag


def _resolve(root: Path, html_file: Path, ref: str) -> Path:
    """A local ref starting with '/' is relative to the site root; otherwise
    it is relative to the HTML file that contains it."""
    if ref.startswith("/"):
        return (root / ref.lstrip("/")).resolve()
    return (html_file.parent / ref).resolve()


def check_site(root: Path) -> list[str]:
    problems: list[str] = []
    html_files = sorted(root.rglob("*.html"))
    if not html_files:
        return [f"no .html file found under {root}"]

    for hf in html_files:
        rel = hf.relative_to(root)
        parser = _SiteParser()
        try:
            parser.feed(hf.read_text(encoding="utf-8", errors="replace"))
            parser.close()
        except Exception as e:                       # html.parser is lenient; this is a safety net
            problems.append(f"{rel}: could not parse HTML ({e})")
            continue

        for ref in parser.local_refs:
            if not _resolve(root, hf, ref).exists():
                problems.append(f"{rel}: references missing file '{ref}'")

        for anchor in parser.anchors:
            if anchor not in parser.ids:
                problems.append(f"{rel}: links to #{anchor} but no element has id=\"{anchor}\"")

    return problems


def main(argv: list[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else Path.cwd()
    problems = check_site(root)
    if problems:
        print(f"static-site oracle: {len(problems)} problem(s) under {root}:")
        for p in problems:
            print(f"   - {p}")
        return 1
    print(f"static-site oracle: OK — every reference resolves under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
