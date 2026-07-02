"""
protocol.py — the format contract between a (dumb) harness and a (smart) model.

A small local model cannot be trusted to emit clean native tool-calls, so
Achilles uses a *text* protocol instead: the model writes a fenced block tagged
`act`, and this module parses it back out with a regex. That is the whole trick
we keep coming back to — the harness understands nothing; it only enforces a
rigid shape so dumb code can pass smart output along.

A tool block looks like this:

    ```act
    tool: write_file
    path: src/foo.py
    ---
    def foo():
        return 42
    ```

Header lines are `key: value`. An optional `---` line separates the headers from
a freeform body (used by write_file as the file content). Tools without a body
(read_file, list_dir, run_command) just use headers.

Parsing is deliberately tolerant: it accepts ```act / ```tool / ```action, and
either `tool:` or `name:` for the tool name — because format compliance is the
single most fragile thing a 7B does, and being forgiving here saves whole retries.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    body: Optional[str] = None


# Match a fenced block tagged act/tool/action. We only ever honour one action
# per model turn — one act, one result, loop. That keeps the loop simple and
# stops a model from queuing five edits before seeing a single result.
#
# The fence marker is captured and back-referenced for the close: a block opened
# with ```  closes with ```, one opened with ~~~ closes with ~~~. The body match
# is GREEDY on purpose so it runs to the LAST matching fence, not the first —
# otherwise a write_file body that itself contains a ``` code fence (writing a
# README, say) would be truncated at the first inner fence, silently losing the
# rest of the file. A model can also sidestep the ambiguity entirely by wrapping
# the block in ~~~act … ~~~, leaving ``` free to appear verbatim in the body.
_FENCE_RE = re.compile(
    r"(?P<fence>```|~~~)(?:act|tool|action)[^\n]*\n(?P<body>.*)\n(?P=fence)",
    re.DOTALL | re.IGNORECASE,
)
_HEADER_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*(.*)$")


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """Return the first ToolCall in `text`, or None if the model didn't act.

    None is meaningful: it means the model produced prose instead of an action,
    which the loop reads as "I'm done acting on this step" (then the oracle, not
    the model, decides whether that's actually true)."""
    m = _FENCE_RE.search(text or "")
    if not m:
        return None

    headers: dict = {}
    body_lines: list[str] = []
    in_body = False

    for line in m.group("body").splitlines():
        if in_body:
            body_lines.append(line)
            continue
        if line.strip() == "---":
            in_body = True
            continue
        hm = _HEADER_RE.match(line)
        if hm:
            headers[hm.group(1).lower()] = hm.group(2).strip()
            continue
        if line.strip() == "":
            continue
        # An unexpected non-header line before any separator: assume the model
        # forgot the `---` and started the body early. Tolerate it.
        in_body = True
        body_lines.append(line)

    name = headers.pop("tool", None) or headers.pop("name", None)
    if not name:
        return None

    body = "\n".join(body_lines) if body_lines else None
    return ToolCall(name=name.strip(), args=headers, body=body)
