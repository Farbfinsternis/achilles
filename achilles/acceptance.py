"""
acceptance.py — the Definition of Done: the WHAT, as a checkable contract.

Achilles' verify_command is the FLOOR: it proves nothing is BROKEN (tests pass,
references resolve). It says nothing about whether the GOAL was achieved. For
generative work ("build a site for Peter Meyer") the floor is trivially
satisfiable, so a weak model can reach green by doing the bare minimum and the
harness, seeing green, would stop. The plan's real intent gets discarded.

The Definition of Done is the CEILING. A second planning pass turns the goal
into acceptance criteria, each tagged by HOW it is checked:

    - [ ] exists:   <path>               (the HARNESS checks os.path)
    - [ ] contains: <path> :: <text>     (the HARNESS checks substring)
    - [ ] judge:    <natural-language>   (model-as-judge, grounded)

The PLANNER is offered only exists/contains/judge. It is NEVER asked to author a
shell command, because a weak model cannot do it reliably — three runs produced
three different failures (`; exit $?`, a forgotten `import sys`, prose glued onto
the command). Letting it author a check reintroduces the very circularity the
judge avoids: the harness would treat the model's OWN command bug as an unmet
requirement and thrash. Execution belongs to the verify_command FLOOR.

All mechanical kinds dispatch THROUGH THE TOOL REGISTRY (exists→file_exists,
contains→file_contains, run→run_command), so acceptance shares one tool system
with the model's hands. A hand-written done.md may therefore also use:
    - [ ] check: <tool> k=v [:: body]   any registry tool (incl. a user oracle plugin); pass = exit 0
    - [ ] run: <command>                a raw command (sanitized of shell tails)
make_acceptance never EMITS run/check (a weak model can't author them); they are
honoured only when a human writes them.

The judge is the SAME model — no second model loaded. It runs in a fresh,
role-isolated context as a strict auditor that has never seen the build. Because
every Achilles call is stateless, the model literally cannot know it authored
the files; we add adversarial framing + a demand for cited evidence so it grades
the artifact, not its pride. (Honest limit: isolation removes authorship BIAS;
it cannot grant a discernment the model lacks — that ceiling is what we probe.)
"""

import re
from dataclasses import dataclass
from pathlib import Path

from .llm import chat, LLMError
from .protocol import ToolCall
from . import style as ui


@dataclass
class Criterion:
    kind: str            # "cmd" | "judge"
    text: str
    done: bool = False


@dataclass
class Failure:
    criterion: Criterion
    reason: str


class JudgeUnavailable(RuntimeError):
    """The judge MODEL could not be reached. This is an INFRASTRUCTURE failure,
    not an unmet criterion: the accept loop must HALT, not treat every judged
    criterion as failed and command the model to "fix" possibly-correct files
    (Bug 11). Raised out of check(); the harness catches it and stops."""


# ---- pass 2 of planning: derive the acceptance contract -------------------

ACCEPT_SYSTEM = """You define the ACCEPTANCE CRITERIA for a coding goal — the checklist a STRICT
reviewer uses to decide the goal is genuinely met, not just "nothing crashed".

Each criterion is ONE line. Choose the most specific check type:
  - [ ] exists: <path>                that file must exist
  - [ ] contains: <path> :: <text>    <text> must appear VERBATIM in that file
  - [ ] judge: <criterion>            a content/quality judgment made by eye

CRITICAL — contains: <text> is a LITERAL substring, matched byte-for-byte. It is
NOT a description of a property. Put a short, concrete token you are SURE will
appear verbatim: a tag, a class name, a keyword, an attribute, a heading string.
  GOOD:  - [ ] contains: index.html :: <canvas
  GOOD:  - [ ] contains: styles.css :: display: grid
  GOOD:  - [ ] contains: index.html :: Agentic Development
  BAD:   - [ ] contains: index.html :: modern CSS styling (gradients or grid)
  BAD:   - [ ] contains: index.html :: self-contained document, no dependencies
The BAD lines describe a QUALITY — that prose never appears literally in the file,
so the check can NEVER pass. Any quality, capability, or "the page looks/feels/is
X" belongs to judge:, never to contains:. If you cannot name the exact string,
use judge:. Also mind WHICH file holds the text: styling checks target the CSS
file, not the HTML, if the CSS is external.

Rules:
- Prefer exists/contains over judge for ANYTHING mechanical. The HARNESS performs
  these itself — you only NAME a path or a substring. NEVER write shell commands,
  pipes, redirects, or path logic.
- Do NOT add a criterion that runs the tests or the build. Whether the project
  runs is verified separately. Acceptance criteria describe the END STATE: which
  files exist, what text they contain, qualities a reviewer would see.
- Use judge: only for what no check can measure (coverage of the request,
  clarity, appropriateness, "looks professional").
- Describe the END STATE that must be true; do NOT restate the build steps.
- 3-7 criteria, each necessary and checkable.

OUTPUT ONLY the checklist, one criterion per line, each starting with
"- [ ] exists: ", "- [ ] contains: ", or "- [ ] judge: "."""

# Appended ONLY when ComfyUI image generation is enabled. It turns "the model
# could make an image" into "the harness checks it did": an exists: for the file
# and a contains: for the reference, so a skipped image goes RED and is fixed.
IMAGE_ACCEPT_NUDGE = """

If the goal requires a picture/photo/image and the project generates one (images
live under assets/), add an `exists:` criterion for that image file and a
`contains:` criterion that the page actually references it."""

ACCEPT_USER_TEMPLATE = """Project files (top level):
{tree}

User request:
{goal}

Write the Definition of Done now."""


_TAGGED = re.compile(
    r"^\s*[-*]\s*(?:\[[ xX]?\]\s*)?(exists|absent|contains|run|cmd|check|judge)\b\s*:?\s*(.+?)\s*$",
    re.I)
_BULLET = re.compile(r"^\s*[-*]\s*(?:\[[ xX]?\]\s*)?(.+?)\s*$")


def _normalise(kind: str, text: str) -> Criterion:
    """Repair the exists+contains fusion a weak planner keeps emitting
    ("exists: index.html :: <html>") at PARSE time, so the persisted done.md is
    written clean. A `::` tail only belongs to contains — reinterpret an exists
    with one as the contains it plainly means (file must exist AND hold the text),
    and strip a stray tail off absent (which has no text half). _criterion_to_call
    keeps the same guard as a safety net for directly-constructed criteria."""
    if kind == "exists" and "::" in text:
        return Criterion(kind="contains", text=text)
    if kind == "absent" and "::" in text:
        return Criterion(kind="absent", text=text.split("::", 1)[0].strip())
    return Criterion(kind=kind, text=text)


def parse_acceptance(text: str) -> list[Criterion]:
    """Tolerant parse. A tagged line wins; an untagged bullet defaults to judge
    (small models drop the tag, and a criterion is too valuable to discard).
    `cmd:` is accepted as a backward-compatible alias for `run:`."""
    out: list[Criterion] = []
    for line in (text or "").splitlines():
        m = _TAGGED.match(line)
        if m:
            kind = m.group(1).lower()
            kind = "run" if kind == "cmd" else kind
            out.append(_normalise(kind, m.group(2).strip()))
            continue
        b = _BULLET.match(line)
        if b:
            t = b.group(1).strip()
            if t and not t.lower().startswith(("here", "note", "definition", "criteria")):
                out.append(Criterion(kind="judge", text=t))
    return out


def render_acceptance(goal: str, criteria: list[Criterion]) -> str:
    lines = [
        "# Achilles — Definition of Done", "",
        f"> Goal: {goal}", "",
        "# exists/contains: checked by the harness. judge: assessed by the model.",
        "# You may also hand-add registry-tool checks (run only the planner won't):",
        "#   - [ ] check: html_valid path=index.html      (any registry tool; pass = exit 0)",
        "#   - [ ] run: python -m pytest -q               (a raw command)",
        "# Edit freely, then re-run to continue.", "",
    ]
    lines += [f"- [ ] {c.kind}: {c.text}" for c in criteria]
    return "\n".join(lines) + "\n"


def make_acceptance(config, goal: str, tree: str) -> list[Criterion]:
    system = ACCEPT_SYSTEM
    if getattr(config, "comfy_url", ""):
        system += IMAGE_ACCEPT_NUDGE
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": ACCEPT_USER_TEMPLATE.format(tree=tree, goal=goal)},
    ]
    # No hard token cap: reasoning models spend tokens thinking before the list,
    # so a fixed cap truncated them. Falls back to config.max_tokens (0 = uncapped).
    reply = chat(config, messages, temperature=config.temperature)
    # The model must never author an executable command or a raw tool-call — it
    # cannot do it reliably. Keep only the kinds the HARNESS checks robustly;
    # run:/check: survive only in a HUMAN-written done.md.
    return [c for c in parse_acceptance(reply)
            if c.kind in ("exists", "contains", "absent", "judge")]


# ---- checking the contract ------------------------------------------------

def check(config, criteria: list[Criterion], registry, ctx, log) -> list[Failure]:
    """Return the FAILED criteria. Mechanical checks (exists/contains/run/check)
    are dispatched THROUGH THE REGISTRY — the same tool system as the model's
    hands — so an oracle plugin is usable as an acceptance check. judge: criteria
    are batched to one fresh, adversarial judge call."""
    failures: list[Failure] = []
    judge_items: list[Criterion] = []

    for c in criteria:
        if c.kind == "judge":
            judge_items.append(c)
            continue
        ok, reason = _check_mechanical(c, registry, ctx)
        mark = ui.ok("✔") if ok else ui.bad("✖")
        log(f"   {mark} " + ui.accent(f"[{c.kind}]") + f" {c.text}")
        if not ok:
            failures.append(Failure(c, reason))

    if judge_items:
        for c, (ok, reason) in zip(judge_items, _judge(config, judge_items, ctx, log)):
            mark = ui.ok("✔") if ok else ui.bad("✖")
            log(f"   {mark} " + ui.accent("[judge]") + f" {c.text}")
            if not ok:
                failures.append(Failure(c, reason))

    return failures


def _criterion_to_call(c: Criterion) -> ToolCall | None:
    """Map an acceptance criterion onto a registry ToolCall. This is the bridge
    that unifies acceptance with the tool system: exists/contains/run/check all
    become tool dispatches, so a user's oracle plugin works as a check too."""
    if c.kind == "exists":
        # A weak planner sometimes fuses exists+contains onto one line
        # ("exists: index.html :: <html>"). Taken literally the whole string
        # becomes the path — a file that can never exist — and the accept loop
        # thrashes forever. Honour the evident intent: a `::` tail means the
        # file must exist AND contain that text, which is exactly `contains`.
        if "::" in c.text:
            rel, needle = (s.strip() for s in c.text.split("::", 1))
            return ToolCall("file_contains", {"path": rel, "text": needle})
        return ToolCall("file_exists", {"path": c.text.strip()})
    if c.kind == "absent":
        # `absent` has no text half; drop any stray `:: tail` from the path.
        path = c.text.split("::", 1)[0].strip()
        return ToolCall("file_absent", {"path": path})
    if c.kind == "contains":
        if "::" not in c.text:
            return None
        rel, needle = (s.strip() for s in c.text.split("::", 1))
        return ToolCall("file_contains", {"path": rel, "text": needle})
    if c.kind == "run":
        return ToolCall("run_command", {"command": _sanitize_run(c.text)})
    if c.kind == "check":
        return _parse_check(c.text)
    return None


def _parse_check(text: str) -> ToolCall | None:
    """Parse `check: <tool> k=v k=v [:: body]` into a ToolCall. Values with
    spaces go in the `::` body (one-line key=val can't hold them)."""
    body = None
    if "::" in text:
        text, body = (s.strip() for s in text.split("::", 1))
    parts = text.split()
    if not parts:
        return None
    args = {k: v for tok in parts[1:] if "=" in tok for k, v in [tok.split("=", 1)]}
    return ToolCall(parts[0], args, body)


def _check_mechanical(c: Criterion, registry, ctx) -> tuple[bool, str]:
    call = _criterion_to_call(c)
    if call is None:
        return False, f"malformed {c.kind} criterion: {c.text}"
    result = registry.dispatch(call, ctx)
    if result.startswith("exit=0"):
        return True, ""
    # Surface the tool's own reason (the line(s) after the exit code, or the lot).
    rest = "\n".join(result.splitlines()[1:]).strip()
    return False, rest or result.splitlines()[0] if result else "no output"


# Strip the brittle shell tails a weak model keeps appending to a `run:` command
# ("; exit $?", "&& echo done", "| tail") — the harness reads the exit code itself.
_RUN_TAIL = re.compile(r"\s*(?:;|&&|\|\||\|)\s.*$")


def _sanitize_run(cmd: str) -> str:
    return _RUN_TAIL.sub("", cmd.strip()).strip()


JUDGE_SYSTEM = """You are a STRICT acceptance reviewer. A contractor has submitted the project
files below. You did NOT write them and owe them no benefit of the doubt — your
job is to catch work that does not meet the bar.

For each numbered criterion, decide whether the SUBMITTED FILES satisfy it:
- Answer PASS only if you can cite concrete evidence in the files — name the file
  and quote the exact snippet. If you cannot cite evidence, answer FAIL.
- Judge only what the files actually CONTAIN, never what they probably intend.

Output EXACTLY one line per criterion, in order, nothing else:
<n>: PASS — <file: quoted evidence>
<n>: FAIL — <what is missing>"""


# "1: PASS — ...", "2) FAIL: ...", "3. pass - ..." etc.
_VERDICT = re.compile(r"^\s*\(?(\d+)\)?\s*[:.\)\-]\s*(PASS|FAIL)\b[\s:．。\-–—]*(.*)$", re.I)


def _parse_verdicts(reply: str, n: int) -> list[tuple[bool, str]]:
    found: dict[int, tuple[bool, str]] = {}
    for line in (reply or "").splitlines():
        m = _VERDICT.match(line.strip())
        if m:
            idx = int(m.group(1))
            ok = m.group(2).upper() == "PASS"
            reason = m.group(3).strip() or ("ok" if ok else "no reason given")
            found[idx] = (ok, reason)
    # A missing verdict is a FAIL by default — strictness, not optimism.
    return [found.get(i, (False, "the judge returned no verdict for this criterion"))
            for i in range(1, n + 1)]


def _judge(config, items: list[Criterion], ctx, log) -> list[tuple[bool, str]]:
    bundle = _gather_context(ctx,
                            per_file=getattr(config, "judge_char_per_file", 6000),
                            total=getattr(config, "judge_char_budget", 24000))
    numbered = "\n".join(f"{i}. {c.text}" for i, c in enumerate(items, 1))
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content":
            f"SUBMITTED FILES:\n\n{bundle}\n\nCRITERIA:\n{numbered}\n\nReview now."},
    ]
    try:
        # temperature 0 — judging should be as deterministic as the model allows.
        # No hard token cap: reasoning judge models need room to think before the
        # verdict. Falls back to config.max_tokens (0 = uncapped).
        reply = chat(config, messages, temperature=0.0,
                     model=config.judge_model or None)
    except LLMError as e:
        # NOT a content FAIL — the judge server is down/unreachable. Signal the
        # harness to halt instead of marking every criterion unmet (Bug 11).
        log(ui.bad(f"   ✖ judge unavailable: {e}"))
        raise JudgeUnavailable(str(e)) from e
    return _parse_verdicts(reply, len(items))


_SKIP_DIRS = {".git", ".achilles", "__pycache__", "node_modules", ".venv", ".pytest_cache"}
_TEXT_SUFFIXES = {".html", ".htm", ".css", ".js", ".ts", ".jsx", ".tsx", ".py",
                  ".json", ".md", ".txt", ".toml", ".yml", ".yaml", ".svg", ".cfg"}


def _gather_context(ctx, per_file: int = 6000, total: int = 24000) -> str:
    """Bundle the workspace's text files for the judge. v1 reads everything that
    fits the budget; large projects will want the repo-map retrieval that is on
    the wishlist — the judge would then see only the relevant slice.

    Files that don't fit are counted once at the end, not appended as per-file
    marker lines (those used to grow unbounded, ironically eating the very budget
    they reported). A trimmed/omitted file means the judge may lack evidence, so
    the budget should be generous enough that this stays rare (Bug 7)."""
    root: Path = ctx.ws
    chunks: list[str] = []
    used = 0
    omitted = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        snippet = text[:per_file]
        if len(text) > per_file:
            snippet += f"\n... [{len(text) - per_file} chars trimmed] ..."
        block = f"=== {rel.as_posix()} ===\n{snippet}\n"
        if used + len(block) > total:
            omitted += 1
            continue
        chunks.append(block)
        used += len(block)
    body = "\n".join(chunks) or "(no readable text files in the workspace)"
    if omitted:
        body += (f"\n[note: {omitted} more text file(s) omitted — context budget "
                 "reached. A criterion whose evidence would live in an omitted or "
                 "trimmed file cannot be judged from what is shown.]\n")
    return body
