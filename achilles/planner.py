"""
planner.py — the "dumb decision": turn a narrated goal into a task list.

This is the part you were most curious about. The harness does NOT understand
your prompt. It does three mechanical things:

  1. wrap your prompt in a fixed planner template that DEMANDS a rigid format,
  2. send it to the model,
  3. regex the model's reply back into a list of steps.

The intelligence is 100% the model's. The harness only owns the format contract
(`- [ ]` lines) so a regex can read smart output. That contract is the entire
reason dumb code can drive a smart model.
"""

import re
from typing import List

from .llm import chat


PLAN_SYSTEM = """You are a planning assistant for a coding agent that works in SMALL, verifiable steps.

Turn the user's request into a checklist. Rules for a GOOD plan:
- Each step is small enough to implement and TEST on its own.
- Order steps so that earlier ones can be verified before later ones depend on them.
- The LAST step is always to run the project's tests / build to confirm the whole thing works.
- Prefer 3-8 steps. Do not pad.

OUTPUT FORMAT — this is strict. Output ONLY the checklist, nothing else:
- [ ] first step
- [ ] second step
- [ ] run the tests to confirm everything passes

No preamble, no numbering, no explanation. Every line starts with "- [ ] "."""


# Appended ONLY when ComfyUI image generation is enabled (config.comfy_url). The
# planner never sees the act-loop's tool list, so without this it plans HTML but
# never an image — the "knowing != doing" gap. This makes the intent a real step,
# which is what actually drives a weak model to call generate_image.
IMAGE_PLAN_NUDGE = """

IMAGE GENERATION IS AVAILABLE via a `generate_image` tool. If the request asks for
a picture, photo, image, illustration or other visual that does not already exist
in the project, add an explicit step to CREATE it (e.g. "generate the landscape
image into assets/landscape.jpg"), saved under assets/, and place that step BEFORE
the step that puts it on the page. Do not rely on placeholder URLs."""


PLAN_USER_TEMPLATE = """Project files (top level):
{tree}

User request:
{goal}

Write the checklist now."""


# Tolerant: accept "- [ ]", "* ", "1. ", "- " — small models drift from the
# exact format, and a step list is too valuable to lose to a missing checkbox.
_STEP_RES = [
    re.compile(r"^\s*[-*]\s*\[[ xX]?\]\s*(.+?)\s*$"),   # - [ ] step   (preferred)
    re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$"),             # 1. step
    re.compile(r"^\s*[-*]\s+(.+?)\s*$"),                # - step
]


def parse_checklist(text: str) -> List[str]:
    steps: List[str] = []
    for line in (text or "").splitlines():
        for rx in _STEP_RES:
            m = rx.match(line)
            if m:
                step = m.group(1).strip()
                # Skip obvious non-steps the model sometimes emits as bullets.
                if step and not step.lower().startswith(("here", "plan:", "note")):
                    steps.append(step)
                break
    return steps


def make_plan(config, goal: str, tree: str) -> List[str]:
    system = PLAN_SYSTEM
    if getattr(config, "comfy_url", ""):
        system += IMAGE_PLAN_NUDGE
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": PLAN_USER_TEMPLATE.format(tree=tree, goal=goal)},
    ]
    # No hard token cap here: reasoning models spend tokens on thinking before
    # the checklist, so a fixed cap (1024) truncated them mid-plan. Fall back to
    # config.max_tokens (0 = use the model's own context window).
    reply = chat(config, messages, temperature=config.temperature)
    return parse_checklist(reply)


# Used when the user picks "edit" at plan approval: they describe a change in plain
# words and the MODEL revises the plan — keeping the steps the change doesn't touch,
# rather than the user hand-rewriting the checklist. The same rigid output contract
# as make_plan, so the reply parses identically.
REVISE_SYSTEM = """You are revising an existing plan for a coding agent that works in SMALL, verifiable steps.

Apply the user's requested change to the plan. CRITICAL:
- KEEP every existing step the change does not touch, in its original order and wording.
- Change ONLY what the request asks: edit, add, remove or reorder the affected steps.
- Do not re-plan from scratch and do not "improve" untouched steps.
- The LAST step stays the one that runs the tests / build to confirm everything works.

OUTPUT FORMAT — strict. Output ONLY the full revised checklist, nothing else:
- [ ] first step
- [ ] second step
No preamble, no numbering, no explanation. Every line starts with "- [ ] "."""


REVISE_USER_TEMPLATE = """Project files (top level):
{tree}

Goal:
{goal}

Current plan:
{plan}

Requested change:
{instruction}

Write the COMPLETE revised checklist now (keep the untouched steps verbatim)."""


def revise_plan(config, goal: str, steps: List[str], instruction: str,
                tree: str) -> List[str]:
    """Re-plan in place: hand the model the current steps plus the user's change and
    return the revised checklist. Preserves untouched steps by contract (see
    REVISE_SYSTEM), so "edit" adjusts a plan instead of discarding it."""
    system = REVISE_SYSTEM
    if getattr(config, "comfy_url", ""):
        system += IMAGE_PLAN_NUDGE
    current = "\n".join(f"- [ ] {s}" for s in steps)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": REVISE_USER_TEMPLATE.format(
            tree=tree, goal=goal, plan=current, instruction=instruction)},
    ]
    reply = chat(config, messages, temperature=config.temperature)
    return parse_checklist(reply)
