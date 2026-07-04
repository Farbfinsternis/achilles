"""
harness.py — the dumb orchestrator that ties everything together.

The whole philosophy of Achilles lives in one sentence:

    The model is smart but forgetful; the harness is dumb but reliable —
    so we move every burden the model is bad at OUT of the model.

  * The model would forget the task        -> the plan lives in .achilles/plan.md, re-read each step.
  * The model can't be trusted to verify   -> the HARNESS runs the tests after every step.
  * Big tasks fill the context window      -> each step runs in a FRESH message list (act -> verify -> commit -> reset).
  * A wrong edit could pile up             -> every green step is git-committed (cheap rollback point).

None of this requires a clever model. It requires a clever *loop*.
"""

import json
import subprocess
from pathlib import Path
from typing import Callable, List

from .config import Config
from .llm import chat, complete_act, ActReply, LLMError
from .planner import make_plan, revise_plan
from .protocol import parse_tool_call, ToolCall
from .tools import build_registry, ToolContext
from . import style as ui
from . import workflows as wf
from . import comfy_client as cc
from . import lmstudio
from .comfy import _store_dir


# The tool list is generated from the registry (so a newly added tool announces
# itself with no prompt edit). {tools} is filled in at run time.

# Shared across both protocols: a web page MAY pull frameworks and fonts from a
# CDN. Framed as a permission (never a ban — per the positive-framing rule), with
# concrete URLs so a weak model doesn't invent a broken one. Nothing is downloaded
# locally; the browser fetches it, so this needs no network tool.
_WEB_ASSETS_RULE = (
    "- For a web page you MAY load CSS/JS frameworks and web fonts from a CDN — link "
    "them in the HTML <head>; you don't need to download or vendor them. Good picks: "
    "Tailwind CSS (<script src=\"https://cdn.tailwindcss.com\"></script>) and Google "
    "Fonts (a <link> to https://fonts.googleapis.com, then use the font-family in your CSS)."
)

EXECUTE_SYSTEM_TEMPLATE = ("""You are Achilles, a coding agent working in ONE small step of a larger plan.

To use a tool, output a single fenced block tagged `act`, with `key: value`
headers and (for write_file) a `---` body. Your tools:

{tools}

Rules:
- Take ONE action per message, then wait for its result before the next.
- Read before you write. Don't guess a file's contents — read_file it.
- To CREATE or CHANGE a file you MUST use the write_file tool inside an `act`
  block. Writing code, HTML or CSS in a normal ``` code block does NOT save it —
  the harness only persists what write_file writes. A file "shown" in prose is lost.
- When the current step is fully implemented, STOP and reply with a short plain
  sentence (no `act` block). Do NOT claim success — the harness verifies itself.
- Keep edits minimal and focused on the current step only.
- The `write_file` body REPLACES the whole file, so include the complete file.
- If the file you write itself contains ``` code fences (e.g. a Markdown README),
  wrap the WHOLE act block in ~~~act … ~~~ instead of ``` so the inner fences are
  not mistaken for the end of your block.
""" + _WEB_ASSETS_RULE)


# The native-tool-calling variant: the tools arrive as real functions (the OpenAI
# `tools` field), so the model calls them directly and we drop the fence syntax.
# The behavioural rules are the same — only the "how to act" part differs.
NATIVE_SYSTEM_TEMPLATE = ("""You are Achilles, a coding agent working in ONE small step of a larger plan.

You have these tools, provided to you as callable functions — call them directly:

{tools}

Rules:
- Take ONE action per message, then wait for its result before the next.
- Read before you write. Don't guess a file's contents — read_file it first.
- To CREATE or CHANGE a file you MUST call the write_file tool. Code shown in a
  normal message is NOT saved — only write_file persists a file.
- write_file's `content` REPLACES the whole file, so pass the complete file.
- When the current step is fully implemented, STOP: reply with a short plain
  sentence and NO tool call. Do NOT claim success — the harness verifies itself.
- Keep changes minimal and focused on the current step only.
""" + _WEB_ASSETS_RULE)


class Harness:
    def __init__(self, config: Config, log: Callable[[str], None] = print):
        self.cfg = config
        self.log = log
        self.ws = config.workspace_path
        self.ctx = ToolContext(self.ws)
        self.registry = build_registry(config, log)
        # Native tool-calling is on unless config says otherwise; the schema is
        # built once. _native_tools may flip to False mid-run if the server turns
        # out not to accept the tools field (see _act_until_done).
        self._native_tools = getattr(config, "native_tools", True)
        self._tool_schema = self.registry.tool_schemas()
        self.state_dir = self.ws / ".achilles"
        self.plan_path = self.state_dir / "plan.md"
        self.dod_path = self.state_dir / "done.md"

    # ---- public entry point -------------------------------------------

    def run(self, goal: str) -> bool:
        self.state_dir.mkdir(exist_ok=True)
        self._maybe_git_init()

        # A cold LM Studio (nothing loaded) makes the very first LLM call — the
        # planner — 400 with "No models loaded", killing the run before it starts.
        # Restore the last model the user actually used, so the run just works.
        lmstudio.ensure_loaded(
            self.cfg,
            lambda m: self.log(ui.warn(m) if "⚠" in m else ui.muted(m)))

        # A user can just write "use this workflow …" and drag the file into the
        # terminal (which inserts its path). Adopt it as the run's image workflow
        # and strip the path from the goal BEFORE planning, so the model never sees
        # a raw filesystem path in its requirement.
        goal = self._adopt_dropped_workflow(goal)

        if not self.cfg.verify_command:
            self.log(ui.warn("⚠  No verify_command set — Achilles has no oracle and is "
                     "flying blind. Steps will be committed without proof. Set "
                     "verify_command in achilles.toml (e.g. \"python -m pytest -q\")."))

        # STATE: PLAN — decide whether to RESUME or start FRESH.
        # The plan is keyed to a goal (its `> Goal:` line). We resume only when a
        # persisted plan belongs to THIS goal and still has unfinished steps. A
        # finished plan, or one written for a different goal, gets archived so the
        # new goal is planned from scratch (otherwise a completed plan silently
        # swallows the next goal — every step `done`, nothing to do).
        plan = self._load_plan()
        if plan and (all(s["done"] for s in plan) or self._stored_goal() != goal):
            self._archive_plan()
            plan = []
        if not plan:
            self.log("\n" + ui.head("PLAN") + "\n" + ui.muted("Goal: ") + ui.bold(goal) + "\n")
            tree = self.registry.dispatch(ToolCall("list_dir", {"path": "."}), self.ctx)
            try:
                steps = make_plan(self.cfg, goal, tree)
            except LLMError as e:
                self.log(ui.bad(f"✖  Planning failed: {e}"))
                return False
            if not steps:
                self.log(ui.bad("✖  The model returned no parseable steps. Try rephrasing the goal."))
                return False
            plan = [{"done": False, "text": s} for s in steps]
            self._save_plan(goal, plan)
            self._print_plan(plan)
            # Second planning pass: the Definition of Done (the ceiling). One
            # approval covers both the steps and the acceptance contract.
            if self.cfg.use_acceptance:
                self._make_and_save_dod(goal, tree)
            plan = self._approve_loop(goal, plan, tree)
            if plan is None:
                return False

        # STATE: EXECUTE.
        return self._execute(goal, plan)

    # ---- dropped-workflow adoption (image gen only) -------------------

    def _adopt_dropped_workflow(self, goal: str) -> str:
        """If the goal carries a path to a ComfyUI workflow (a .json that parses as
        an API export), register it ad-hoc as the run's default image workflow and
        return the goal with the path removed. No-op unless image generation is on
        (comfy_url set) and such a path is actually present. Echo only — the resolved
        slots are printed for the human to eyeball; the run then proceeds."""
        if not self.cfg.comfy_url:
            return goal
        found = wf.find_workflow_path(goal)
        if not found:
            return goal
        raw, path = found
        self.log("\n" + ui.muted(f"   … a workflow was included in your request — "
                                 f"analysing {Path(path).name}"))
        store = wf.Store(_store_dir(self.cfg))
        try:
            rep = wf.register_adhoc(store, Path(path), "_adhoc", self.cfg,
                                    self._object_info())
        except wf.WorkflowError as e:
            self.log(ui.warn(f"   ⚠ could not read the dropped workflow: {e}"))
            return wf.strip_workflow_path(goal, raw)
        for note in rep.notes:
            self.log(ui.warn(f"   ⚠ {note}"))
        for line in rep.echo:
            self.log("     " + ui.muted(line))
        if not rep.ok:
            self.log(ui.bad("   ✖ the dropped workflow could not be used:"))
            for err in rep.errors:
                self.log(ui.bad(f"       · {err}"))
            self.log(ui.muted("   (continuing without it — an existing default, if any, is used)"))
            return wf.strip_workflow_path(goal, raw)
        store.set_default("_adhoc")
        self.log(ui.ok("   ✔ using the dropped workflow for images (as _adhoc, now default)"))
        return wf.strip_workflow_path(goal, raw)

    def _object_info(self):
        """ComfyUI's installed node types (to verify the workflow's custom nodes and
        read enum options). None if ComfyUI is unreachable — registration still works
        with name-hint detection, just without the node check."""
        client = cc.ComfyClient(self.cfg.comfy_url)
        if not client.reachable():
            self.log(ui.warn("   ⚠ ComfyUI not reachable — adopting the workflow without "
                             "the custom-node check."))
            return None
        try:
            return client.object_info()
        except cc.ComfyError as e:
            self.log(ui.warn(f"   ⚠ could not read /object_info ({e}) — continuing."))
            return None

    # ---- execution: FLOOR (oracle) then CEILING (Definition of Done) --

    def _execute(self, goal: str, plan: List[dict]) -> bool:
        """Three phases. (1) Work the WHOLE plan — no early-exit on first green;
        the plan carries the intent and discarding it is what crippled generative
        tasks. (2) The FLOOR: a real oracle must be green. (3) The CEILING: the
        Definition of Done, judged and fix-looped until met. Absent a DoD, the
        floor (or a finished plan) is the whole story — the configured fallback."""
        # Load the Definition of Done FIRST and pin its required file paths, so the
        # executor names files to match the contract from step one — not after an
        # acceptance round finds "styles.css" missing because it wrote "style.css".
        criteria = self._load_dod()
        from .acceptance import expected_paths
        self._expected_paths = expected_paths(criteria) if criteria else []

        ok, last_verify = self._work_through_plan(goal, plan)
        if not ok:
            return False

        if self.cfg.verify_command:
            ok, last_verify = self._ensure_floor_green(goal, plan)
            if not ok:
                return False

        if not criteria:
            done_msg = "Oracle green" if self.cfg.verify_command else "All steps done"
            self.log("\n" + ui.ok(f"✔  {done_msg}") + ui.muted(" (no Definition of Done to check)."))
            return True
        return self._acceptance_phase(goal, plan, criteria, last_verify)

    def _work_through_plan(self, goal: str, plan: List[dict]):
        """Run every unfinished step once. The oracle runs after each step as a
        progress/regression SIGNAL (and feeds the next step's prompt), but it is
        NOT a gate — "read the tests" can never go green, and the plan's later
        steps must still run.

        Returns (ok, last_verify). ok is False when the model was unreachable:
        the current step is left UNFINISHED (never marked done) so a resume
        retries it — marking it done would burn the whole plan (every step done,
        nothing to resume) even though no work happened."""
        last_verify = None
        for idx, step in enumerate(plan):
            if step["done"]:
                continue
            self.log("\n" + ui.head(f"STEP {idx + 1}/{len(plan)}") + "\n" + ui.bold(step['text']))
            if not self._work(self._work_prompt(step["text"], plan, last_verify)):
                # Model error: this step did NOT run. Leave it open and stop, so
                # a later resume picks up exactly here (Bug 1: silent plan burn).
                self._save_plan(goal, plan)
                self.log(ui.bad(f"\n✖  Halted at step {idx + 1}: the model was unreachable.") + "\n"
                         + ui.muted("The step is left unfinished — fix the model server and re-run to resume."))
                return False, last_verify
            passed, last_verify = self._verify()
            step["done"] = True
            self._save_plan(goal, plan)
            self._commit(f"achilles: step {idx + 1} — {step['text'][:60]}")
            # A per-step checklist mark: ✔ when the oracle is green (or absent),
            # ✖ when it went red. The step still counts as done either way (the
            # oracle is a signal, not a gate) — this is just the visual receipt.
            mark = ui.ok("✔") if passed else ui.bad("✖")
            self.log(f"   {mark} " + ui.muted(f"step {idx + 1}/{len(plan)} done"))
        return True, last_verify

    def _ensure_floor_green(self, goal: str, plan: List[dict]):
        """The oracle is the floor: nothing may be broken. Focused fix-loop if red."""
        passed, last_verify = self._verify()
        if passed:
            return True, last_verify
        for attempt in range(1, self.cfg.max_retries_per_step + 1):
            self.log("\n" + ui.head(f"FLOOR FIX {attempt}/{self.cfg.max_retries_per_step}", color="yellow")
                     + " " + ui.muted("(oracle still red)"))
            instruction = ("The verification command still fails. Make the failing "
                           "checks pass. Read files as needed.")
            if not self._work(self._work_prompt(instruction, plan, last_verify)):
                self.log(ui.bad("✖  Halted: the model was unreachable during the floor fix."))
                return False, last_verify
            passed, last_verify = self._verify()
            self._commit(f"achilles: floor fix {attempt}")
            if passed:
                return True, last_verify
        self.log(ui.bad("\n✖  Halted: the verification oracle never went green.") + "\n"
                 + ui.muted("Last output:\n" + (last_verify or "")))
        return False, last_verify

    def _acceptance_phase(self, goal: str, plan: List[dict], criteria, last_verify) -> bool:
        """The ceiling: check the Definition of Done, fix the unmet criteria, repeat.
        This is where a small model is pushed past "nothing crashed" toward the
        actual goal — and where we keep the floor green after each fix."""
        from .acceptance import check, JudgeUnavailable
        for rnd in range(1, self.cfg.max_accept_rounds + 1):
            self.log("\n" + ui.head(f"ACCEPT {rnd}/{self.cfg.max_accept_rounds}", color="magenta")
                     + " " + ui.muted("(checking the Definition of Done)"))
            try:
                failures = check(self.cfg, criteria, self.registry, self.ctx, self.log)
            except JudgeUnavailable as e:
                self.log(ui.bad(f"\n✖  Halted: the acceptance judge is unavailable ({e}).") + "\n"
                         + ui.muted("This is an infrastructure failure, not unmet criteria — "
                                    "start the judge model server and re-run to resume."))
                return False
            if not failures:
                self.log("\n" + ui.ok("✔  Definition of Done met — task complete."))
                return True
            self.log(ui.warn(f"   {len(failures)} criterion(s) unmet") + ui.muted(" — sending a targeted fix."))
            unmet = "\n".join(f"- {f.criterion.text}  (problem: {f.reason})" for f in failures)
            instruction = ("The work is NOT done. The following acceptance criteria are not "
                           "yet satisfied. Make the changes needed to meet them. A missing "
                           "FILE (e.g. index.html) must be CREATED with the write_file tool; "
                           "only call generate_image if an actual IMAGE file is missing — do "
                           "not regenerate images to satisfy a missing HTML/CSS file:\n\n" + unmet)
            if not self._work(self._work_prompt(instruction, plan, last_verify)):
                self.log(ui.bad("✖  Halted: the model was unreachable during an acceptance fix."))
                return False
            self._commit(f"achilles: acceptance round {rnd}")
            if self.cfg.verify_command:                      # a fix must not break the floor
                ok, last_verify = self._ensure_floor_green(goal, plan)
                if not ok:
                    return False

        try:
            failures = check(self.cfg, criteria, self.registry, self.ctx, self.log)
        except JudgeUnavailable as e:
            self.log(ui.bad(f"\n✖  Halted: the acceptance judge is unavailable ({e}).") + "\n"
                     + ui.muted("Infrastructure failure, not unmet criteria — re-run when it is back."))
            return False
        if not failures:
            self.log("\n" + ui.ok("✔  Definition of Done met — task complete."))
            return True
        self.log(ui.bad(f"\n✖  Halted: Definition of Done not met after {self.cfg.max_accept_rounds} "
                 "rounds. Still unmet:"))
        for f in failures:
            self.log("   " + ui.bad("-") + f" {f.criterion.text}  " + ui.muted(f"({f.reason})"))
        return False

    def _verify(self):
        """Run the oracle. Returns (passed, output). No oracle -> (True, None)."""
        if not self.cfg.verify_command:
            return True, None
        self.log("   " + ui.muted("verify>") + " " + ui.accent(self.cfg.verify_command))
        result = self.ctx.shell(self.cfg.verify_command)
        passed = result.startswith("exit=0")
        self.log("   " + (ui.ok("✔ green") if passed else ui.bad("✖ red")))
        return passed, result

    def _work(self, user_prompt: str) -> bool:
        """One unit of work in a FRESH context (the 'reset'): system + pinned
        plan + instruction. Nothing from previous steps carries over, so the
        window never fills up no matter how big the overall task is.

        Returns False if the model was unreachable/errored, so the caller can
        leave the step UNFINISHED instead of committing empty work as done."""
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_prompt},
        ]
        return self._act_until_done(messages)

    def _system_prompt(self) -> str:
        """The act-loop system prompt, matched to the current protocol: native
        tool-calling drops the fence syntax and the usage examples (the JSON schema
        carries the arg shape), the text protocol keeps both."""
        tools = self.registry.describe(include_usage=not self._native_tools)
        template = NATIVE_SYSTEM_TEMPLATE if self._native_tools else EXECUTE_SYSTEM_TEMPLATE
        return template.format(tools=tools)

    def _act_until_done(self, messages: List[dict]) -> bool:
        """Inner loop: let the model take actions until it stops acting.

        Native tool-calling first (structured tool_calls), with the text `act`
        protocol as the fallback — for a model that answers in prose, and for a
        whole run if the server rejects the tools field. Returns True when the
        model finished acting (or hit the act ceiling), False when a model error
        aborted the turn — so an outage is never mistaken for a completed step."""
        acted = False
        nudged = False
        for _ in range(self.cfg.max_acts_per_step):
            if self._native_tools:
                try:
                    reply = complete_act(self.cfg, messages, self._tool_schema,
                                         temperature=self.cfg.temperature)
                except LLMError as e:
                    # The server may not accept the tools field. Drop native for the
                    # rest of this run and retry the SAME turn on the text protocol;
                    # if the error was something else (e.g. no model loaded) the
                    # retry surfaces it cleanly.
                    self._native_tools = False
                    self.log(ui.muted("   (native tool-calling unavailable — "
                                      "falling back to the text protocol)"))
                    messages[0]["content"] = self._system_prompt()
                    continue
            else:
                try:
                    text = chat(self.cfg, messages, temperature=self.cfg.temperature)
                except LLMError as e:
                    self.log(ui.bad(f"   ✖ model error: {e}"))
                    return False
                reply = ActReply(content=text, tool_calls=[])

            # Native path: run every tool_call the model emitted, threading a
            # tool-role result back for each (the shape the native protocol wants).
            if reply.tool_calls:
                acted = True
                messages.append(self._assistant_tool_msg(reply))
                for tc in reply.tool_calls:
                    call = self.registry.build_call(tc["name"], tc["arguments"])
                    self._log_act(call)
                    result = self.registry.dispatch(call, self.ctx)
                    self._log_result(call, result)
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": result})
                continue

            # Text path (fallback, and the only path in text mode): parse an `act`
            # block out of the prose reply.
            reply_text = reply.content
            call = parse_tool_call(reply_text)
            if call is None:
                # No action. Normally that means the model considers the step done.
                # But a weak model often DUMPS a file's content as a plain ```code```
                # block instead of writing it — silently losing the step's work.
                # Nudge once when the reply looks like a lost write (a code fence, the
                # tell-tale dump) or the step did NOTHING. If it really is done, it
                # just repeats and we accept.
                if not nudged and ("```" in (reply_text or "") or not acted):
                    nudged = True
                    self.log(ui.muted("   (no action taken — reminding the model to use its tools)"))
                    messages.append({"role": "assistant", "content": reply_text})
                    messages.append({"role": "user", "content": self._nudge_text()})
                    continue
                return True
            acted = True
            self._log_act(call)
            messages.append({"role": "assistant", "content": reply_text})
            result = self.registry.dispatch(call, self.ctx)
            self._log_result(call, result)
            messages.append({"role": "user", "content": f"[tool result: {call.name}]\n{result}"})
        self.log(ui.muted("   (hit max acts for this step — verifying what we have)"))
        return True

    def _log_act(self, call: ToolCall) -> None:
        self.log("   " + ui.muted("act>") + " " + ui.accent(call.name) + " "
                 + (call.args.get("path", call.args.get("command", "")) or ""))

    def _log_result(self, call: ToolCall, result: str) -> None:
        """Print a one-line RECEIPT for a tool call, so the human sees what actually
        happened — not just that an action was taken. Without this, a write_file
        with no visible outcome (or a silent error) left the user in the dark. The
        tool result otherwise only ever goes to the model. read_file's result IS the
        file content, so it's summarised, not dumped; everything else shows its
        status line, coloured by success/failure."""
        text = result or ""
        if call.name == "read_file":
            n = len(text.splitlines())
            self.log("        " + ui.muted(f"↳ read {call.args.get('path', '')} ({n} lines)"))
            return
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        if not first:
            self.log("        " + ui.muted("↳ (no output)"))
            return
        if len(first) > 120:
            first = first[:119] + "…"
        if first.startswith("OK") or first.startswith("exit=0"):
            paint = ui.ok
        elif first.startswith("ERROR") or first.startswith("exit=") or "not found" in first.lower():
            paint = ui.bad
        else:
            paint = ui.muted
        self.log("        " + paint(f"↳ {first}"))

    @staticmethod
    def _assistant_tool_msg(reply: ActReply) -> dict:
        """Rebuild the assistant message with its tool_calls in OpenAI shape, so the
        tool-role results that follow are accepted by the server."""
        return {
            "role": "assistant",
            "content": reply.content or None,
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"],
                              "arguments": json.dumps(tc["arguments"])}}
                for tc in reply.tool_calls],
        }

    def _nudge_text(self) -> str:
        if self._native_tools:
            return ("You called no tool, so NOTHING was saved. If you meant to create "
                    "or change a file, call the write_file tool. If the step is "
                    "genuinely complete, reply with a short plain sentence.")
        return ("You produced no `act` block, so NOTHING was saved. If you meant to "
                "create or change a file, you MUST use the write_file tool inside an "
                "```act``` block — content shown in a plain code block is discarded. "
                "If the step is genuinely complete, reply again with a short plain sentence.")

    def _work_prompt(self, instruction: str, plan: List[dict], last_verify: str | None) -> str:
        checklist = "\n".join(
            f"- [{'x' if s['done'] else ' '}] {s['text']}" for s in plan
        )
        parts = [
            "Here is the full plan (for context). Work ONLY on the current task.",
            "",
            checklist,
        ]
        # Pin the exact file paths the Definition of Done checks, so the model uses
        # THESE names instead of inventing its own (styles.css vs style.css). Only
        # the mechanical exists/contains paths — never the judge criteria, which a
        # weak model could otherwise game.
        if getattr(self, "_expected_paths", None):
            parts += [
                "",
                "Required file paths — the Definition of Done checks these EXACT "
                "paths. When the current task involves one of these files, use this "
                "exact path; do NOT invent a different name:",
            ] + [f"- {p}" for p in self._expected_paths]
        parts += [
            "",
            f"CURRENT TASK: {instruction}",
        ]
        if last_verify:
            parts += [
                "",
                "Current state of the verification command (this is the truth to fix):",
                "```",
                last_verify,
                "```",
            ]
        parts += ["", "Begin. Read what you need, then make the change."]
        return "\n".join(parts)

    # ---- plan persistence (the externalised memory) -------------------

    def _save_plan(self, goal: str, plan: List[dict]) -> None:
        lines = [f"# Achilles plan", "", f"> Goal: {goal}", ""]
        lines += [f"- [{'x' if s['done'] else ' '}] {s['text']}" for s in plan]
        self.plan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _load_plan(self) -> List[dict]:
        if not self.plan_path.is_file():
            return []
        plan = []
        for line in self.plan_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("- ["):
                done = s[3:4].lower() == "x"
                text = s.split("]", 1)[1].strip()
                if text:
                    plan.append({"done": done, "text": text})
        return plan

    def _stored_goal(self) -> str | None:
        """The goal the persisted plan was written for (its `> Goal:` line)."""
        if not self.plan_path.is_file():
            return None
        for line in self.plan_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("> Goal:"):
                return s[len("> Goal:"):].strip()
        return None

    def _archive_plan(self) -> None:
        """Move a stale/finished plan AND its Definition of Done aside (plan.<n>.md,
        done.<n>.md) so a new goal starts clean while the old ones stay for reference.
        Both share the same <n> so a plan and its contract archive together."""
        if not self.plan_path.is_file():
            return
        n = 1
        while (self.state_dir / f"plan.{n}.md").exists():
            n += 1
        for label, path in (("plan", self.plan_path), ("done", self.dod_path)):
            if path.is_file():
                dest = self.state_dir / f"{label}.{n}.md"
                path.rename(dest)
                self.log(ui.muted(f"   ⤓ archived previous {label} → {dest.name}"))

    def _print_plan(self, plan: List[dict]) -> None:
        self.log(ui.bold("Plan:"))
        for i, s in enumerate(plan, 1):
            self.log("  " + ui.accent(f"{i}.") + f" {s['text']}")

    # ---- Definition of Done (the ceiling) -----------------------------

    def _make_and_save_dod(self, goal: str, tree: str) -> None:
        from .acceptance import make_acceptance, render_acceptance
        try:
            criteria = make_acceptance(self.cfg, goal, tree)
        except LLMError as e:
            self.log(ui.warn(f"⚠  Could not generate a Definition of Done ({e}); "
                     "proceeding with the verification oracle alone."))
            return
        if not criteria:
            self.log(ui.warn("⚠  No Definition of Done produced; proceeding with the oracle alone."))
            return
        self.dod_path.write_text(render_acceptance(goal, criteria), encoding="utf-8")
        self.log("\n" + ui.bold("Definition of Done:"))
        for c in criteria:
            self.log("  " + ui.accent(f"[{c.kind}]") + f" {c.text}")

    def _load_dod(self):
        from .acceptance import parse_acceptance
        if not self.dod_path.is_file():
            return []
        return parse_acceptance(self.dod_path.read_text(encoding="utf-8"))

    def _approve(self) -> str:
        """The user's verdict on the plan: 'yes', 'no', or 'edit'. Auto-approve and
        non-interactive runs proceed."""
        if self.cfg.auto_approve_plan:
            return "yes"
        import sys
        if not sys.stdin.isatty():
            return "yes"  # non-interactive run: proceed
        ans = input("\nProceed with this plan? [Y/n/edit] ").strip().lower()
        if ans in ("", "y", "yes"):
            return "yes"
        if ans in ("e", "edit"):
            return "edit"
        return "no"

    def _approve_loop(self, goal: str, plan: List[dict], tree: str):
        """Approve the plan, allowing MODEL-driven edits that keep the untouched steps.

        'edit' asks you to describe the change in plain words; the model then revises
        the plan — keeping the steps your change doesn't touch — and we re-print and
        re-ask. This replaces the old dead-end ("edit the file and re-run"), which
        rebuilt the whole plan from scratch unless you retyped the goal exactly.
        Returns the plan to execute, or None if declined."""
        while True:
            decision = self._approve()
            if decision == "yes":
                return plan
            if decision == "no":
                self.log("Stopped. Re-run the same goal to resume from the saved plan.")
                return None
            # edit: the model revises the plan from a plain-words instruction.
            instruction = self._ask_edit_instruction()
            if not instruction:
                continue                       # empty → nothing to change, re-ask
            try:
                steps = revise_plan(self.cfg, goal, [s["text"] for s in plan],
                                    instruction, tree)
            except LLMError as e:
                self.log(ui.bad(f"✖  Could not revise the plan: {e}"))
                continue                       # keep the current plan, re-ask
            if not steps:
                self.log(ui.warn("⚠  The model returned no revised steps — keeping the "
                                 "current plan."))
                continue
            plan = [{"done": False, "text": s} for s in steps]
            self._save_plan(goal, plan)
            self.log("")
            self._print_plan(plan)

    def _ask_edit_instruction(self) -> str:
        """Ask the user, in plain words, what to change about the plan."""
        try:
            return input("\nDescribe the change for the model to make "
                         "(plain words, empty to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    # ---- git checkpointing --------------------------------------------

    def _git(self, *args) -> subprocess.CompletedProcess:
        # utf-8 + replace: git output (filenames, commit messages) can carry
        # bytes the Windows default cp1252 cannot decode, which would crash
        # subprocess's pipe-reader thread. Checkpointing must never do that.
        return subprocess.run(["git", *args], cwd=self.ws,
                              capture_output=True, encoding="utf-8",
                              errors="replace")

    def _maybe_git_init(self) -> None:
        if not self.cfg.use_git:
            return
        inside = self._git("rev-parse", "--is-inside-work-tree")
        if inside.returncode != 0:
            self.log(ui.muted("Initialising git repo for checkpoints…"))
            self._git("init")
            self._git("add", "-A")
            res = self._git("commit", "-m", "achilles: initial checkpoint")
            self._warn_if_commit_failed(res, "initial checkpoint failed — commits may not work")

    def _commit(self, message: str) -> None:
        if not self.cfg.use_git:
            return
        self._git("add", "-A")
        res = self._git("commit", "-m", message)
        if res.returncode == 0:
            self.log("   " + ui.paint("⎇", "green") + " " + ui.muted(f"committed: {message}"))
            return
        self._warn_if_commit_failed(res, "commit failed — no rollback checkpoint for this step")

    @staticmethod
    def _commit_had_nothing(res: subprocess.CompletedProcess) -> bool:
        out = (res.stdout or "") + (res.stderr or "")
        return "nothing to commit" in out or "no changes added" in out

    def _warn_if_commit_failed(self, res: subprocess.CompletedProcess, what: str) -> None:
        """A no-op commit (nothing changed) is normal and stays silent. Any other
        failure means checkpointing — Achilles' core rollback promise — is broken,
        so it must be surfaced, not swallowed (Bug 5). The usual real cause is an
        unconfigured git identity (user.name/user.email) in a fresh repo."""
        if res.returncode == 0 or self._commit_had_nothing(res):
            return
        out = ((res.stderr or "") + (res.stdout or "")).strip()
        detail = out.splitlines()[-1] if out else f"git exited {res.returncode}"
        self.log(ui.warn(f"   ⚠ git {what}: {detail}"))
