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

import subprocess
from pathlib import Path
from typing import Callable, List

from .config import Config
from .llm import chat, LLMError
from .planner import make_plan
from .protocol import parse_tool_call, ToolCall
from .tools import build_registry, ToolContext
from . import style as ui


# The tool list is generated from the registry (so a newly added tool announces
# itself with no prompt edit). {tools} is filled in at run time.
EXECUTE_SYSTEM_TEMPLATE = """You are Achilles, a coding agent working in ONE small step of a larger plan.

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
  not mistaken for the end of your block."""


class Harness:
    def __init__(self, config: Config, log: Callable[[str], None] = print):
        self.cfg = config
        self.log = log
        self.ws = config.workspace_path
        self.ctx = ToolContext(self.ws)
        self.registry = build_registry(config, log)
        self.state_dir = self.ws / ".achilles"
        self.plan_path = self.state_dir / "plan.md"
        self.dod_path = self.state_dir / "done.md"

    # ---- public entry point -------------------------------------------

    def run(self, goal: str) -> bool:
        self.state_dir.mkdir(exist_ok=True)
        self._maybe_git_init()

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
            if not self._approve():
                self.log("Stopped. Edit .achilles/plan.md (and done.md) and re-run to continue.")
                return False

        # STATE: EXECUTE.
        return self._execute(goal, plan)

    # ---- execution: FLOOR (oracle) then CEILING (Definition of Done) --

    def _execute(self, goal: str, plan: List[dict]) -> bool:
        """Three phases. (1) Work the WHOLE plan — no early-exit on first green;
        the plan carries the intent and discarding it is what crippled generative
        tasks. (2) The FLOOR: a real oracle must be green. (3) The CEILING: the
        Definition of Done, judged and fix-looped until met. Absent a DoD, the
        floor (or a finished plan) is the whole story — the configured fallback."""
        ok, last_verify = self._work_through_plan(goal, plan)
        if not ok:
            return False

        if self.cfg.verify_command:
            ok, last_verify = self._ensure_floor_green(goal, plan)
            if not ok:
                return False

        criteria = self._load_dod()
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
        system = EXECUTE_SYSTEM_TEMPLATE.format(tools=self.registry.describe())
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        return self._act_until_done(messages)

    def _act_until_done(self, messages: List[dict]) -> bool:
        """Inner loop: let the model take actions until it stops acting.

        Returns True when the model finished acting normally (or hit the act
        ceiling), False when a model error aborted the turn — the distinction the
        caller needs so an outage can't be mistaken for a completed step."""
        acted = False
        nudged = False
        for _ in range(self.cfg.max_acts_per_step):
            try:
                reply = chat(self.cfg, messages, temperature=self.cfg.temperature)
            except LLMError as e:
                self.log(ui.bad(f"   ✖ model error: {e}"))
                return False
            call = parse_tool_call(reply)
            if call is None:
                # No `act` block. Normally that means the model considers the step
                # done. But a weak model often DUMPS a file's content as a plain
                # ```code``` block instead of a write_file act — which the harness
                # never saves, silently losing the whole step's work. Nudge once
                # toward the protocol when the reply looks like a lost write: it
                # carries a code fence (the tell-tale dump), or the step did
                # NOTHING at all. If it really is done, it just repeats and we accept.
                if not nudged and ("```" in (reply or "") or not acted):
                    nudged = True
                    self.log(ui.muted("   (no action taken — reminding the model to use write_file)"))
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content":
                        "You produced no `act` block, so NOTHING was saved. If you "
                        "meant to create or change a file, you MUST use the write_file "
                        "tool inside an ```act``` block — content shown in a plain code "
                        "block is discarded. If the step is genuinely complete, reply "
                        "again with a short plain sentence."})
                    continue
                return True
            acted = True
            self.log("   " + ui.muted("act>") + " " + ui.accent(call.name) + " "
                     + call.args.get('path', call.args.get('command', '')))
            messages.append({"role": "assistant", "content": reply})
            result = self.registry.dispatch(call, self.ctx)
            messages.append({"role": "user", "content": f"[tool result: {call.name}]\n{result}"})
        self.log(ui.muted("   (hit max acts for this step — verifying what we have)"))
        return True

    def _work_prompt(self, instruction: str, plan: List[dict], last_verify: str | None) -> str:
        checklist = "\n".join(
            f"- [{'x' if s['done'] else ' '}] {s['text']}" for s in plan
        )
        parts = [
            "Here is the full plan (for context). Work ONLY on the current task.",
            "",
            checklist,
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

    def _approve(self) -> bool:
        if self.cfg.auto_approve_plan:
            return True
        import sys
        if not sys.stdin.isatty():
            return True  # non-interactive run: proceed
        ans = input("\nProceed with this plan? [Y/n/edit] ").strip().lower()
        if ans in ("", "y", "yes"):
            return True
        if ans in ("e", "edit"):
            self.log(f"Edit {self.plan_path} then re-run.")
        return False

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
