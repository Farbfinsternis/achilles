"""
repl.py — the interactive session.

One concept: a running conversation with Achilles. You type goals in plain
words; each goal is handed to a fresh `Harness.run()`, so the act/verify/commit
loop is exactly the one-shot CLI's loop. What *persists* across goals within a
session is the config (model, workspace, verify command) and the git repo — so
you can line up several goals, switch models, or retarget the workspace without
restarting the process.

Lines beginning with ':' are meta-commands. They steer the SESSION and never
reach the model. Everything else is a goal narrated in plain words.
"""

from pathlib import Path

from .config import Config
from .harness import Harness
from . import style as ui
from . import comfy_client as cc
from . import lmstudio
from . import workflows as wf
from .comfy import _store_dir


BANNER = (
    ui.paint("Achilles", "bold", "cyan")
    + " — interactive session. Type a goal in plain words, or "
    + ui.accent(":help") + " for commands.\n"
    + ui.muted("(end a line with \\ to continue on the next; :quit or Ctrl-D to leave)")
)

HELP = """Commands (everything else is treated as a goal for the model):
  :help                 show this
  :config               show the current session config
  :model [<id>]         show, or switch to, the model used for this session
  :verify [<command>]   show, or set, the oracle command run after each step
  :workspace [<path>]   show, or change, the project directory
  :plan                 print the current .achilles/plan.md
  :status               how many plan steps are done
  :tools                print the exact tool list the model is given
  :workflow …           manage ComfyUI image workflows (see :workflow help)
  :quit                 leave the session"""

WORKFLOW_HELP = """:workflow — register ComfyUI workflows so Achilles can make images.
  :workflow list                    show registered workflows (★ = default)
  :workflow register <path> [name]  validate a marked ComfyUI API-format export
                                    (title a node achilles:prompt / achilles:aspect)
                                    and save it if the markers resolve
  :workflow try <path> [name]       let the model find the prompt/aspect nodes
                                    itself (no marking needed); saves as _adhoc
                                    (or [name]) and makes it the default
  :workflow default <name>          the workflow used when the model names none
  :workflow rm <name>               remove a workflow"""


class Repl:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self) -> int:
        print(BANNER)
        # Resolve the model BEFORE showing the config: ensure_loaded adopts the real
        # loaded model-key over the "local-model" placeholder (and warms a cold LM
        # Studio), so the config table shows the model that will actually be used.
        lmstudio.ensure_loaded(
            self.cfg,
            lambda m: print(ui.warn(m) if "⚠" in m else ui.muted(m)))
        self._show_config()
        while True:
            try:
                line = self._read_line()
            except EOFError:
                print()
                break
            if not line:
                continue
            if line.startswith(":"):
                if self._meta(line):
                    break
                continue
            # A plain line is a goal. Run it through the same loop as the CLI.
            # The plan is keyed to the goal, so re-typing it resumes; a new goal
            # starts fresh (the Harness handles that).
            try:
                Harness(self.cfg).run(line)
            except KeyboardInterrupt:
                print("\n(interrupted — progress is saved; re-type the same goal to resume)")
        print("bye.")
        return 0

    def _read_line(self) -> str:
        """Read one goal, allowing MULTI-LINE entry: a line ending with a backslash
        continues on the next, and the break becomes a real newline in the goal. A
        plain Enter still submits a single-line goal, exactly as before — so nothing
        changes for the common case. Ctrl-D on the FIRST line quits; on a
        continuation it submits whatever was typed so far."""
        prompt = "\n" + ui.paint("achilles>", "bold", "cyan") + " "
        cont = ui.paint("      ...>", "bold", "cyan") + " "
        buf: list[str] = []
        while True:
            try:
                chunk = input(prompt if not buf else cont)
            except EOFError:
                if buf:
                    break            # don't lose a half-typed multi-line goal
                raise
            if chunk.endswith("\\"):
                buf.append(chunk[:-1])
                continue
            buf.append(chunk)
            break
        return "\n".join(buf).strip()

    # ---- meta-commands ------------------------------------------------

    def _meta(self, line: str) -> bool:
        """Handle a ':' command. Returns True if the session should quit."""
        parts = line[1:].split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("q", "quit", "exit"):
            return True
        if cmd in ("h", "help", "?"):
            print(HELP)
        elif cmd == "config":
            self._show_config()
        elif cmd == "model":
            if arg:
                self.cfg.model = arg
                print(f"   model → {arg}")
            else:
                print(f"   model = {self.cfg.model}")
        elif cmd == "verify":
            if arg:
                self.cfg.verify_command = arg
                print(f"   verify_command → {arg}")
            else:
                print(f"   verify_command = {self.cfg.verify_command or '(none — flying blind)'}")
        elif cmd in ("workspace", "ws", "cd"):
            if arg:
                p = Path(arg).expanduser()
                if p.is_dir():
                    self.cfg.workspace = str(p)
                    print(f"   workspace → {self.cfg.workspace_path}")
                else:
                    print(ui.bad(f"   ✖ not a directory: {p}"))
            else:
                print(f"   workspace = {self.cfg.workspace_path}")
        elif cmd == "plan":
            self._show_plan()
        elif cmd == "status":
            self._show_status()
        elif cmd in ("workflow", "wf"):
            self._workflow(arg)
        elif cmd == "tools":
            self._show_tools(arg)
        else:
            print(ui.bad(f"   ✖ unknown command ':{cmd}'") + ui.muted(" — try :help"))
        return False

    # ---- read-only views ----------------------------------------------

    def _show_config(self) -> None:
        c = self.cfg
        def row(label, value):
            print("   " + ui.muted(f"{label:<14}") + " " + ui.accent(str(value)))
        row("model", c.model)
        row("base_url", c.base_url)
        row("workspace", c.workspace_path)
        row("verify_command", c.verify_command or ui.warn("(none — flying blind)"))
        row("use_git", c.use_git)
        if c.comfy_url:
            default = self._store().get_default()
            wf_note = f"default: {default}" if default else ui.warn("no default workflow — :workflow default <name>")
            row("image gen", f"{c.comfy_url}  ({wf_note})")
        else:
            row("image gen", ui.warn("off — set comfy_url to enable generate_image"))

    def _show_tools(self, arg: str = "") -> None:
        """The definitive check for 'does the model even see generate_image?'.
        Compact by default (just the names); `:tools all` dumps the full block
        the model actually receives."""
        from .tools import build_registry
        reg = build_registry(self.cfg, lambda *_: None)
        names = reg.names()
        print("   " + ui.muted("tools given to the model: ") + ui.accent(", ".join(names)))
        if "generate_image" in names:
            print(ui.ok("   ✔ generate_image is available."))
        else:
            print(ui.warn("   ⚠ generate_image is NOT offered — check :config "
                          "(image gen off? comfy_url unset?)."))
        if arg.strip().lower() in ("all", "full", "-v"):
            print()
            print(reg.describe())
        else:
            print(ui.muted("   (:tools all for the full tool block the model sees)"))

    def _show_plan(self) -> None:
        h = Harness(self.cfg)
        if not h.plan_path.is_file():
            print("   (no plan yet — type a goal to make one)")
            return
        print(h.plan_path.read_text(encoding="utf-8"))

    def _show_status(self) -> None:
        h = Harness(self.cfg)
        plan = h._load_plan()
        if not plan:
            print("   (no plan yet — type a goal to make one)")
            return
        done = sum(1 for s in plan if s["done"])
        goal = h._stored_goal()
        if goal:
            print(f"   goal: {goal}")
        print(f"   {done}/{len(plan)} steps done")

    # ---- ComfyUI workflow registry ------------------------------------

    def _store(self) -> wf.Store:
        return wf.Store(_store_dir(self.cfg))

    def _object_info(self):
        """Ask the configured ComfyUI for its installed node types (to verify a
        workflow's custom nodes and read enum options). None if unreachable — the
        register still works, just with name-hint detection and no node check."""
        if not self.cfg.comfy_url:
            return None
        client = cc.ComfyClient(self.cfg.comfy_url)
        if not client.reachable():
            print(ui.warn("   ⚠ ComfyUI not reachable — registering without the "
                          "custom-node check (name-hint detection only)."))
            return None
        try:
            return client.object_info()
        except cc.ComfyError as e:
            print(ui.warn(f"   ⚠ could not read /object_info ({e}) — continuing."))
            return None

    def _workflow(self, arg: str) -> None:
        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""
        store = self._store()

        if sub in ("help", "?"):
            print(WORKFLOW_HELP)
        elif sub in ("list", "ls", ""):
            self._workflow_list(store)
        elif sub in ("register", "add"):
            self._workflow_register(store, rest)
        elif sub in ("try", "adhoc"):
            self._workflow_try(store, rest)
        elif sub in ("default", "std"):
            self._workflow_default(store, rest)
        elif sub in ("rm", "remove", "del"):
            self._workflow_remove(store, rest)
        else:
            print(ui.bad(f"   ✖ unknown :workflow subcommand '{sub}'")
                  + ui.muted(" — try :workflow help"))

    def _workflow_list(self, store: wf.Store) -> None:
        names = store.names()
        if not names:
            print("   (no workflows registered — :workflow register <path>)")
            return
        default = store.get_default()
        for n in names:
            star = ui.accent(" ★ default") if n == default else ""
            meta = store.load_meta(n)
            slots = []
            if meta.prompt:
                slots.append("prompt")
            if meta.resolution:
                slots.append(f"aspect:{','.join(meta.resolution.mapping) or 'none'}")
            print(f"   {ui.bold(n)}{star}  " + ui.muted(" · ".join(slots)))

    def _workflow_register(self, store: wf.Store, rest: str) -> None:
        if not rest:
            print(ui.bad("   ✖ usage: :workflow register <path> [name]"))
            return
        # Tolerate a quoted path (with or without spaces) plus an optional name.
        path_str, name = _split_path_and_name(rest)
        path = Path(path_str).expanduser()
        if not path.is_file():
            print(ui.bad(f"   ✖ no such file: {path}"))
            return
        name = name or _slug(path.stem)
        try:
            report = wf.register(store, path, name, self._object_info())
        except wf.WorkflowError as e:
            print(ui.bad(f"   ✖ {e}"))
            return
        # Echo what the markers resolved to (the human's mirror), then accept/reject.
        for line in report.echo:
            print("     " + ui.muted(line))
        if not report.ok:
            print(ui.bad(f"   ✖ not registered — {len(report.errors)} problem(s):"))
            for err in report.errors:
                print(ui.bad(f"       · {err}"))
            return
        print(ui.ok(f"   ✔ registered '{name}'"))
        if report.unsupported_aspects:
            print(ui.warn("     note: no mapping for "
                          + ", ".join(report.unsupported_aspects)
                          + " — those aspects are refused at render time."))
        if not store.get_default():
            store.set_default(name)
            print(ui.muted(f"     (set as default — first workflow)"))

    def _workflow_try(self, store: wf.Store, rest: str) -> None:
        """Ad-hoc import: the model nominates the prompt/aspect nodes, we validate
        that nomination and save under _adhoc (or a given name) as the default."""
        if not rest:
            print(ui.bad("   ✖ usage: :workflow try <path> [name]"))
            return
        path_str, name = _split_path_and_name(rest)
        path = Path(path_str).expanduser()
        if not path.is_file():
            print(ui.bad(f"   ✖ no such file: {path}"))
            return
        name = name or "_adhoc"
        print(ui.muted("   … asking the model to find the prompt & aspect nodes"))
        try:
            report = wf.register_adhoc(store, path, name, self.cfg, self._object_info())
        except wf.WorkflowError as e:
            print(ui.bad(f"   ✖ {e}"))
            return
        # Warnings first (double expansion etc.), then the resolved-slot mirror.
        for note in report.notes:
            print(ui.warn(f"   ⚠ {note}"))
        for line in report.echo:
            print("     " + ui.muted(line))
        if not report.ok:
            print(ui.bad(f"   ✖ not registered — {len(report.errors)} problem(s):"))
            for err in report.errors:
                print(ui.bad(f"       · {err}"))
            print(ui.muted("     (the model's guess did not resolve — mark the nodes "
                           "by hand and use :workflow register)"))
            return
        print(ui.ok(f"   ✔ imported '{name}'"))
        if report.unsupported_aspects:
            print(ui.warn("     note: no mapping for "
                          + ", ".join(report.unsupported_aspects)
                          + " — those aspects are refused at render time."))
        store.set_default(name)
        print(ui.muted(f"     (set as default)"))

    def _workflow_default(self, store: wf.Store, name: str) -> None:
        try:
            store.set_default(name)
            print(ui.ok(f"   ✔ default → {name}"))
        except wf.WorkflowError as e:
            print(ui.bad(f"   ✖ {e}"))

    def _workflow_remove(self, store: wf.Store, name: str) -> None:
        if not store.exists(name):
            print(ui.bad(f"   ✖ no workflow named '{name}'"))
            return
        store.remove(name)
        print(ui.ok(f"   ✔ removed {name}"))


def _split_path_and_name(rest: str):
    """Parse `<path> [name]`, honouring a quoted path so spaces don't split it."""
    rest = rest.strip()
    if rest and rest[0] in "\"'":
        close = rest.find(rest[0], 1)
        if close != -1:
            return rest[1:close], rest[close + 1:].strip()
    bits = rest.split()
    if len(bits) >= 2:
        return bits[0], bits[1]
    return rest, ""


def _slug(stem: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-").lower() or "workflow"
