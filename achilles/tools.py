"""
tools.py — the tools system: the model's hands, now an extensible registry.

Achilles used to hardcode exactly four tools. Now they live in a REGISTRY, so the
same act-protocol can drive any number of tools:

  * built-in   — the four hands below (read/write/list/run),
  * manifest   — declared in achilles.toml as `[[tool]]` (a name + a command
                 template, NO code) — anyone can add one,
  * plugin     — a Python file in `tools_dir` that registers richer tools
                 (your static_site oracle, BreadCraft's .crumb checker, …).

protocol.py already routes purely by NAME; this module was the only place that
knew "there are exactly four", and now it doesn't. The model's prompt is built
FROM the registry, so a newly added tool announces itself automatically.

All file paths stay confined to the workspace. The shell is not sandboxed — same
honest stance as before — which is exactly why every green step is git-committed.
"""

import importlib.util
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .protocol import ToolCall

# Cap any single tool result so one noisy command can't blow the context window.
MAX_OUTPUT_CHARS = 8000

# Above this many lines, write_file appends a gentle split-it-up hint (backlog #14).
# A big file overflows read_file's cap (whole-file rewrites then lose the tail) and
# the judge's per-file budget (true criteria FAIL unseen), so we nudge toward
# separate files while the context is still warm.
_LARGE_FILE_LINES = 200


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.5)]
    tail = text[-int(limit * 0.4):]
    return f"{head}\n... [{len(text) - len(head) - len(tail)} chars trimmed] ...\n{tail}"


class ToolContext:
    """What every tool is handed: the jailed workspace plus shared helpers, so a
    built-in, a manifest tool and a plugin all resolve paths and run the shell
    the same safe way."""

    def __init__(self, workspace: Path):
        self.ws = workspace.resolve()

    def resolve(self, rel: str) -> Path:
        """Join a model-supplied path to the workspace and refuse to escape it."""
        p = (self.ws / (rel or "").strip()).resolve()
        if self.ws not in p.parents and p != self.ws:
            raise ValueError(f"path '{rel}' escapes the workspace")
        return p

    def truncate(self, text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
        return _truncate(text, limit)

    def shell(self, command: str, timeout: int = 120) -> str:
        """Run a shell command in the workspace. Returns exit code + combined output."""
        try:
            # utf-8 + replace, not the platform default: on Windows text=True
            # decodes as cp1252 and a single stray byte in a command's output
            # crashes the pipe-reader thread. Never let output encoding halt work.
            proc = subprocess.run(command, shell=True, cwd=self.ws,
                                  capture_output=True, encoding="utf-8",
                                  errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {timeout}s: {command}"
        out = (proc.stdout or "") + (proc.stderr or "")
        return _truncate(f"exit={proc.returncode}\n{out}".strip())


def _obj(props: dict, required=()) -> dict:
    """A minimal JSON-Schema object, for native (OpenAI) tool definitions."""
    return {"type": "object", "properties": props, "required": list(required)}


def _str(desc: str) -> dict:
    return {"type": "string", "description": desc}


@dataclass
class Tool:
    """One capability. `run(args, body, ctx) -> str`; `usage` is the act-block
    example shown to the model. `act=False` marks a CHECK-only tool: dispatchable
    (e.g. by the acceptance phase) but hidden from the act-loop prompt, so the
    model's "hands" stay uncluttered. A check tool returns an `exit=0` line on
    pass (same convention as run_command) so acceptance can read its verdict.

    `parameters` is the JSON-Schema of the tool's arguments, used when we offer
    NATIVE tool-calling (the OpenAI `tools` field) to a tool-tuned model. Left
    None, a permissive schema is synthesised. `body_param` names the one schema
    field that carries the freeform BODY (write_file's file content) rather than a
    header arg — so a native tool_call can be mapped back onto the run(args, body)
    contract the text protocol already uses."""
    name: str
    description: str
    run: Callable[[dict, Optional[str], ToolContext], str]
    usage: str = ""
    act: bool = True
    parameters: Optional[dict] = None
    body_param: Optional[str] = None


# ---- the four built-in hands ---------------------------------------------

def _read_file(args, body, ctx: ToolContext) -> str:
    path = args.get("path", "")
    p = ctx.resolve(path)
    if not p.is_file():
        return f"ERROR: no such file: {path}"
    return ctx.truncate(p.read_text(encoding="utf-8", errors="replace"))


def _write_file(args, body, ctx: ToolContext) -> str:
    # An empty path resolves to the workspace root, whose write_text then fails with
    # a cryptic "Permission denied" on the directory. Reject it up front with an
    # actionable message so the model retries with a real path in the SAME step,
    # instead of losing the step's work to a swallowed directory-write error.
    path = (args.get("path") or "").strip()
    if not path:
        return "ERROR: write_file needs a non-empty 'path' (with the file body in 'content')."
    p = ctx.resolve(path)
    if p.is_dir():
        return f"ERROR: '{path}' is a directory, not a file — give a file path."
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body or "", encoding="utf-8")
    n = (body or "").count("\n") + 1
    ok = f"OK: wrote {path} ({n} lines)"
    if n > _LARGE_FILE_LINES:
        # A second line, so the receipt's first line stays a clean green "OK".
        ok += ("\nNOTE: this file is large. If it mixes separable concerns (e.g. an "
               "HTML page with inline CSS and JS), splitting the CSS into a .css file "
               "and the JS into a .js file — linked from the HTML — keeps each file "
               "small and easier to edit correctly next time.")
    return ok


def _list_dir(args, body, ctx: ToolContext) -> str:
    path = args.get("path", ".") or "."
    p = ctx.resolve(path)
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"
    rows = []
    for entry in sorted(p.iterdir()):
        if entry.name in {".git", ".achilles", "__pycache__"}:
            continue
        rows.append(f"{'dir ' if entry.is_dir() else 'file'}  {entry.name}")
    return "\n".join(rows) or "(empty)"


def _run_command(args, body, ctx: ToolContext) -> str:
    cmd = args.get("command") or (body or "").strip()
    return ctx.shell(cmd)


# Check-only tools (act=False): not shown as "hands", but the acceptance phase
# dispatches them. Each returns an `exit=0` line on pass so a verdict is readable.

def _file_exists(args, body, ctx: ToolContext) -> str:
    try:
        p = ctx.resolve(args.get("path", ""))
    except ValueError:
        return f"exit=1\npath escapes the workspace: {args.get('path', '')}"
    return "exit=0" if p.exists() else f"exit=1\nfile not found: {args.get('path', '')}"


def _file_absent(args, body, ctx: ToolContext) -> str:
    try:
        p = ctx.resolve(args.get("path", ""))
    except ValueError:
        return f"exit=1\npath escapes the workspace: {args.get('path', '')}"
    return "exit=0" if not p.exists() else f"exit=1\nfile should be absent: {args.get('path', '')}"


def _file_contains(args, body, ctx: ToolContext) -> str:
    path = args.get("path", "")
    needle = args.get("text", body or "")
    try:
        p = ctx.resolve(path)
    except ValueError:
        return f"exit=1\npath escapes the workspace: {path}"
    if not p.is_file():
        return f"exit=1\nfile not found: {path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    return "exit=0" if needle in text else f"exit=1\n'{needle}' not found in {path}"


BUILTINS = [
    Tool("read_file", "read a file's contents", _read_file,
         usage="```act\ntool: read_file\npath: src/foo.py\n```",
         parameters=_obj({"path": _str("file path, relative to the workspace")},
                         ["path"])),
    Tool("list_dir", "list a directory", _list_dir,
         usage="```act\ntool: list_dir\npath: .\n```",
         parameters=_obj({"path": _str("directory path, relative to the workspace; "
                                       "'.' for the root")})),
    Tool("run_command", "run a shell command; see its output and exit code", _run_command,
         usage="```act\ntool: run_command\ncommand: python -m pytest -q\n```",
         parameters=_obj({"command": _str("the shell command to run")}, ["command"])),
    Tool("write_file", "write a file — the body REPLACES the whole file", _write_file,
         usage="```act\ntool: write_file\npath: src/foo.py\n---\ndef foo():\n    return 42\n```",
         parameters=_obj({"path": _str("file path, relative to the workspace"),
                          "content": _str("the COMPLETE new file contents; replaces "
                                          "the whole file")}, ["path", "content"]),
         body_param="content"),
    # check-only (acceptance), hidden from the act-loop prompt:
    Tool("file_exists", "pass if a file exists. arg: path", _file_exists, act=False),
    Tool("file_absent", "pass if a file does NOT exist. arg: path", _file_absent, act=False),
    Tool("file_contains", "pass if a file contains text. args: path, text", _file_contains, act=False),
]


# ---- the registry --------------------------------------------------------

class Registry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name.lower()] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def dispatch(self, call: ToolCall, ctx: ToolContext) -> str:
        tool = self._tools.get(call.name.lower())
        if tool is None:
            return (f"ERROR: unknown tool '{call.name}'. "
                    f"Valid tools: {', '.join(self._tools)}.")
        try:
            return tool.run(call.args, call.body, ctx)
        except Exception as e:   # never let a tool crash the loop
            return f"ERROR: {call.name} failed: {e}"

    def build_call(self, name: str, arguments: dict) -> ToolCall:
        """Turn a NATIVE tool_call (name + parsed JSON arguments) into the same
        ToolCall the text protocol produces, so dispatch() is format-agnostic. The
        tool's body_param field (if any) is routed to ToolCall.body; the rest stay
        as header args."""
        tool = self._tools.get((name or "").lower())
        args = dict(arguments or {})
        body = None
        if tool and tool.body_param and tool.body_param in args:
            body = args.pop(tool.body_param)
            if body is not None and not isinstance(body, str):
                body = str(body)
        return ToolCall(name=name, args=args, body=body)

    def tool_schemas(self) -> list:
        """The act-tools as OpenAI `tools` definitions, for native tool-calling.
        Tools without an explicit schema get a permissive one so they still appear
        (a native model just gets less guidance on their args)."""
        out = []
        for t in self._tools.values():
            if not t.act:
                continue
            params = t.parameters or {"type": "object", "properties": {},
                                      "additionalProperties": True}
            out.append({"type": "function",
                        "function": {"name": t.name,
                                     "description": t.description,
                                     "parameters": params}})
        return out

    def content_json_schema(self) -> dict:
        """The act-loop's constrained-decoding schema, for act_protocol="json": a
        single FLAT object whose `tool` enum is the act tools plus the "finish"
        sentinel (how a grammar-forced reply signals "step done"), with every tool's
        argument fields as optional strings. A weak model's reply is grammar-forced
        into this shape on the content channel — the channel LM Studio actually
        enforces. Flat + only `tool` required is deliberate: it survives strict mode
        and needs no per-tool oneOf grammar (both verified against LM Studio)."""
        tools = [t for t in self._tools.values() if t.act]
        props: dict = {"tool": {"type": "string",
                                "enum": [t.name for t in tools] + ["finish"]}}
        for t in tools:
            for field in ((t.parameters or {}).get("properties") or {}):
                props.setdefault(field, {"type": "string"})
            if t.body_param:
                props.setdefault(t.body_param, {"type": "string"})
        return {"type": "object", "properties": props,
                "required": ["tool"], "additionalProperties": False}

    def describe(self, include_usage: bool = True) -> str:
        """Render the tool list for the model's system prompt — this is how a new
        tool announces itself without any prompt edit. In native mode the act-block
        `usage` examples are omitted (include_usage=False): the JSON schema carries
        the arg shape, and showing the text format would only muddy the water."""
        parts = []
        for t in self._tools.values():
            if not t.act:        # check-only tools are not "hands"
                continue
            block = f"- {t.name}: {t.description}"
            if include_usage and t.usage:
                block += "\n" + t.usage
            parts.append(block)
        return "\n\n".join(parts)


# ---- user tools: manifest (TOML) and plugins (Python) --------------------

def _manifest_tool(spec: dict) -> Tool:
    """A [[tool]] from achilles.toml: name + a command template with {arg}
    placeholders (and optional {body}). No code required."""
    name = spec["name"]
    template = spec["command"]

    def run(args, body, ctx, _t=template):
        cmd = _t
        for k, v in (args or {}).items():
            cmd = cmd.replace("{" + k + "}", str(v))
        if body is not None:
            cmd = cmd.replace("{body}", body)
        return ctx.shell(cmd)

    usage = spec.get("usage") or f"```act\ntool: {name}\n# fills args into: {template}\n```"
    # Derive a native schema from the template's {placeholders}: each is a string
    # arg, except {body} which maps to the freeform body (like write_file's).
    placeholders = re.findall(r"\{(\w+)\}", template)
    props = {p: _str(f"value for {{{p}}}") for p in placeholders if p != "body"}
    body_param = "body" if "body" in placeholders else None
    if body_param:
        props["body"] = _str("freeform body substituted for {body}")
    parameters = _obj(props, [p for p in props if p != "body"]) if props else None
    return Tool(name, spec.get("description", ""), run, usage,
                parameters=parameters, body_param=body_param)


def _load_plugins(dir_path: Path, reg: "Registry", log) -> None:
    """Import every *.py in dir_path and let it register tools, via either a
    `register(registry)` function or a module-level `TOOLS = [Tool(...)]`."""
    if not dir_path.is_dir():
        return
    for f in sorted(dir_path.glob("*.py")):
        if f.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f.stem, f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "register"):
                mod.register(reg)
            elif hasattr(mod, "TOOLS"):
                for t in mod.TOOLS:
                    reg.register(t)
            else:
                log(f"⚠  tool plugin {f.name}: no register() or TOOLS — skipped")
        except Exception as e:
            log(f"⚠  failed to load tool plugin {f.name}: {e}")


def build_registry(config, log=print) -> Registry:
    """Assemble the registry: built-ins, then manifest tools, then plugins.
    Later registrations override earlier ones of the same name."""
    reg = Registry()
    for t in BUILTINS:
        reg.register(t)
    for spec in getattr(config, "tools", None) or []:
        try:
            reg.register(_manifest_tool(spec))
        except Exception as e:
            log(f"⚠  bad [[tool]] entry {spec!r}: {e}")
    tools_dir = getattr(config, "tools_dir", "")
    if tools_dir:
        d = Path(tools_dir)
        if not d.is_absolute():
            d = config.workspace_path / d
        _load_plugins(d, reg, log)
    # ComfyUI image generation is opt-in: only when comfy_url is set does the
    # model get a generate_image hand. Imported lazily so the core never depends
    # on the comfy stack.
    if getattr(config, "comfy_url", ""):
        try:
            from .comfy import build_tool
            reg.register(build_tool(config))
        except Exception as e:
            log(f"⚠  could not enable generate_image: {e}")
    return reg
