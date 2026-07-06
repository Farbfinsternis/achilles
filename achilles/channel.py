"""
channel.py — the engine/UI boundary (see docs/protocol.md §5).

The harness never talks to a terminal or a socket directly. It talks to a
`Channel` with two methods:

    emit(type, data)          fire-and-forget event  (engine → UI)
    request(type, data)       blocking gate: event + await a reply

`request` hides the async from the harness — it blocks until the reply is in and
returns it as a dict, so run() stays linear code. A terminal client answers via
input(); a future web client answers over a WebSocket. Same engine, no
`if terminal:` in the harness.

v1 wires the NEW interactive points (the interview and the spec approval gate)
through this seam. The existing plan-approval path still uses input() directly;
migrating it — and typing the log stream into semantic events — is the later
mechanical pass noted in the protocol.
"""

import sys


class Channel:
    """The abstract boundary. Subclass per transport."""

    def emit(self, type: str, data: dict) -> None:
        raise NotImplementedError

    def request(self, type: str, data: dict) -> dict:
        raise NotImplementedError


class TerminalChannel(Channel):
    """The terminal client: emit prints, request prompts via input().

    `auto_approve` (from -y / config) and a non-interactive stdin both short-circuit
    a gate to "proceed" — an interview takes all defaults, an approval auto-approves
    — so batch/CI runs never block on input(). This policy is terminal-specific and
    lives here, not in the harness."""

    def __init__(self, log=print, auto_approve: bool = False):
        self.log = log
        self.auto_approve = auto_approve

    # ---- events -------------------------------------------------------
    def emit(self, type: str, data: dict) -> None:
        if type == "log":
            self.log(data.get("text", ""))
        # Other semantic event types (step.started, verify.result, …) are not yet
        # emitted; the terminal already prints those lines via self.log elsewhere.

    # ---- gates --------------------------------------------------------
    def request(self, type: str, data: dict) -> dict:
        if type == "interview.question":
            return self._ask_interview(data)
        if type == "approval.request":
            return self._ask_approval(data)
        return {}

    @staticmethod
    def _tty() -> bool:
        stdin = getattr(sys, "stdin", None)
        return bool(stdin and stdin.isatty())

    def _ask_interview(self, data: dict) -> dict:
        if not self._tty():
            return {"skip": True}                     # non-interactive → slot default
        prompt = data.get("prompt", "?")
        default = data.get("default", "")
        hint = f"  (Enter = Default: {default})" if default else "  (Enter = überspringen)"
        try:
            value = input(f"\n{prompt}{hint}\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return {"skip": True}
        return {"value": value}

    def _ask_approval(self, data: dict) -> dict:
        if self.auto_approve or not self._tty():
            return {"decision": "approve"}
        subject = data.get("subject", "plan")
        content = data.get("content")
        if content:
            self.log(content)
        try:
            ans = input(f"\nProceed with this {subject}? [Y/n/edit] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return {"decision": "approve"}
        if ans in ("", "y", "yes"):
            return {"decision": "approve"}
        if ans in ("e", "edit"):
            try:
                instr = input("Describe the change (plain words, empty to cancel): ").strip()
            except (EOFError, KeyboardInterrupt):
                instr = ""
            return {"decision": "edit", "instruction": instr}
        return {"decision": "reject"}
