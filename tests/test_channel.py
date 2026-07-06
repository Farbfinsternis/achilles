"""Tests for the TerminalChannel batch-safety contract.

A non-interactive stdin (CI, piped input) must never block a gate: the interview
takes defaults (skip) and an approval auto-proceeds. Auto-approve does the same
even on a tty. These pin the "batch runs never hang on input()" invariant.
"""

import types

import achilles.channel as channel_mod
from achilles.channel import TerminalChannel


def _stdin(monkeypatch, isatty):
    monkeypatch.setattr(channel_mod.sys, "stdin",
                        types.SimpleNamespace(isatty=lambda: isatty))


def test_non_tty_interview_skips(monkeypatch):
    _stdin(monkeypatch, isatty=False)
    c = TerminalChannel(log=lambda *a: None)
    assert c.request("interview.question", {"prompt": "x", "default": "d"}) == {"skip": True}


def test_non_tty_approval_approves(monkeypatch):
    _stdin(monkeypatch, isatty=False)
    c = TerminalChannel(log=lambda *a: None)
    assert c.request("approval.request", {"subject": "spec"}) == {"decision": "approve"}


def test_auto_approve_approves_on_tty(monkeypatch):
    _stdin(monkeypatch, isatty=True)
    c = TerminalChannel(log=lambda *a: None, auto_approve=True)
    assert c.request("approval.request", {"subject": "plan"}) == {"decision": "approve"}


def test_emit_log_routes_to_log():
    seen = []
    c = TerminalChannel(log=lambda t: seen.append(t))
    c.emit("log", {"text": "hello"})
    assert seen == ["hello"]
