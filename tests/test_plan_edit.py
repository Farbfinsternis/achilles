"""
Plan approval with MODEL-driven editing. 'edit' asks the user to describe a change
in plain words; the model then revises the plan, keeping the steps the change
doesn't touch. (The old behaviour dead-ended to "edit the file and re-run", which
rebuilt the whole plan unless the goal was retyped byte-for-byte.)
"""
import types

import pytest

from achilles import harness as H
from achilles.llm import LLMError


def _cfg(tmp_path):
    return types.SimpleNamespace(
        workspace_path=tmp_path, act_protocol="native", tools=[], tools_dir="",
        comfy_url="", max_acts_per_step=6, temperature=0.2, max_tokens=0,
        auto_approve_plan=False)


def _mk(tmp_path):
    h = H.Harness(_cfg(tmp_path), log=lambda *_: None)
    h.state_dir.mkdir(parents=True, exist_ok=True)
    return h


class _ScriptedApprovalChannel:
    """Drives _approve_loop over the REAL channel boundary (no monkeypatching of
    _approve): each approval.request returns the next scripted reply."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def emit(self, type, data):
        pass

    def request(self, type, data):
        self.requests.append((type, dict(data)))
        return self.replies.pop(0)


def test_approve_loop_edit_flows_through_channel(tmp_path, monkeypatch):
    # The plan gate now runs over the channel: an 'edit' reply folds decision +
    # instruction into one round trip, which _approve stashes for _ask_edit_instruction.
    ch = _ScriptedApprovalChannel([
        {"decision": "edit", "instruction": "make step two about CSS"},
        {"decision": "approve"},
    ])
    h = H.Harness(_cfg(tmp_path), channel=ch)
    h.state_dir.mkdir(parents=True, exist_ok=True)

    def _revise(config, goal, steps, instruction, tree):
        assert instruction == "make step two about CSS"      # carried by the reply
        return ["step one", "step two about CSS"]
    monkeypatch.setattr(H, "revise_plan", _revise)

    plan = [{"done": False, "text": "step one"}, {"done": False, "text": "step two"}]
    result = h._approve_loop("build", plan, "tree")

    assert [s["text"] for s in result] == ["step one", "step two about CSS"]
    assert [t for t, _ in ch.requests] == ["approval.request", "approval.request"]
    assert ch.requests[0][1] == {"subject": "plan"}


def test_approve_loop_reject_through_channel_returns_none(tmp_path):
    ch = _ScriptedApprovalChannel([{"decision": "reject"}])
    h = H.Harness(_cfg(tmp_path), channel=ch)
    h.state_dir.mkdir(parents=True, exist_ok=True)
    assert h._approve_loop("g", [{"done": False, "text": "x"}], "t") is None


def test_edit_revises_via_model_and_keeps_untouched(tmp_path, monkeypatch):
    h = _mk(tmp_path)
    plan = [{"done": False, "text": "step one"},
            {"done": False, "text": "step two"}]

    _decisions = iter(["edit", "yes"])
    monkeypatch.setattr(h, "_approve", lambda: next(_decisions))
    monkeypatch.setattr(h, "_ask_edit_instruction", lambda: "make step two about CSS")

    def _revise(config, goal, steps, instruction, tree):
        assert steps == ["step one", "step two"]         # current plan handed over
        assert instruction == "make step two about CSS"
        return ["step one", "step two about CSS"]         # model keeps step one
    monkeypatch.setattr(H, "revise_plan", _revise)

    result = h._approve_loop("build", plan, "tree")
    assert [s["text"] for s in result] == ["step one", "step two about CSS"]
    # and it was persisted to plan.md
    assert [s["text"] for s in h._load_plan()] == ["step one", "step two about CSS"]


def test_no_returns_none(tmp_path, monkeypatch):
    h = _mk(tmp_path)
    monkeypatch.setattr(h, "_approve", lambda: "no")
    assert h._approve_loop("g", [{"done": False, "text": "x"}], "t") is None


def test_yes_returns_same_plan(tmp_path, monkeypatch):
    h = _mk(tmp_path)
    plan = [{"done": False, "text": "x"}]
    monkeypatch.setattr(h, "_approve", lambda: "yes")
    assert h._approve_loop("g", plan, "t") is plan


def test_empty_instruction_cancels_edit(tmp_path, monkeypatch):
    # Cancelling the edit (empty instruction) must NOT call the model or touch plan.
    h = _mk(tmp_path)
    plan = [{"done": False, "text": "x"}]
    _decisions = iter(["edit", "yes"])
    monkeypatch.setattr(h, "_approve", lambda: next(_decisions))
    monkeypatch.setattr(h, "_ask_edit_instruction", lambda: "")

    def _boom(*a, **k):
        raise AssertionError("revise_plan must not be called on an empty instruction")
    monkeypatch.setattr(H, "revise_plan", _boom)

    assert h._approve_loop("g", plan, "t") is plan       # unchanged, then approved


def test_revise_error_keeps_current_plan(tmp_path, monkeypatch):
    h = _mk(tmp_path)
    plan = [{"done": False, "text": "x"}]
    _decisions = iter(["edit", "yes"])
    monkeypatch.setattr(h, "_approve", lambda: next(_decisions))
    monkeypatch.setattr(h, "_ask_edit_instruction", lambda: "do a thing")

    def _boom(*a, **k):
        raise LLMError("model unreachable")
    monkeypatch.setattr(H, "revise_plan", _boom)

    assert h._approve_loop("g", plan, "t") is plan       # error → keep, re-ask, approve


def test_revise_empty_result_keeps_current_plan(tmp_path, monkeypatch):
    h = _mk(tmp_path)
    plan = [{"done": False, "text": "x"}]
    _decisions = iter(["edit", "yes"])
    monkeypatch.setattr(h, "_approve", lambda: next(_decisions))
    monkeypatch.setattr(h, "_ask_edit_instruction", lambda: "do a thing")
    monkeypatch.setattr(H, "revise_plan", lambda *a, **k: [])

    assert h._approve_loop("g", plan, "t") is plan
