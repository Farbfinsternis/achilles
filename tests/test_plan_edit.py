"""
Plan approval with in-place editing. 'edit' must let the user change part of the
plan and KEEP the rest — the old behaviour dead-ended to "edit the file and re-run",
and a re-run rebuilt the whole plan unless the goal was retyped byte-for-byte.
"""
import types

from achilles import harness as H


def _cfg(tmp_path):
    return types.SimpleNamespace(
        workspace_path=tmp_path, native_tools=True, tools=[], tools_dir="",
        comfy_url="", max_acts_per_step=6, temperature=0.2, max_tokens=0,
        auto_approve_plan=False)


def _mk(tmp_path):
    h = H.Harness(_cfg(tmp_path), log=lambda *_: None)
    h.state_dir.mkdir(parents=True, exist_ok=True)
    return h


def test_edit_keeps_untouched_steps(tmp_path, monkeypatch):
    h = _mk(tmp_path)
    plan = [{"done": False, "text": "step one"},
            {"done": False, "text": "step two"}]
    h._save_plan("build it", plan)

    decisions = iter(["edit", "yes"])
    monkeypatch.setattr(h, "_approve", lambda: next(decisions))

    # The edit revises step two on disk but leaves step one alone.
    def _edit():
        h._save_plan("build it", [{"done": False, "text": "step one"},
                                  {"done": False, "text": "step two REVISED"}])
    monkeypatch.setattr(h, "_prompt_plan_edit", _edit)

    result = h._approve_loop("build it", plan)
    assert [s["text"] for s in result] == ["step one", "step two REVISED"]


def test_no_returns_none(tmp_path, monkeypatch):
    h = _mk(tmp_path)
    monkeypatch.setattr(h, "_approve", lambda: "no")
    assert h._approve_loop("g", [{"done": False, "text": "x"}]) is None


def test_yes_returns_same_plan(tmp_path, monkeypatch):
    h = _mk(tmp_path)
    plan = [{"done": False, "text": "x"}]
    monkeypatch.setattr(h, "_approve", lambda: "yes")
    assert h._approve_loop("g", plan) is plan


def test_edit_to_empty_plan_aborts(tmp_path, monkeypatch):
    h = _mk(tmp_path)
    plan = [{"done": False, "text": "x"}]
    h._save_plan("g", plan)
    monkeypatch.setattr(h, "_approve", lambda: "edit")

    def _wipe():   # user deletes every step line
        h.plan_path.write_text("# Achilles plan\n\n> Goal: g\n\n", encoding="utf-8")
    monkeypatch.setattr(h, "_prompt_plan_edit", _wipe)

    assert h._approve_loop("g", plan) is None


def test_repeated_edits_accumulate(tmp_path, monkeypatch):
    # edit → edit → yes: each edit reloads from disk, so successive tweaks stick.
    h = _mk(tmp_path)
    plan = [{"done": False, "text": "a"}]
    h._save_plan("g", plan)
    decisions = iter(["edit", "edit", "yes"])
    monkeypatch.setattr(h, "_approve", lambda: next(decisions))

    edits = iter([
        [{"done": False, "text": "a"}, {"done": False, "text": "b"}],
        [{"done": False, "text": "a"}, {"done": False, "text": "b"},
         {"done": False, "text": "c"}],
    ])
    monkeypatch.setattr(h, "_prompt_plan_edit", lambda: h._save_plan("g", next(edits)))

    result = h._approve_loop("g", plan)
    assert [s["text"] for s in result] == ["a", "b", "c"]
