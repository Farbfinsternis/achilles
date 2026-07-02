"""Tests for the harness's work-loss guards (Bug 1, Bug 5).

These drive _work_through_plan directly with a stubbed model so no server is
needed. The invariant under test: a model outage must never mark a step done.
"""

from achilles.config import Config
from achilles.harness import Harness


def _harness(tmp_path, monkeypatch):
    cfg = Config(workspace=str(tmp_path), use_git=False, verify_command="")
    h = Harness(cfg, log=lambda *a, **k: None)
    h.state_dir.mkdir(exist_ok=True)   # run() normally does this

    # Never touch git or run a real oracle in these unit tests.
    monkeypatch.setattr(h, "_commit", lambda *a, **k: None)
    monkeypatch.setattr(h, "_verify", lambda: (True, None))
    return h


def test_model_error_leaves_step_unfinished(tmp_path, monkeypatch):
    h = _harness(tmp_path, monkeypatch)
    monkeypatch.setattr(h, "_work", lambda prompt: False)   # model unreachable
    plan = [{"done": False, "text": "step one"},
            {"done": False, "text": "step two"}]

    ok, _ = h._work_through_plan("goal", plan)

    assert ok is False
    assert plan[0]["done"] is False           # NOT burned — resumable
    assert plan[1]["done"] is False


def test_all_steps_done_on_success(tmp_path, monkeypatch):
    h = _harness(tmp_path, monkeypatch)
    monkeypatch.setattr(h, "_work", lambda prompt: True)
    plan = [{"done": False, "text": "step one"},
            {"done": False, "text": "step two"}]

    ok, _ = h._work_through_plan("goal", plan)

    assert ok is True
    assert all(s["done"] for s in plan)


def test_second_step_outage_keeps_first_done(tmp_path, monkeypatch):
    h = _harness(tmp_path, monkeypatch)
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        return calls["n"] == 1                 # first step ok, second errors

    monkeypatch.setattr(h, "_work", flaky)
    plan = [{"done": False, "text": "step one"},
            {"done": False, "text": "step two"}]

    ok, _ = h._work_through_plan("goal", plan)

    assert ok is False
    assert plan[0]["done"] is True             # real progress is preserved
    assert plan[1]["done"] is False            # the outage step stays open
