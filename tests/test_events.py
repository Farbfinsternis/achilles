"""Tests for the semantic event stream (docs/protocol.md §3).

A RecordingChannel captures every emit; the harness phases are driven with stubs so
no model/oracle is needed. The invariant: the meaningful state changes surface as
typed events (plan.ready, dod.ready, step.*, verify.result, accept.*), not only as
prose log lines.
"""

from achilles import acceptance as A
from achilles.acceptance import Criterion, Failure
from achilles.config import Config
from achilles.harness import Harness
import achilles.harness as H


class RecordingChannel:
    """Captures emitted events; answers every gate with approve."""
    def __init__(self):
        self.events = []

    def emit(self, type, data):
        self.events.append((type, data))

    def request(self, type, data):
        return {"decision": "approve"}

    def of(self, type):
        return [d for t, d in self.events if t == type]


def _h(tmp_path, ch, **cfg_kw):
    kw = {"use_git": False, "verify_command": ""}
    kw.update(cfg_kw)
    h = Harness(Config(workspace=str(tmp_path), **kw), channel=ch)
    h.state_dir.mkdir(exist_ok=True)          # run() normally does this
    return h


# ---- step.* ---------------------------------------------------------------

def test_step_events_on_success(tmp_path, monkeypatch):
    ch = RecordingChannel()
    h = _h(tmp_path, ch)
    monkeypatch.setattr(h, "_work", lambda p: True)
    monkeypatch.setattr(h, "_verify", lambda: (True, None))
    monkeypatch.setattr(h, "_commit", lambda *a, **k: None)

    h._work_through_plan("g", [{"done": False, "text": "a"}, {"done": False, "text": "b"}])

    assert ch.of("step.started") == [
        {"index": 1, "total": 2, "text": "a"},
        {"index": 2, "total": 2, "text": "b"},
    ]
    assert [d["status"] for d in ch.of("step.finished")] == ["done", "done"]


def test_step_finished_unfinished_on_outage(tmp_path, monkeypatch):
    ch = RecordingChannel()
    h = _h(tmp_path, ch)
    monkeypatch.setattr(h, "_work", lambda p: False)          # model unreachable

    h._work_through_plan("g", [{"done": False, "text": "a"}])

    assert ch.of("step.finished") == [{"index": 1, "status": "unfinished"}]


# ---- verify.result --------------------------------------------------------

def test_verify_result_event(tmp_path, monkeypatch):
    ch = RecordingChannel()
    h = _h(tmp_path, ch, verify_command="run-the-oracle")
    monkeypatch.setattr(h.ctx, "shell", lambda cmd: "exit=0\nall good")

    h._verify()

    ev = ch.of("verify.result")
    assert ev == [{"command": "run-the-oracle", "passed": True, "output": "exit=0\nall good"}]


# ---- accept.* -------------------------------------------------------------

def test_accept_round_and_failures_events(tmp_path, monkeypatch):
    ch = RecordingChannel()
    h = _h(tmp_path, ch, max_accept_rounds=3)
    monkeypatch.setattr(h, "_commit", lambda *a, **k: None)
    monkeypatch.setattr(h, "_work", lambda p: True)
    fail = Failure(Criterion("contains_any", "Bäckerei Sonnenschein"), "missing")
    seq = iter([[fail], []])                                   # one unmet round, then met
    monkeypatch.setattr(A, "check", lambda *a, **k: next(seq))

    ok = h._acceptance_phase("g", [{"done": True, "text": "t"}],
                             [Criterion("contains_any", "Bäckerei Sonnenschein")], None)

    assert ok is True
    assert ch.of("accept.round")[0] == {"round": 1, "max": 3}
    assert ch.of("accept.failures")[0]["failures"] == [
        {"kind": "contains_any", "text": "Bäckerei Sonnenschein", "reason": "missing"}]


# ---- plan.ready / dod.ready (autopilot run) -------------------------------

def test_run_emits_plan_and_dod_ready(tmp_path, monkeypatch):
    ch = RecordingChannel()
    monkeypatch.setattr(H, "make_plan", lambda cfg, goal, tree: ["step 1", "step 2"])
    monkeypatch.setattr(A, "make_acceptance", lambda *a, **k: [Criterion("judge", "looks good")])
    h = _h(tmp_path, ch, auto_approve_plan=True)
    monkeypatch.setattr(h, "_execute", lambda goal, plan: True)

    assert h.run("build a thing") is True

    plan_ready = ch.of("plan.ready")[0]
    assert plan_ready["resumed"] is False
    assert [s["text"] for s in plan_ready["steps"]] == ["step 1", "step 2"]
    assert ch.of("dod.ready")[0]["criteria"] == [{"kind": "judge", "text": "looks good"}]


def test_run_plan_ready_marks_resume(tmp_path, monkeypatch):
    # A saved, unfinished plan for the same goal → plan.ready carries resumed:true and
    # no re-planning happens.
    ch = RecordingChannel()
    h = _h(tmp_path, ch)
    h.state_dir.mkdir(exist_ok=True)
    h._save_plan("build a thing", [{"done": False, "text": "step 1"}])
    monkeypatch.setattr(H, "make_plan",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not re-plan")))
    monkeypatch.setattr(h, "_execute", lambda goal, plan: True)

    assert h.run("build a thing") is True
    assert ch.of("plan.ready")[0]["resumed"] is True
