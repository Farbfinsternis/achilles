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


# ---- acceptance non-convergence guard -------------------------------------

def _accept_harness(tmp_path, monkeypatch, rounds=5):
    cfg = Config(workspace=str(tmp_path), use_git=False, verify_command="",
                 max_accept_rounds=rounds)
    h = Harness(cfg, log=lambda *a, **k: None)
    h.state_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(h, "_commit", lambda *a, **k: None)
    return h


def test_acceptance_halts_on_non_convergence(tmp_path, monkeypatch):
    # The same criterion failing two rounds in a row (a fix that moved nothing) must
    # HALT early with a pointer, not churn silently to the round cap.
    from achilles import acceptance as A
    from achilles.acceptance import Criterion, Failure
    h = _accept_harness(tmp_path, monkeypatch, rounds=5)
    crit = Criterion("contains", "index.html :: PLAN -> ACT -> VERIFY")
    checks = {"n": 0}
    works = {"n": 0}
    monkeypatch.setattr(A, "check",
                        lambda *a, **k: (checks.__setitem__("n", checks["n"] + 1),
                                         [Failure(crit, "not found")])[1])
    monkeypatch.setattr(h, "_work",
                        lambda *a, **k: works.__setitem__("n", works["n"] + 1) or True)

    ok = h._acceptance_phase("goal", [{"done": True, "text": "t"}], [crit], None)

    assert ok is False
    assert checks["n"] == 2          # round 1 (check+fix), round 2 check → same → halt
    assert works["n"] == 1           # only one fix, not five rounds of churn


def test_acceptance_continues_when_making_progress(tmp_path, monkeypatch):
    # Shrinking failures (real progress) must NOT trip the guard — it runs until met.
    from achilles import acceptance as A
    from achilles.acceptance import Criterion, Failure
    h = _accept_harness(tmp_path, monkeypatch, rounds=5)
    monkeypatch.setattr(h, "_work", lambda *a, **k: True)
    a = Failure(Criterion("contains", "x :: A"), "no")
    b = Failure(Criterion("contains", "x :: B"), "no")
    seq = iter([[a, b], [b], []])                 # two, then one, then met
    monkeypatch.setattr(A, "check", lambda *a, **k: next(seq))

    ok = h._acceptance_phase("goal", [{"done": True, "text": "t"}], [], None)

    assert ok is True                             # progressed each round, then met
