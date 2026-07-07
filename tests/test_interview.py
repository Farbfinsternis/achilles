"""Tests for the interview/spec wiring in the harness (Planungsmodus).

A FakeChannel scripts the interview answers and the spec-gate decision, and the
model calls (normalize / make_plan / make_acceptance) are stubbed, so no server is
needed. The invariants under test: the interview routes skip/answer, the DoD seed
anchors win over the model's, resume skips the interview, and a full interview run
plans from the ENGLISH goal while pinning the executor to the ORIGINAL one.
"""

import pytest

from achilles import spec as spec_mod
from achilles import acceptance
from achilles.acceptance import Criterion
from achilles.config import Config
from achilles.harness import Harness
import achilles.harness as H


class FakeChannel:
    """Scripts interview answers (None = skip) and the approval decision(s)."""

    def __init__(self, answers, approval="approve"):
        self.answers = list(answers)
        self.approval = approval
        self.emitted = []

    def emit(self, type, data):
        self.emitted.append((type, data))

    def request(self, type, data):
        if type == "interview.question":
            v = self.answers.pop(0)
            return {"skip": True} if v is None else {"value": v}
        if type == "approval.request":
            dec = self.approval.pop(0) if isinstance(self.approval, list) else self.approval
            return dec if isinstance(dec, dict) else {"decision": dec}
        return {}


class _NoChannel:
    """A channel that must never be asked (resume path)."""

    def emit(self, *a, **k):
        pass

    def request(self, *a, **k):
        raise AssertionError("channel must not be used on resume")


def _cfg(tmp_path, **kw):
    return Config(workspace=str(tmp_path), use_git=False, verify_command="", **kw)


def _sample_spec():
    return spec_mod.Spec(
        source_language="de",
        original_goal="Baue eine Seite für die Bäckerei Sonnenschein.",
        purpose="Advertise a neighbourhood bakery on one page.",
        audience="Local walk-in customers.",
        features=["Hero with name and tagline"],
        scope="Static HTML/CSS/JS.",
        ui_ux="Warm, rustic, mobile-first.",
        verbatim=["Bäckerei Sonnenschein"],
    )


# ---- _interview -----------------------------------------------------------

def test_interview_routes_skip_and_answer(tmp_path):
    ch = FakeChannel(["a bakery page", None, "hero, hours", None, None])
    h = Harness(_cfg(tmp_path), log=lambda *a, **k: None, channel=ch)
    answers = h._interview(spec_mod.SLOTS)
    assert answers["purpose"] == "a bakery page"
    assert answers["audience"] == ""            # skip → empty (normalize fills it)
    assert answers["features"] == "hero, hours"
    assert answers["scope"] == ""
    assert answers["ui_ux"] == ""


# ---- _merge_dod -----------------------------------------------------------

def test_merge_dod_seed_first_collision_and_dedup():
    seed = [Criterion("contains_any", "Bäckerei Sonnenschein")]
    model = [
        Criterion("exists", "index.html"),
        Criterion("contains", "index.html :: Bäckerei Sonnenschein"),  # collision → dropped
        Criterion("contains_any", "Bäckerei Sonnenschein"),            # dup → dropped
        Criterion("judge", "looks professional"),
    ]
    merged = Harness._merge_dod(seed, model)
    pairs = [(c.kind, c.text) for c in merged]
    assert pairs[0] == ("contains_any", "Bäckerei Sonnenschein")       # seed anchor first
    assert ("contains", "index.html :: Bäckerei Sonnenschein") not in pairs
    assert pairs.count(("contains_any", "Bäckerei Sonnenschein")) == 1
    assert ("exists", "index.html") in pairs
    assert ("judge", "looks professional") in pairs


# ---- resume ---------------------------------------------------------------

def test_prepare_spec_resumes_without_interview(tmp_path):
    h = Harness(_cfg(tmp_path), log=lambda *a, **k: None, mode="interview",
                channel=_NoChannel())
    h.state_dir.mkdir(exist_ok=True)
    h._save_spec(_sample_spec())
    got = h._prepare_spec("Baue eine Seite für die Bäckerei Sonnenschein.")
    assert got is not None
    assert got.verbatim == ["Bäckerei Sonnenschein"]        # loaded, gate skipped


# ---- full interview run ---------------------------------------------------

def test_run_interview_mode_end_to_end(tmp_path, monkeypatch):
    spec = _sample_spec()
    monkeypatch.setattr(spec_mod, "normalize", lambda *a, **k: spec)
    monkeypatch.setattr(H, "make_plan", lambda cfg, goal, tree: ["build index.html"])
    monkeypatch.setattr(acceptance, "make_acceptance", lambda *a, **k: [])

    ch = FakeChannel([None, None, None, None, None], approval="approve")
    cfg = _cfg(tmp_path, auto_approve_plan=True)
    h = Harness(cfg, log=lambda *a, **k: None, mode="interview", channel=ch)

    captured = {}

    def fake_execute(goal, plan):
        captured["goal"] = goal
        return True

    monkeypatch.setattr(h, "_execute", fake_execute)
    ok = h.run(spec.original_goal)

    assert ok is True
    assert h.spec_path.is_file()                                  # spec persisted
    # the plan is keyed to the ENGLISH goal (en_goal), not the raw German prompt
    assert "Advertise a neighbourhood bakery" in h.plan_path.read_text(encoding="utf-8")
    # the executor is pinned to the ORIGINAL-language goal (content truth)
    assert captured["goal"] == spec.original_goal
    # the Definition of Done carries the deterministic verbatim anchor
    assert "contains_any: Bäckerei Sonnenschein" in h.dod_path.read_text(encoding="utf-8")


def test_run_interview_spec_reject_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(spec_mod, "normalize", lambda *a, **k: _sample_spec())
    ch = FakeChannel([None, None, None, None, None], approval="reject")
    h = Harness(_cfg(tmp_path), log=lambda *a, **k: None, mode="interview", channel=ch)
    # _execute must never be reached when the spec gate is declined.
    monkeypatch.setattr(h, "_execute", lambda *a, **k: pytest.fail("should not execute"))
    assert h.run(_sample_spec().original_goal) is False
