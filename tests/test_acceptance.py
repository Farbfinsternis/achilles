"""Tests for acceptance.py's pure parsing layer.

The Definition-of-Done parsers repair the sloppy output a weak planner emits.
Two documented bugs (contains-prose, exists+contains fusion) lived here, so
these tests pin the repairs.
"""

import pytest

from achilles import acceptance
from achilles.acceptance import (
    parse_acceptance,
    expected_paths,
    _normalise,
    _parse_verdicts,
    _sanitize_run,
    _judge,
    _gather_context,
    Criterion,
    JudgeUnavailable,
)
from achilles.llm import LLMError


# ---- expected_paths (executor ↔ DoD filename coordination) ----------------

def test_expected_paths_collects_exists_and_contains():
    criteria = [
        Criterion("exists", "assets/hero.jpg"),
        Criterion("contains", "index.html :: <canvas"),
        Criterion("judge", "looks professional"),
    ]
    assert expected_paths(criteria) == ["assets/hero.jpg", "index.html"]


def test_expected_paths_excludes_absent():
    # An `absent:` file must NOT exist — it is not a target the executor creates.
    criteria = [Criterion("exists", "styles.css"), Criterion("absent", "TODO.txt")]
    assert expected_paths(criteria) == ["styles.css"]


def test_expected_paths_dedups_and_handles_fusion():
    # exists+contains fusion resolves to the same path; it must appear once.
    criteria = [
        Criterion("exists", "index.html :: <html"),
        Criterion("contains", "index.html :: <body"),
    ]
    assert expected_paths(criteria) == ["index.html"]


# ---- _normalise / parse_acceptance ---------------------------------------

def test_exists_with_double_colon_becomes_contains():
    # The exists+contains fusion a weak planner keeps emitting.
    c = _normalise("exists", "index.html :: <html>")
    assert c.kind == "contains"
    assert c.text == "index.html :: <html>"


def test_absent_strips_stray_tail():
    c = _normalise("absent", "secret.key :: whatever")
    assert c.kind == "absent"
    assert c.text == "secret.key"


def test_plain_exists_unchanged():
    c = _normalise("exists", "index.html")
    assert c.kind == "exists"
    assert c.text == "index.html"


def test_parse_tagged_lines():
    text = (
        "- [ ] exists: index.html\n"
        "- [ ] contains: styles.css :: display: grid\n"
        "- [ ] judge: the page looks professional\n"
    )
    crit = parse_acceptance(text)
    assert [c.kind for c in crit] == ["exists", "contains", "judge"]


def test_cmd_is_alias_for_run():
    crit = parse_acceptance("- [ ] cmd: pytest -q\n")
    assert len(crit) == 1
    assert crit[0].kind == "run"


def test_untagged_bullet_defaults_to_judge():
    crit = parse_acceptance("- the navigation must be keyboard accessible\n")
    assert len(crit) == 1
    assert crit[0].kind == "judge"


def test_prose_preamble_lines_are_dropped():
    text = (
        "Here is the Definition of Done:\n"
        "- [ ] exists: index.html\n"
        "Note: these are strict.\n"
    )
    crit = parse_acceptance(text)
    assert len(crit) == 1
    assert crit[0].kind == "exists"


# ---- _parse_verdicts ------------------------------------------------------

def test_verdicts_various_formats():
    reply = (
        "1: PASS - index.html has <canvas>\n"
        "2) FAIL: no stylesheet linked\n"
        "3. pass — looks fine\n"
    )
    out = _parse_verdicts(reply, 3)
    assert out[0][0] is True
    assert out[1][0] is False
    assert out[2][0] is True


def test_missing_verdict_defaults_to_fail():
    out = _parse_verdicts("1: PASS - ok\n", 3)
    assert out[0][0] is True
    assert out[1][0] is False
    assert out[2][0] is False
    assert "no verdict" in out[1][1]


# ---- _sanitize_run --------------------------------------------------------

def test_sanitize_strips_shell_tails():
    assert _sanitize_run("pytest -q ; exit $?") == "pytest -q"
    assert _sanitize_run("npm test && echo done") == "npm test"
    assert _sanitize_run("cargo test | tail") == "cargo test"


def test_sanitize_leaves_clean_command():
    assert _sanitize_run("python -m pytest -q") == "python -m pytest -q"


# ---- judge infrastructure error (Bug 11) ----------------------------------

class _Ctx:
    def __init__(self, ws):
        self.ws = ws


class _JudgeCfg:
    judge_model = ""
    judge_char_budget = 24000
    judge_char_per_file = 6000


def test_judge_unavailable_raises_not_fail(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")

    def boom(*a, **k):
        raise LLMError("connection refused")

    monkeypatch.setattr(acceptance, "chat", boom)
    with pytest.raises(JudgeUnavailable):
        _judge(_JudgeCfg(), [Criterion("judge", "looks good")],
               _Ctx(tmp_path), lambda *a, **k: None)


def test_judge_returns_verdicts_when_reachable(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<canvas></canvas>", encoding="utf-8")
    monkeypatch.setattr(acceptance, "chat",
                        lambda *a, **k: "1: PASS - index.html has <canvas>")
    out = _judge(_JudgeCfg(), [Criterion("judge", "has a canvas")],
                 _Ctx(tmp_path), lambda *a, **k: None)
    assert out == [(True, "index.html has <canvas>")]


# ---- context budget (Bug 7) ------------------------------------------------

def test_gather_context_summarises_omitted_files(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("x" * 100, encoding="utf-8")
    out = _gather_context(_Ctx(tmp_path), per_file=100, total=200)
    # One summary note, not the old unbounded per-file marker lines.
    assert "more text file(s) omitted" in out
    assert "(omitted — context budget reached)" not in out


def test_gather_context_includes_priority_file_whole(tmp_path):
    # A contract-referenced file (one big self-contained page) must be shown WHOLE,
    # while an incidental file still respects the small per-file cap. Otherwise the
    # judge FAILs criteria whose evidence lives past the per-file cut.
    (tmp_path / "index.html").write_text("A" * 10000, encoding="utf-8")
    (tmp_path / "other.py").write_text("B" * 10000, encoding="utf-8")
    out = _gather_context(_Ctx(tmp_path), per_file=2000, total=40000,
                          priority=["index.html"])
    assert "A" * 10000 in out                    # priority file untrimmed
    assert "B" * 2001 not in out                 # incidental file trimmed at per_file
    assert "chars trimmed" in out                # …and marked as trimmed


def test_judge_forwards_priority_to_context(monkeypatch, tmp_path):
    captured = {}

    def fake_gather(ctx, per_file, total, priority=()):
        captured["priority"] = list(priority)
        return "FILES"
    monkeypatch.setattr(acceptance, "_gather_context", fake_gather)
    monkeypatch.setattr(acceptance, "chat", lambda *a, **k: "1: PASS - ok")
    _judge(_JudgeCfg(), [Criterion("judge", "x")], _Ctx(tmp_path),
           lambda *a, **k: None, priority=["index.html"])
    assert captured["priority"] == ["index.html"]


def test_check_uses_expected_paths_as_judge_priority(monkeypatch, tmp_path):
    # check() must hand the contract's exists/contains paths to the judge as
    # priority, so the judged file is shown whole.
    from achilles.tools import Registry, BUILTINS, ToolContext as TC
    (tmp_path / "index.html").write_text("hello world", encoding="utf-8")
    reg = Registry()
    for t in BUILTINS:
        reg.register(t)
    seen = {}

    def fake_judge(config, items, ctx, log, priority=()):
        seen["priority"] = list(priority)
        return [(True, "ok") for _ in items]
    monkeypatch.setattr(acceptance, "_judge", fake_judge)
    criteria = [Criterion("contains", "index.html :: hello"),
                Criterion("judge", "looks good")]
    acceptance.check(_JudgeCfg(), criteria, reg, TC(tmp_path), lambda *a, **k: None)
    assert seen["priority"] == ["index.html"]
