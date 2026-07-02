"""Tests for acceptance.py's pure parsing layer.

The Definition-of-Done parsers repair the sloppy output a weak planner emits.
Two documented bugs (contains-prose, exists+contains fusion) lived here, so
these tests pin the repairs.
"""

from achilles.acceptance import (
    parse_acceptance,
    _normalise,
    _parse_verdicts,
    _sanitize_run,
)


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
