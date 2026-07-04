"""
Repl._read_line: multi-line goal entry. A line ending in a backslash continues on
the next (the break becomes a real newline); a plain Enter still submits a single
line. These pin that behaviour without a real terminal.
"""
import builtins

import pytest

from achilles.repl import Repl


def _feed(monkeypatch, lines):
    it = iter(lines)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(it))


def test_single_line_unchanged(monkeypatch):
    _feed(monkeypatch, ["build a landing page"])
    assert Repl(None)._read_line() == "build a landing page"


def test_backslash_continuation_joins_with_newlines(monkeypatch):
    _feed(monkeypatch, ["first line\\", "second line\\", "third line"])
    assert Repl(None)._read_line() == "first line\nsecond line\nthird line"


def test_trailing_backslash_is_dropped(monkeypatch):
    _feed(monkeypatch, ["a\\", "b"])
    assert Repl(None)._read_line() == "a\nb"


def test_eof_on_continuation_submits_buffer(monkeypatch):
    seq = ["half a goal\\"]

    def _inp(*a, **k):
        if not seq:
            raise EOFError
        return seq.pop(0)
    monkeypatch.setattr(builtins, "input", _inp)
    assert Repl(None)._read_line() == "half a goal"


def test_eof_on_first_line_raises(monkeypatch):
    def _inp(*a, **k):
        raise EOFError
    monkeypatch.setattr(builtins, "input", _inp)
    with pytest.raises(EOFError):
        Repl(None)._read_line()
