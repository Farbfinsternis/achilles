"""
Repl._read_line: multi-line goal entry. A line ending in a backslash continues on
the next (the break becomes a real newline); a plain Enter still submits a single
line. These pin that behaviour without a real terminal.
"""
import builtins
import types

import pytest

from achilles import repl as R
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


def test_paste_is_coalesced_into_one_goal(monkeypatch):
    # A pasted multi-line block (no backslashes) must become ONE goal, not one goal
    # per line — the fragmentation bug where each stray line got its own run.
    _feed(monkeypatch, ["Build a landing page", "with a hero section", "and a footer"])
    r = Repl(None)
    pending = iter([True, True, False])          # two lines buffered, then done
    monkeypatch.setattr(r, "_input_pending", lambda: next(pending))
    assert r._read_line() == "Build a landing page\nwith a hero section\nand a footer"


def test_typed_line_not_coalesced_when_nothing_pending(monkeypatch):
    # Nothing buffered after the line (human typing) → a plain single-line goal.
    _feed(monkeypatch, ["first goal"])
    r = Repl(None)
    monkeypatch.setattr(r, "_input_pending", lambda: False)
    assert r._read_line() == "first goal"


def test_input_pending_false_for_non_tty(monkeypatch):
    # The guard that keeps coalescing from ever firing on piped/captured stdin.
    monkeypatch.setattr(R.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    assert Repl(None)._input_pending() is False


def test_startup_resolves_model_before_showing_config(monkeypatch, capsys):
    # The config table must show the real loaded model, not the placeholder: run()
    # calls ensure_loaded (which adopts config.model) BEFORE _show_config.
    cfg = types.SimpleNamespace(
        model="local-model", base_url="http://localhost:1234/v1",
        workspace_path="/ws", verify_command="", act_protocol="native",
        use_git=True, comfy_url="")

    def _fake_ensure(config, log=print):
        config.model = "google/gemma-4-12b"       # adopt the loaded key
    monkeypatch.setattr(R.lmstudio, "ensure_loaded", _fake_ensure)

    r = Repl(cfg)
    monkeypatch.setattr(r, "_read_line",
                        lambda: (_ for _ in ()).throw(EOFError()))   # exit at once
    r.run()

    out = capsys.readouterr().out
    assert "google/gemma-4-12b" in out
    assert "local-model" not in out
