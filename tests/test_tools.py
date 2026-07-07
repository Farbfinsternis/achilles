"""Tests for the shell normalisation and the portable file tools.

A POSIX-trained model emits `/workspace/...` paths, `cp`/`ls`, and `mkdir -p` under
cmd.exe. These pin the three absorbing layers: /workspace rewriting, the cmd-vs-bash
resolver, and the copy_file / file_exists hands that replace shelled-out file ops.
The `windows`/`bash` params keep the resolver tests platform-neutral.
"""

import achilles.tools as T
from achilles.tools import (
    _strip_workspace_prefix, _fix_mkdir_parents, _resolve_command,
    _copy_file, ToolContext, BUILTINS,
)


# ---- /workspace hallucination (Layer 3) -----------------------------------

def test_strip_workspace_prefix():
    assert _strip_workspace_prefix("cp /workspace/assets/a /workspace/assets/b") \
        == "cp assets/a assets/b"
    assert _strip_workspace_prefix("ls /workspace") == "ls ."


def test_strip_workspace_leaves_lookalikes_alone():
    # /home/workspace is a real nested path, not the hallucinated mount.
    assert _strip_workspace_prefix("cat /home/workspace/x") == "cat /home/workspace/x"


# ---- mkdir -p on the cmd fallback -----------------------------------------

def test_fix_mkdir_parents():
    assert _fix_mkdir_parents("mkdir -p src/app") == "mkdir src/app"
    assert _fix_mkdir_parents("mkdir --parents build") == "mkdir build"


def test_fix_mkdir_leaves_others_and_chains_alone():
    assert _fix_mkdir_parents("python -p x") == "python -p x"
    assert _fix_mkdir_parents("mkdir -p a && echo -p") == "mkdir -p a && echo -p"


# ---- the resolver: cmd vs bash --------------------------------------------

def test_resolver_routes_through_bash_on_windows():
    args, use_shell = _resolve_command(
        "cp /workspace/assets/a /workspace/assets/b", windows=True, bash="C:/git/bash.exe")
    assert use_shell is False
    assert args == ["C:/git/bash.exe", "-c", "cp assets/a assets/b"]   # argv, /workspace stripped


def test_resolver_cmd_fallback_fixes_mkdir():
    args, use_shell = _resolve_command("mkdir -p /workspace/src/app", windows=True, bash=None)
    assert use_shell is True
    assert args == "mkdir src/app"                    # /workspace stripped + -p dropped for cmd


def test_resolver_posix_keeps_mkdir_p():
    # On a POSIX host -p is load-bearing; only /workspace is rewritten.
    args, use_shell = _resolve_command("mkdir -p /workspace/src/app", windows=False)
    assert (args, use_shell) == ("mkdir -p src/app", True)


# ---- shell(model=…) wiring ------------------------------------------------

class _Result:
    returncode = 0
    stdout = ""
    stderr = ""


def _capture(seen):
    def run(args, **kw):
        seen["args"] = args
        return _Result()
    return run


def test_shell_model_true_resolves(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(T.subprocess, "run", _capture(seen))
    monkeypatch.setattr(T.os, "name", "nt")
    monkeypatch.setattr(T, "_find_git_bash", lambda: None)     # force the cmd path

    ToolContext(tmp_path).shell("mkdir -p /workspace/build/assets", model=True)
    assert seen["args"] == "mkdir build/assets"


def test_shell_model_false_is_verbatim(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(T.subprocess, "run", _capture(seen))
    # A user-authored verify_command runs exactly as written.
    ToolContext(tmp_path).shell("mkdir -p build", model=False)
    assert seen["args"] == "mkdir -p build"


# ---- copy_file (Layer 1) --------------------------------------------------

def test_copy_file_copies_and_makes_parents(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "a.jpg").write_bytes(b"IMG")
    ctx = ToolContext(tmp_path)

    out = _copy_file({"src": "assets/a.jpg", "dst": "public/img/b.jpg"}, None, ctx)

    assert out.startswith("OK")
    assert (tmp_path / "public" / "img" / "b.jpg").read_bytes() == b"IMG"


def test_copy_file_missing_source(tmp_path):
    out = _copy_file({"src": "nope.jpg", "dst": "b.jpg"}, None, ToolContext(tmp_path))
    assert out.startswith("ERROR") and "no such file" in out


def test_copy_file_rejects_escape(tmp_path):
    out = _copy_file({"src": "a.jpg", "dst": "../escape.jpg"}, None, ToolContext(tmp_path))
    assert out.startswith("ERROR")


# ---- file_exists is now a hand --------------------------------------------

def test_file_exists_and_copy_file_are_hands():
    by_name = {t.name: t for t in BUILTINS}
    assert by_name["file_exists"].act is True         # exposed to the model now
    assert by_name["copy_file"].act is True
    # its check-only siblings stay hidden
    assert by_name["file_absent"].act is False
