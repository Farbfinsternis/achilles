"""
style.py — a tiny, dependency-free ANSI skin for Achilles' output.

The harness talks to you through one thin `log` channel; this module is the only
place that knows about colour. Everything is a plain string in, styled string
out, so callers stay readable (`ok("green")`, `head("PLAN")`) and NOTHING here
changes behaviour — turn colour off and you get the same text you always had.

Colour is enabled only when it makes sense: a real TTY, no NO_COLOR in the
environment, and (on Windows) with virtual-terminal processing switched on. Set
ACHILLES_COLOR=0/1 to force it either way; honour NO_COLOR (https://no-color.org).
"""

import os
import sys


# ---- ANSI SGR codes -------------------------------------------------------

_CODES = {
    "reset": "0",
    "bold": "1", "dim": "2", "italic": "3", "underline": "4",
    "red": "31", "green": "32", "yellow": "33", "blue": "34",
    "magenta": "35", "cyan": "36", "grey": "90",
    "bright_red": "91", "bright_green": "92", "bright_yellow": "93",
    "bright_cyan": "96", "white": "97",
}


def _detect() -> bool:
    force = os.environ.get("ACHILLES_COLOR")
    if force is not None:
        return force.strip().lower() in ("1", "true", "yes", "always", "on")
    if os.environ.get("NO_COLOR") is not None:
        return False
    if not sys.stdout.isatty():
        return False
    return _enable_windows_vt()


def _enable_windows_vt() -> bool:
    """Non-Windows terminals already speak ANSI. On Windows 10+, flip on
    ENABLE_VIRTUAL_TERMINAL_PROCESSING so the escape codes render instead of
    printing raw. Any failure just means 'no colour', never a crash."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        # 0x0004 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


ENABLED = _detect()


def paint(text: str, *names: str) -> str:
    """Wrap `text` in the named SGR codes (bold/dim/colour). A no-op when colour
    is disabled, so call sites never branch on it themselves."""
    if not ENABLED or not names:
        return text
    codes = ";".join(_CODES[n] for n in names if n in _CODES)
    if not codes:
        return text
    return f"\033[{codes}m{text}\033[0m"


# ---- semantic helpers (call these, not paint(), from the harness) ---------

def head(label: str, width: int = 46, color: str = "cyan") -> str:
    """A section rule like ──────  PLAN  ──────, centred and coloured. Used for
    the phase banners (PLAN / STEP / FLOOR FIX / ACCEPT)."""
    label = f" {label} "
    pad = max(width - len(label), 4)
    left = pad // 2
    right = pad - left
    rule = paint("─" * left, "grey") + paint(label, "bold", color) + paint("─" * right, "grey")
    return rule


def ok(text: str) -> str:
    return paint(text, "bright_green")


def bad(text: str) -> str:
    return paint(text, "bright_red")


def warn(text: str) -> str:
    return paint(text, "bright_yellow")


def accent(text: str) -> str:
    return paint(text, "cyan")


def muted(text: str) -> str:
    return paint(text, "grey")


def bold(text: str) -> str:
    return paint(text, "bold")
