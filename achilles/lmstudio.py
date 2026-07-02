"""
lmstudio.py — swap the language model out of VRAM and back.

On an 8 GB card a chat model and an image model cannot both be resident, so
before ComfyUI renders we must UNLOAD the LM Studio model, and after we must
LOAD it again. We drive LM Studio's `lms` CLI (shipped with LM Studio) rather
than its REST API, because unloading is exactly the kind of lifecycle action the
CLI is built for.

The one rule that matters: **the reload must always run.** If ComfyUI crashes
mid-render, Achilles must still get its brain back, or the very next LLM call
(llm.py) fails with a connection error. So the caller wraps generation in a
try/finally and the finally calls `load()`. This module only provides the two
verbs and an availability check to fail *before* we ever unload.
"""

import shutil
import subprocess
from typing import Callable


class LMStudioError(RuntimeError):
    pass


def available(lms_command: str = "lms") -> bool:
    """Is the `lms` CLI on PATH? Check this BEFORE unloading, so we never strand
    Achilles brainless because the tool to reload it was missing all along."""
    return shutil.which(lms_command) is not None


def _run(lms_command: str, *args: str, timeout: int = 300) -> str:
    try:
        # Decode with utf-8 + replace, NOT the platform default: on Windows the
        # default is cp1252, and `lms`'s progress spinner emits bytes cp1252
        # cannot decode — which crashes subprocess's pipe-reader thread mid-load.
        proc = subprocess.run([lms_command, *args], capture_output=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except FileNotFoundError as e:
        raise LMStudioError(f"`{lms_command}` not found on PATH. Is LM Studio's "
                            "CLI installed? (LM Studio ships `lms`.)") from e
    except subprocess.TimeoutExpired as e:
        raise LMStudioError(f"`{lms_command} {' '.join(args)}` timed out "
                            f"after {timeout}s.") from e
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:500]
        raise LMStudioError(f"`{lms_command} {' '.join(args)}` failed: {detail}")
    return (proc.stdout or "").strip()


def unload_all(lms_command: str = "lms", log: Callable[[str], None] = print) -> None:
    """Free the VRAM the chat model holds. Idempotent — unloading with nothing
    loaded is not an error."""
    log("   unloading LM Studio model…")
    _run(lms_command, "unload", "--all")


def load(model: str, lms_command: str = "lms",
         log: Callable[[str], None] = print) -> None:
    """Reload the chat model by its LM Studio identifier. `-y` so it never blocks
    on an interactive prompt (this may run inside a finally, unattended)."""
    log(f"   reloading LM Studio model {model}…")
    _run(lms_command, "load", model, "-y")
