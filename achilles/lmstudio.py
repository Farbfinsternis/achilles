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

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional


class LMStudioError(RuntimeError):
    pass


# Where we remember the last model in use, so a cold LM Studio (nothing loaded)
# can be restored to the model the user actually ran. Machine-global on purpose:
# "the model you had loaded" is a fact about LM Studio, not about one workspace.
_STATE_DIR = Path.home() / ".achilles"
_LAST_MODEL_FILE = _STATE_DIR / "last_model"

# The shipped config placeholder. LM Studio's API serves whatever is loaded, so
# users never have to set `model` to a real key — but that means it is NOT a
# loadable target either. Treated as "no real model" everywhere below.
_PLACEHOLDER_MODELS = {"", "local-model"}


def _real_key(model: str) -> Optional[str]:
    """The model-key if it looks like a real one to `lms load`, else None."""
    m = (model or "").strip()
    return m if m and m not in _PLACEHOLDER_MODELS else None


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


def loaded_llm(lms_command: str = "lms") -> Optional[str]:
    """Return the model-key of the currently loaded chat model, or None.

    We reload THIS exact key after an image render, not config.model — config.model
    is often a placeholder (users leave the OpenAI `model` field as "local-model";
    ensure_loaded then adopts the real loaded key so requests stay valid), and
    `lms load local-model` would fail with "Model not found". Query `lms ps` while
    the model is still resident, before unload, to learn its real key.

    Best-effort: any failure (CLI error, unexpected JSON, nothing loaded) returns
    None so the caller can fall back to config.model rather than crash the swap.
    Embedding models are skipped — we only care about the chat model we displace."""
    try:
        out = _run(lms_command, "ps", "--json")
        entries = json.loads(out) if out else []
    except (LMStudioError, json.JSONDecodeError):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("type") == "llm":
            key = entry.get("modelKey") or entry.get("identifier")
            if key:
                return key
    return None


def remember_model(model_key: str) -> None:
    """Record the model-key currently in use, so a later cold start (LM Studio
    launched with nothing loaded) can restore it. Placeholders are never saved —
    they can't be reloaded. Best-effort: a write failure must not break a run."""
    if not _real_key(model_key):
        return
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _LAST_MODEL_FILE.write_text(model_key.strip() + "\n", encoding="utf-8")
    except OSError:
        pass


def last_remembered() -> Optional[str]:
    """The last model-key remember_model() saved, or None."""
    try:
        key = _LAST_MODEL_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


def ensure_loaded(config, log: Callable[[str], None] = print) -> None:
    """Guarantee a chat model is resident before Achilles ever calls it.

    LM Studio can sit with NOTHING loaded — a fresh launch, or after our own image
    swap died mid-render. Every LLM call then 400s with "No models loaded" and the
    run dies before it starts. So at startup: if a model is loaded, just remember
    it; if not, load the last one we saw (the model the user actually used), and
    only fall back to config.model when that is a real key.

    Best-effort and silent on the happy path — it never raises, so a genuinely
    unresolvable state still reaches llm.py's clear error instead of a traceback."""
    if not available(config.lms_command):
        return  # no `lms` CLI: we can't manage models; llm.py reports the rest
    current = loaded_llm(config.lms_command)
    if current:
        remember_model(current)
        _adopt_model_id(config, current)
        return
    target = last_remembered() or _real_key(getattr(config, "model", ""))
    if not target:
        log("   ⚠ no model loaded in LM Studio and none remembered — load one "
            "(`lms load`) or set a real `model` in achilles.toml.")
        return
    log(f"   no model loaded — loading last used {target}…")
    try:
        _run(config.lms_command, "load", target, "-y")
        remember_model(target)
        _adopt_model_id(config, target)
    except LMStudioError as e:
        log(f"   ⚠ could not load {target}: {e}")


def _adopt_model_id(config, key: str) -> None:
    """Point config.model at the model that is ACTUALLY loaded, when config.model is
    just the placeholder. Loading a model (above) only fills VRAM; the chat request
    still sends config.model as the OpenAI `model` field, and newer LM Studio
    REJECTS an unknown id like "local-model" ("Invalid model identifier") instead of
    leniently serving whatever is loaded. Adopting the real key closes that gap. A
    real, user-set model id is left untouched."""
    if key and not _real_key(getattr(config, "model", "")):
        config.model = key


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
