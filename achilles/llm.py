"""
llm.py — the engine connector. Dependency-free.

Talks to any OpenAI-compatible /chat/completions endpoint: llama.cpp's server,
Ollama's OpenAI-compat port, vLLM, LM Studio, or a cloud API. We use the stdlib
(urllib) on purpose — a minimal harness should have nothing to `pip install`.

Non-streaming: a plan/act turn is short and we want the whole text before we
parse it, so streaming would only add complexity here.

Slow-model waiting: a big model on modest hardware can think for many minutes.
We run the blocking call in a worker thread and give the socket NO hard read
deadline, so we never kill a healthy in-flight generation. When the wait crosses
request_timeout we ASK the user whether to keep waiting or abort — and "keep
waiting" continues the SAME generation, because the connection was never cut.
"""

import json
import sys
import threading
import urllib.error
import urllib.request
from typing import List, Dict


class LLMError(RuntimeError):
    pass


def _ask_keep_waiting(url: str, waited: int) -> bool:
    """The wait has crossed request_timeout. Ask whether to keep waiting (True) or
    abort (False), instead of the old behaviour of killing a still-healthy
    generation. Non-interactive (no TTY): return False so batch runs fail fast
    rather than block forever on input()."""
    stdin = getattr(sys, "stdin", None)
    if stdin is None or not stdin.isatty():
        return False
    try:
        ans = input(f"\n… the model has been working for {waited}s with no reply yet. "
                    "Keep waiting? [Y/n] ").strip().lower()
    except EOFError:
        return False
    return ans in ("", "y", "yes", "j", "ja")


def _request_worker(req, result: Dict) -> None:
    """Run the blocking, non-streaming HTTP call in a thread so the caller can keep
    managing the clock. The socket gets NO read deadline (timeout=None), so only
    the caller's prompt — never a premature timeout — ends a live generation.
    Stores the raw body text under 'body', or the exception under 'error'."""
    try:
        with urllib.request.urlopen(req, timeout=None) as resp:
            result["body"] = resp.read().decode("utf-8")
    except BaseException as e:   # re-raised in the caller thread via _llm_error_for
        result["error"] = e


def _llm_error_for(err: BaseException, url: str, config) -> "LLMError":
    """Map a worker exception to a clean LLMError with an actionable message."""
    if isinstance(err, urllib.error.HTTPError):
        detail = err.read().decode("utf-8", "replace")[:500]
        return LLMError(f"HTTP {err.code} from {url}: {detail}")
    reason = getattr(err, "reason", None)
    if isinstance(err, TimeoutError) or isinstance(reason, TimeoutError):
        return LLMError(
            f"No response from {url} within request_timeout={config.request_timeout}s. "
            "The model may still be generating a long reply — raise request_timeout "
            "in achilles.toml, or cap output with max_tokens.")
    if isinstance(err, urllib.error.URLError):
        return LLMError(
            f"Could not reach {url}: {err.reason}. Is your model server running?")
    return LLMError(f"Request to {url} failed: {err!r}")


def chat(config, messages: List[Dict], temperature: float = 0.2,
         max_tokens: int | None = None, model: str | None = None) -> str:
    """Send messages, return the assistant's text content.

    `model` overrides config.model for this one call — used by the acceptance
    judge, which MAY point at a stronger local model. Left None (the default) it
    reuses config.model, so judging needs no second model loaded (8GB-VRAM safe).

    `max_tokens` caps generated tokens. Left None it falls back to
    config.max_tokens; when the effective value is 0 (or negative) we omit the
    field entirely, so the engine fills the model's own remaining context window
    instead of Achilles guessing a ceiling that truncates whole-file writes."""
    if max_tokens is None:
        max_tokens = getattr(config, "max_tokens", 0) or 0
    url = config.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model or config.model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    # Only send a cap when one is asked for. Omitting max_tokens lets LM Studio /
    # llama.cpp generate up to the loaded model's context limit.
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    # Run the blocking call in a worker and manage the clock here, so a slow-but-
    # healthy generation is never killed. Every request_timeout seconds of silence
    # we ask the human whether to keep waiting; "yes" just joins again — the same
    # generation is still running on the untouched connection.
    result: Dict = {}
    worker = threading.Thread(target=_request_worker, args=(req, result), daemon=True)
    worker.start()
    interval = max(1, int(getattr(config, "request_timeout", 300) or 300))
    waited = 0
    while True:
        worker.join(interval)
        if not worker.is_alive():
            break
        waited += interval
        if not _ask_keep_waiting(url, waited):
            raise LLMError(
                f"Aborted after waiting {waited}s for a reply from {url}. The model "
                "may still be generating — re-run to retry, or try a faster model "
                "or a smaller context.")

    err = result.get("error")
    if err is not None:
        raise _llm_error_for(err, url, config) from err
    try:
        body = json.loads(result["body"])
    except (ValueError, KeyError) as e:
        raise LLMError(f"Could not decode the response from {url}: {e}") from e

    try:
        choice = body["choices"][0]
        content = choice["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected response shape: {json.dumps(body)[:500]}") from e

    # A response truncated at the token ceiling is NOT a complete reply — the
    # closing ``` of an `act` block (or a whole plan) can be cut off, which the
    # parser would silently read as "no action / done" and the harness would
    # commit as finished work. That is the exact silent-work-loss class Achilles
    # keeps hitting, so we surface it as an error instead of parsing a fragment.
    if choice.get("finish_reason") == "length":
        if max_tokens and max_tokens > 0:
            raise LLMError(
                f"model output hit the {max_tokens}-token max_tokens cap and was "
                "truncated; raise max_tokens (or set it to 0 to use the model's "
                "full context) or split the step (a partial reply is unsafe to parse)."
            )
        raise LLMError(
            "model output filled the model's context window and was truncated; "
            "load the model with a larger context in LM Studio or split the step "
            "(a partial reply is unsafe to parse)."
        )
    return content
