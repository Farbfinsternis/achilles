"""
llm.py — the engine connector. Dependency-free.

Talks to any OpenAI-compatible /chat/completions endpoint: llama.cpp's server,
Ollama's OpenAI-compat port, vLLM, LM Studio, or a cloud API. We use the stdlib
(urllib) on purpose — a minimal harness should have nothing to `pip install`.

Non-streaming: a plan/act turn is short and we want the whole text before we
parse it, so streaming would only add complexity here.
"""

import json
import urllib.error
import urllib.request
from typing import List, Dict


class LLMError(RuntimeError):
    pass


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
    try:
        with urllib.request.urlopen(req, timeout=config.request_timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise LLMError(f"HTTP {e.code} from {url}: {detail}") from e
    except TimeoutError as e:
        # A slow generation (common now that max_tokens is uncapped and the model
        # fills its whole context) can outlast request_timeout while the server is
        # still healthy. This surfaces as a bare TimeoutError during the response
        # read — NOT a URLError — so catch it explicitly. Left uncaught it crashes
        # the whole REPL with a raw traceback instead of halting just the step.
        raise LLMError(
            f"No response from {url} within request_timeout={config.request_timeout}s. "
            "The model may still be generating a long reply — raise request_timeout "
            "in achilles.toml, or cap output with max_tokens."
        ) from e
    except urllib.error.URLError as e:
        # URLError can also wrap a socket timeout (when it fires during connect
        # rather than read); treat that the same as the bare TimeoutError above.
        if isinstance(e.reason, TimeoutError):
            raise LLMError(
                f"No response from {url} within request_timeout={config.request_timeout}s. "
                "The model may still be generating a long reply — raise request_timeout "
                "in achilles.toml, or cap output with max_tokens."
            ) from e
        raise LLMError(
            f"Could not reach {url}: {e.reason}. Is your model server running?"
        ) from e

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
