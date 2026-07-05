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
from dataclasses import dataclass
from typing import List, Dict, Optional


class LLMError(RuntimeError):
    pass


def wants_constrained_json(config) -> bool:
    """True when the run is on the constrained content-JSON protocol
    (act_protocol="json"). The planner, the Definition of Done and the judge honour
    the SAME switch as the act-loop, so one setting turns grammar-enforced structure
    on everywhere a weak model would otherwise fumble the output format."""
    return (getattr(config, "act_protocol", "native") or "").strip().lower() == "json"


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
    body = _send(config, payload)

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
        raise _length_error(max_tokens)
    return content


def _length_error(max_tokens: int) -> "LLMError":
    """The shared 'truncated at the token ceiling' error — a partial reply (text
    OR a tool_call with cut-off arguments) is unsafe to parse."""
    if max_tokens and max_tokens > 0:
        return LLMError(
            f"model output hit the {max_tokens}-token max_tokens cap and was "
            "truncated; raise max_tokens (or set it to 0 to use the model's "
            "full context) or split the step (a partial reply is unsafe to parse).")
    return LLMError(
        "model output filled the model's context window and was truncated; "
        "load the model with a larger context in LM Studio or split the step "
        "(a partial reply is unsafe to parse).")


def _send(config, payload: Dict) -> Dict:
    """POST one completion request and return the parsed JSON body.

    Runs the blocking call in a worker thread and manages the clock here, so a
    slow-but-healthy generation is never killed: every request_timeout seconds of
    silence we ask the human whether to keep waiting; "yes" just joins again — the
    same generation is still running on the untouched connection."""
    url = config.base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
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
        return json.loads(result["body"])
    except (ValueError, KeyError) as e:
        raise LLMError(f"Could not decode the response from {url}: {e}") from e


@dataclass
class ActReply:
    """One native-tool-calling turn: the assistant's text (a preamble, or the
    'I'm done' sentence) plus any tool_calls it emitted. tool_calls entries are
    {'id', 'name', 'arguments': dict} — already JSON-decoded."""
    content: str
    tool_calls: List[Dict]
    finish_reason: Optional[str] = None


@dataclass
class JsonReply:
    """One constrained-content-JSON turn: the raw content string and, when it parsed
    as a JSON object, the decoded dict. `obj` is None if the server ignored the
    json_schema and returned something unparseable — the caller then degrades to
    the text protocol on the SAME content instead of trusting a fragment."""
    obj: Optional[dict]
    content: str
    finish_reason: Optional[str] = None


def complete_json(config, messages: List[Dict], schema: Dict,
                  temperature: float = 0.2, max_tokens: int | None = None,
                  model: str | None = None) -> "JsonReply":
    """Like chat(), but constrains the reply to `schema` on the CONTENT channel via
    response_format=json_schema — the one channel LM Studio actually grammar-enforces
    (its tool-call `arguments` channel is only a hint). No `tools` field is sent: the
    act-call rides entirely in the content JSON. Returns the parsed object (or None if
    the content wasn't valid JSON, so the caller can fall back)."""
    if max_tokens is None:
        max_tokens = getattr(config, "max_tokens", 0) or 0
    payload = {
        "model": model or config.model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "act", "strict": True,
                                            "schema": schema}},
    }
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    body = _send(config, payload)

    try:
        choice = body["choices"][0]
        content = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected response shape: {json.dumps(body)[:500]}") from e

    # A truncated JSON object is an unsafe fragment (same class as the text/tool
    # paths): the closing brace or a whole "content" field can be cut off.
    if choice.get("finish_reason") == "length":
        raise _length_error(max_tokens)

    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        obj = None
    if not isinstance(obj, dict):
        obj = None
    return JsonReply(obj=obj, content=content,
                     finish_reason=choice.get("finish_reason"))


def complete_act(config, messages: List[Dict], tools: List[Dict],
                 temperature: float = 0.2, max_tokens: int | None = None,
                 model: str | None = None) -> "ActReply":
    """Like chat(), but offers NATIVE tool-calling: send the `tools` schema and
    return the assistant's tool_calls (if any) alongside its text. A tool-tuned
    model answers with structured tool_calls here where it would fumble the text
    `act` protocol. If it returns plain text instead, tool_calls is empty and the
    caller falls back to parsing the text — so both formats coexist."""
    if max_tokens is None:
        max_tokens = getattr(config, "max_tokens", 0) or 0
    payload = {
        "model": model or config.model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "tools": tools,
    }
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    body = _send(config, payload)

    try:
        choice = body["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected response shape: {json.dumps(body)[:500]}") from e

    calls: List[Dict] = []
    for raw in (msg.get("tool_calls") or []):
        fn = raw.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append({"id": raw.get("id") or f"call_{len(calls)}",
                      "name": fn.get("name") or "",
                      "arguments": args})

    # Truncation still means an unsafe fragment — but only when there is text to
    # parse; a complete set of tool_calls with finish_reason 'length' is rare and
    # the structured args are self-delimiting, so we only guard the text path.
    if choice.get("finish_reason") == "length" and not calls:
        raise _length_error(max_tokens)
    return ActReply(content=msg.get("content") or "", tool_calls=calls,
                    finish_reason=choice.get("finish_reason"))
