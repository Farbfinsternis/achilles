"""Tests for llm.py — specifically the truncation guard.

A reply truncated at max_tokens (finish_reason == "length") is an incomplete
reply: parsing the fragment silently loses work. chat() must raise instead.
"""

import io
import json
import contextlib

import pytest

from achilles import llm
from achilles.llm import chat, LLMError


class _FakeConfig:
    base_url = "http://localhost:9/v1"
    api_key = "no-key"
    model = "test-model"
    request_timeout = 5
    max_tokens = 0   # auto: adopt the model's own context limit


def _fake_response(body: dict):
    """A urlopen() stand-in: a context manager whose .read() yields JSON bytes."""
    @contextlib.contextmanager
    def _cm(req, timeout=None):
        yield io.BytesIO(json.dumps(body).encode("utf-8"))
    return _cm


def test_normal_reply_returned(monkeypatch):
    body = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_response(body))
    assert chat(_FakeConfig(), [{"role": "user", "content": "hi"}]) == "hello"


def test_truncated_reply_raises(monkeypatch):
    body = {"choices": [{"message": {"content": "def foo(): # cut o"},
                         "finish_reason": "length"}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_response(body))
    with pytest.raises(LLMError, match="truncated"):
        chat(_FakeConfig(), [{"role": "user", "content": "hi"}])


def test_missing_finish_reason_is_tolerated(monkeypatch):
    # Some servers omit finish_reason; absence must not be treated as truncation.
    body = {"choices": [{"message": {"content": "ok"}}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_response(body))
    assert chat(_FakeConfig(), [{"role": "user", "content": "hi"}]) == "ok"


def _capturing_response(body: dict, sink: dict):
    """Like _fake_response but records the outgoing request payload in `sink`."""
    @contextlib.contextmanager
    def _cm(req, timeout=None):
        sink["payload"] = json.loads(req.data.decode("utf-8"))
        yield io.BytesIO(json.dumps(body).encode("utf-8"))
    return _cm


def test_max_tokens_omitted_when_auto(monkeypatch):
    # config.max_tokens == 0 → no cap sent, so the engine fills the model's context.
    body = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    sink = {}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _capturing_response(body, sink))
    chat(_FakeConfig(), [{"role": "user", "content": "hi"}])
    assert "max_tokens" not in sink["payload"]


def test_max_tokens_sent_when_capped(monkeypatch):
    # An explicit positive cap is forwarded to the engine as-is.
    body = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    sink = {}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _capturing_response(body, sink))
    chat(_FakeConfig(), [{"role": "user", "content": "hi"}], max_tokens=512)
    assert sink["payload"]["max_tokens"] == 512


def test_read_timeout_becomes_llmerror(monkeypatch):
    # A slow generation outlasting request_timeout surfaces as a bare TimeoutError
    # during the response read. It must be caught (halt the step), not crash the REPL.
    def _boom(req, timeout=None):
        raise TimeoutError("timed out")
    monkeypatch.setattr(llm.urllib.request, "urlopen", _boom)
    with pytest.raises(LLMError, match="request_timeout"):
        chat(_FakeConfig(), [{"role": "user", "content": "hi"}])


def test_connect_timeout_wrapped_in_urlerror(monkeypatch):
    # A timeout during connect arrives wrapped in URLError; same clean handling.
    def _boom(req, timeout=None):
        raise llm.urllib.error.URLError(TimeoutError("timed out"))
    monkeypatch.setattr(llm.urllib.request, "urlopen", _boom)
    with pytest.raises(LLMError, match="request_timeout"):
        chat(_FakeConfig(), [{"role": "user", "content": "hi"}])


def test_keeps_waiting_then_returns(monkeypatch):
    # A slow generation crosses request_timeout once; the user says "keep waiting",
    # and the SAME call then completes and its content is returned. No restart.
    import threading
    release = threading.Event()
    body = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    @contextlib.contextmanager
    def _slow(req, timeout=None):
        release.wait(2)          # block as if the model were still generating
        yield io.BytesIO(json.dumps(body).encode("utf-8"))
    monkeypatch.setattr(llm.urllib.request, "urlopen", _slow)

    asked = []

    def _fake_ask(url, waited):
        asked.append(waited)
        release.set()            # let the "still running" generation finish
        return True
    monkeypatch.setattr(llm, "_ask_keep_waiting", _fake_ask)

    class _Cfg(_FakeConfig):
        request_timeout = 0.05   # force at least one keep-waiting check-in

    assert chat(_Cfg(), [{"role": "user", "content": "hi"}]) == "ok"
    assert asked and asked[0] >= 0.05   # the prompt actually fired


def test_abort_while_waiting_raises(monkeypatch):
    # The user declines to keep waiting → a clean abort, not a hang or a restart.
    import threading
    release = threading.Event()
    body = {"choices": [{"message": {"content": "never read"}}]}

    @contextlib.contextmanager
    def _slow(req, timeout=None):
        release.wait(2)
        yield io.BytesIO(json.dumps(body).encode("utf-8"))
    monkeypatch.setattr(llm.urllib.request, "urlopen", _slow)
    monkeypatch.setattr(llm, "_ask_keep_waiting", lambda url, waited: False)

    class _Cfg(_FakeConfig):
        request_timeout = 0.05

    with pytest.raises(LLMError, match="Aborted"):
        chat(_Cfg(), [{"role": "user", "content": "hi"}])
    release.set()   # unblock the orphaned worker so it can exit


def test_context_full_truncation_raises(monkeypatch):
    # Uncapped (max_tokens=0) but still truncated → the context window is full;
    # the error must point at the model's context, not a max_tokens cap.
    body = {"choices": [{"message": {"content": "def foo(): # cut o"},
                         "finish_reason": "length"}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_response(body))
    with pytest.raises(LLMError, match="context window"):
        chat(_FakeConfig(), [{"role": "user", "content": "hi"}])
