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
