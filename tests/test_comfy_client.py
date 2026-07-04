"""Tests for comfy_client.py — the thin HTTP wire to ComfyUI.

Focus: a mid-render ComfyUI crash (the socket resets or refuses) must surface as a
ComfyError, not a raw OSError that escapes the tool's `except ComfyError` and takes
down the whole run. This was found the hard way when a heavy workflow OOM'd ComfyUI
during /history polling.
"""

import urllib.error

import pytest

from achilles import comfy_client as cc


def test_get_wraps_connection_reset(monkeypatch):
    def boom(*a, **k):
        raise ConnectionResetError(10054, "connection reset by peer")
    monkeypatch.setattr(cc.urllib.request, "urlopen", boom)
    client = cc.ComfyClient("http://127.0.0.1:8188")
    with pytest.raises(cc.ComfyError) as ei:
        client._get("/history/abc")
    assert "Lost the connection" in str(ei.value)


def test_wait_for_outputs_wraps_reset_not_raw(monkeypatch):
    def boom(*a, **k):
        raise ConnectionResetError(10054, "connection reset by peer")
    monkeypatch.setattr(cc.urllib.request, "urlopen", boom)
    client = cc.ComfyClient("http://127.0.0.1:8188")
    with pytest.raises(cc.ComfyError):        # NOT a bare ConnectionResetError
        client.wait_for_outputs("abc", timeout=5)


def test_urlerror_still_wrapped(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("refused")
    monkeypatch.setattr(cc.urllib.request, "urlopen", boom)
    client = cc.ComfyClient("http://127.0.0.1:8188")
    assert client.reachable() is False        # reachable() swallows ComfyError
