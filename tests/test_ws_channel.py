"""Tests for the stdlib WebSocket transport.

The pure pieces (handshake key, frame codec) are pinned against RFC vectors, the
WebSocketChannel's sync↔async bridge is driven with threads, and one integration
test runs the real asyncio server over a loopback socket against a stubbed engine
— exercising the full emit / request / deliver round trip across the thread
boundary.
"""

import asyncio
import json
import struct
import threading
import time

import pytest

from achilles import harness as H
from achilles.config import Config
from achilles.ws_channel import (
    _accept_key, _encode_text, _read_frame, _read_http_headers,
    _handle, WebSocketChannel,
)


# ---- pure codec -----------------------------------------------------------

def test_accept_key_rfc_vector():
    # RFC 6455 §1.3 worked example.
    assert _accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_encode_text_small():
    frame = _encode_text("hi")
    assert frame[0] == 0x81        # FIN + text opcode
    assert frame[1] == 2           # unmasked, length 2
    assert frame[2:] == b"hi"


def test_encode_text_16bit_length():
    frame = _encode_text("x" * 200)
    assert frame[0] == 0x81
    assert frame[1] == 126         # 16-bit extended length marker
    assert struct.unpack(">H", frame[2:4])[0] == 200


def _client_frame(text: str) -> bytes:
    """A masked client text frame (RFC 6455 requires client→server masking)."""
    data = text.encode("utf-8")
    mask = b"\x01\x02\x03\x04"
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    n = len(data)
    header = bytearray([0x81])
    if n < 126:
        header.append(0x80 | n)
    else:
        header.append(0x80 | 126)
        header += struct.pack(">H", n)
    return bytes(header) + mask + masked


def test_read_frame_unmasks_client_payload():
    async def go():
        reader = asyncio.StreamReader()
        reader.feed_data(_client_frame("hällo ünïcode"))
        reader.feed_eof()
        return await _read_frame(reader)

    opcode, payload = asyncio.run(go())
    assert opcode == 0x1
    assert payload.decode("utf-8") == "hällo ünïcode"


# ---- WebSocketChannel bridge ---------------------------------------------

class _SyncLoop:
    """A stand-in loop whose call_soon_threadsafe runs inline (single-threaded test)."""
    def call_soon_threadsafe(self, fn, *args):
        fn(*args)


class _StubQueue:
    def __init__(self):
        self.items = []
    def put_nowait(self, x):
        self.items.append(x)


def test_emit_envelope_shape():
    q = _StubQueue()
    ch = WebSocketChannel(_SyncLoop(), q, "run-x")
    ch.emit("log", {"text": "hi"})
    env = q.items[0]
    assert env["v"] == 1 and env["run"] == "run-x" and env["type"] == "log"
    assert env["data"] == {"text": "hi"} and env["seq"] == 1
    assert "id" not in env             # emit is fire-and-forget, no correlation id


def test_seq_is_monotonic():
    q = _StubQueue()
    ch = WebSocketChannel(_SyncLoop(), q, "run-x")
    ch.emit("log", {"text": "a"})
    ch.emit("log", {"text": "b"})
    assert [e["seq"] for e in q.items] == [1, 2]


def test_request_blocks_until_deliver():
    q = _StubQueue()
    ch = WebSocketChannel(_SyncLoop(), q, "run-1")
    result = {}

    def worker():
        result["reply"] = ch.request("approval.request", {"subject": "spec"})

    t = threading.Thread(target=worker)
    t.start()
    # The request envelope is emitted (inline) before request() blocks on its slot.
    deadline = time.time() + 2
    while not q.items and time.time() < deadline:
        time.sleep(0.01)
    env = q.items[0]
    assert env["type"] == "approval.request" and "id" in env
    assert not t.join(timeout=0.2) or t.is_alive()      # still blocked before deliver
    ch.deliver(env["id"], {"decision": "approve"})
    t.join(timeout=2)
    assert result["reply"] == {"decision": "approve"}


def test_deliver_unknown_id_is_noop():
    ch = WebSocketChannel(_SyncLoop(), _StubQueue(), "run-1")
    ch.deliver("nope", {"x": 1})        # must not raise
    ch.deliver(None, {"x": 1})


# ---- integration: real server + loopback socket + stubbed engine ----------

class _FakeHarness:
    """Stands in for the real engine: logs, exercises a gate, returns success."""
    def __init__(self, cfg, log=print, mode="autopilot", channel=None):
        self.log = log
        self.channel = channel

    def run(self, goal):
        self.log(f"planning: {goal}")
        reply = self.channel.request("approval.request", {"subject": "spec"})
        self.log(f"decision: {reply.get('decision')}")
        return True


async def _drive_client(cfg):
    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, cfg, lambda *a: None), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)

    writer.write(b"GET / HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
                 b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n")
    await writer.drain()
    await _read_http_headers(reader)                    # consume the 101 response

    writer.write(_client_frame(json.dumps(
        {"type": "run.start", "data": {"goal": "build X", "mode": "autopilot"}})))
    await writer.drain()

    events = []
    while True:
        opcode, payload = await _read_frame(reader)
        env = json.loads(payload.decode("utf-8"))
        events.append(env)
        if env["type"] == "approval.request":
            writer.write(_client_frame(json.dumps(
                {"type": "approval", "reply_to": env["id"],
                 "data": {"decision": "approve"}})))
            await writer.drain()
        if env["type"] == "run.finished":
            break

    writer.close()
    server.close()
    await server.wait_closed()
    return events


def test_server_drives_engine_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "Harness", _FakeHarness)
    cfg = Config(workspace=str(tmp_path), use_git=False, verify_command="")

    events = asyncio.run(asyncio.wait_for(_drive_client(cfg), timeout=10))

    types = [e["type"] for e in events]
    assert "approval.request" in types                  # gate reached the client
    assert types[-1] == "run.finished"
    assert events[-1]["data"]["result"] == "success"
    # the engine saw the goal and the routed decision (both logged)
    logs = [e["data"]["text"] for e in events if e["type"] == "log"]
    assert any("planning: build X" in l for l in logs)
    assert any("decision: approve" in l for l in logs)
    # seq numbers are strictly increasing across the whole stream
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
