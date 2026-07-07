"""
ws_channel.py — WebSocket transport for the engine/UI boundary (docs/protocol.md §9).

Stdlib only (asyncio, hashlib, base64, struct) — Achilles keeps its zero-dependency
promise. A minimal RFC 6455 server: the HTTP upgrade handshake, a text-frame codec,
and a WebSocketChannel that bridges the SYNCHRONOUS engine to the async socket.

The bridge is the whole point of keeping the harness synchronous:

  * The engine runs in a WORKER THREAD; Harness.run() stays linear, blocking code.
  * emit(type, data) schedules the message onto the loop's outbound queue
    (call_soon_threadsafe — emit is called from the worker thread).
  * request(type, data) emits a request event and BLOCKS the worker thread on a
    thread-safe slot until the receive loop routes back a command whose reply_to
    matches — then request() returns that command's data.

Only this module knows async; the harness sees just emit/request. One Harness.run()
per connection (v1: one run per process-ish, no auth).
"""

import asyncio
import base64
import hashlib
import json
import queue
import struct
import threading
import time

from .channel import Channel

# RFC 6455: the server appends this GUID to the client's key, SHA-1s it, and
# base64s the digest into Sec-WebSocket-Accept.
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Opcodes we care about.
_OP_TEXT = 0x1
_OP_CLOSE = 0x8
_OP_PING = 0x9
_OP_PONG = 0xA


def _accept_key(client_key: str) -> str:
    digest = hashlib.sha1((client_key + _WS_GUID).encode()).digest()
    return base64.b64encode(digest).decode()


def _encode_text(payload: str) -> bytes:
    """A single unmasked text frame (server → client). FIN=1, opcode=text."""
    data = payload.encode("utf-8")
    header = bytearray([0x80 | _OP_TEXT])
    n = len(data)
    if n < 126:
        header.append(n)
    elif n < 65536:
        header.append(126)
        header += struct.pack(">H", n)
    else:
        header.append(127)
        header += struct.pack(">Q", n)
    return bytes(header) + data


async def _read_frame(reader) -> tuple:
    """Read one frame → (opcode, payload_bytes). Client frames are masked (RFC 6455
    requires it); we unmask. Raises asyncio.IncompleteReadError when the peer closes."""
    b0, b1 = await reader.readexactly(2)
    opcode = b0 & 0x0F
    masked = b1 & 0x80
    length = b1 & 0x7F
    if length == 126:
        length = struct.unpack(">H", await reader.readexactly(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", await reader.readexactly(8))[0]
    mask = await reader.readexactly(4) if masked else b"\x00\x00\x00\x00"
    data = bytearray(await reader.readexactly(length))
    if masked:
        for i in range(length):
            data[i] ^= mask[i % 4]
    return opcode, bytes(data)


class WebSocketChannel(Channel):
    """The engine/UI boundary over a WebSocket. emit/request are called from the
    engine's worker thread; both hand off to the asyncio loop thread-safely."""

    def __init__(self, loop, out_queue, run_id: str):
        self._loop = loop
        self._out = out_queue            # asyncio.Queue drained by the sender task
        self._run = run_id
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._pending = {}               # request id -> queue.Queue(maxsize=1)
        self._pending_lock = threading.Lock()
        self._req_counter = 0

    def _envelope(self, type: str, data: dict, id: str = None) -> dict:
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        env = {"v": 1, "run": self._run, "seq": seq,
               "ts": int(time.time() * 1000), "type": type, "data": data or {}}
        if id is not None:
            env["id"] = id
        return env

    def _send(self, env: dict) -> None:
        # Called from the worker thread → schedule the put on the loop thread.
        self._loop.call_soon_threadsafe(self._out.put_nowait, env)

    def emit(self, type: str, data: dict) -> None:
        self._send(self._envelope(type, data))

    def request(self, type: str, data: dict) -> dict:
        with self._pending_lock:
            self._req_counter += 1
            rid = f"req-{self._req_counter}"
            slot = queue.Queue(maxsize=1)
            self._pending[rid] = slot
        self._send(self._envelope(type, data, id=rid))
        reply = slot.get()               # blocks the worker thread until delivered
        with self._pending_lock:
            self._pending.pop(rid, None)
        return reply or {}

    def deliver(self, reply_to: str, data: dict) -> None:
        """Route an incoming command (from the receive loop) to the blocked request."""
        if not reply_to:
            return
        with self._pending_lock:
            slot = self._pending.get(reply_to)
        if slot is not None:
            slot.put(data or {})


# ---- server ---------------------------------------------------------------

async def _read_http_headers(reader) -> dict:
    await reader.readline()              # request line: GET /… HTTP/1.1
    headers = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        key, _, val = line.decode("latin1").partition(":")
        if val:
            headers[key.strip().lower()] = val.strip()
    return headers


def _start_engine(cfg, goal: str, mode: str, channel: WebSocketChannel,
                  loop, out) -> threading.Thread:
    """Run one Harness.run() in a worker thread, driven by the WebSocketChannel.
    Emits the run.started/run.finished/error lifecycle (the driver owns it; the
    harness emits the domain events) and then signals the sender to close."""
    from .harness import Harness

    def run():
        try:
            channel.emit("run.started", {"goal": goal, "mode": mode})
            # The channel is the log sink too — Harness routes self.log through it.
            harness = Harness(cfg, mode=mode, channel=channel)
            ok = harness.run(goal)
            channel.emit("run.finished", {"result": "success" if ok else "halted"})
        except Exception as e:                      # noqa: BLE001 — surface, don't crash the loop
            channel.emit("error", {"fatal": True, "message": str(e)})
            channel.emit("run.finished", {"result": "failed", "reason": str(e)})
        finally:
            loop.call_soon_threadsafe(out.put_nowait, None)   # stop the sender

    t = threading.Thread(target=run, name="achilles-engine", daemon=True)
    t.start()
    return t


async def _handle(reader, writer, cfg, log) -> None:
    headers = await _read_http_headers(reader)
    key = headers.get("sec-websocket-key")
    if not key:
        writer.close()
        return
    resp = ("HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {_accept_key(key)}\r\n\r\n")
    writer.write(resp.encode())
    await writer.drain()

    loop = asyncio.get_running_loop()
    out = asyncio.Queue()
    run_id = f"run-{int(time.time() * 1000)}"
    channel = WebSocketChannel(loop, out, run_id)

    async def sender():
        while True:
            env = await out.get()
            if env is None:                          # sentinel: run finished
                break
            writer.write(_encode_text(json.dumps(env)))
            await writer.drain()

    send_task = asyncio.create_task(sender())
    engine = None
    try:
        while True:
            opcode, payload = await _read_frame(reader)
            if opcode == _OP_CLOSE:
                break
            if opcode == _OP_PING:
                writer.write(bytes([0x80 | _OP_PONG]) + b"\x00")
                await writer.drain()
                continue
            if opcode != _OP_TEXT:
                continue
            msg = json.loads(payload.decode("utf-8"))
            mtype = msg.get("type")
            data = msg.get("data", {}) or {}
            if mtype == "run.start" and engine is None:
                engine = _start_engine(cfg, data.get("goal", ""),
                                       data.get("mode", "autopilot"), channel, loop, out)
            elif mtype in ("answer", "approval"):
                channel.deliver(msg.get("reply_to") or data.get("reply_to"), data)
            # cancel and other commands: not wired in v1 (see protocol §10).
    except (asyncio.IncompleteReadError, ConnectionError):
        pass
    finally:
        loop.call_soon_threadsafe(out.put_nowait, None)     # ensure the sender stops
        try:
            await send_task
        except Exception:
            pass
        writer.close()
    log(f"   connection closed ({run_id})")


def serve(cfg, host: str = "127.0.0.1", port: int = 8765, log=print) -> int:
    """Run the WebSocket server until interrupted. One Harness.run() per connection."""
    async def main():
        server = await asyncio.start_server(
            lambda r, w: _handle(r, w, cfg, log), host, port)
        log(f"Achilles WebSocket server on ws://{host}:{port}  (Ctrl-C to stop)")
        async with server:
            await server.serve_forever()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("\nstopped.")
    return 0
