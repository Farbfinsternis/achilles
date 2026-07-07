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
import urllib.parse
import urllib.request
from pathlib import Path

from .channel import Channel
from . import sessions

# The static web UI lives next to this package; `achilles --serve` serves it over
# HTTP on the SAME port as the WebSocket (the handler tells a GET from an Upgrade).
_WEB_DIR = Path(__file__).resolve().parent / "web"
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}

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

async def _read_request(reader) -> tuple:
    """Read the request line + headers → (method, path, headers). The method/path
    let one handler route an HTTP GET (static UI / api) apart from a WS Upgrade."""
    request_line = await reader.readline()             # e.g. b"GET /app.js HTTP/1.1"
    parts = request_line.decode("latin1").split()
    method = parts[0] if parts else ""
    path = parts[1] if len(parts) > 1 else "/"
    headers = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        key, _, val = line.decode("latin1").partition(":")
        if val:
            headers[key.strip().lower()] = val.strip()
    return method, path, headers


def _http_response(status: str, body: bytes, content_type: str) -> bytes:
    head = (f"HTTP/1.1 {status}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "Cache-Control: no-cache\r\n\r\n")
    return head.encode("latin1") + body


def _list_models(cfg) -> bytes:
    """Proxy the model server's catalogue so the UI's model dropdown is populated
    from LM Studio (/v1/models) without a cross-origin fetch. Best-effort: an
    unreachable server yields an empty list rather than an error, so the UI still
    loads (the user can type/leave the config default)."""
    ids = []
    try:
        with urllib.request.urlopen(cfg.base_url.rstrip("/") + "/models", timeout=5) as r:
            body = json.loads(r.read().decode("utf-8"))
        ids = [m.get("id") for m in body.get("data", []) if m.get("id")]
    except Exception:                                  # noqa: BLE001 — offline model server is fine
        ids = []
    payload = {"models": ids, "default": cfg.model}
    return json.dumps(payload).encode("utf-8")


def _json_ok(obj) -> bytes:
    return _http_response("200 OK", json.dumps(obj).encode("utf-8"), _CONTENT_TYPES[".json"])


async def _serve_http(reader, writer, method: str, path: str, headers: dict, cfg) -> None:
    """Serve the static UI and the small JSON API over HTTP on the WebSocket port.
      GET  /api/models              — the model dropdown catalogue (LM Studio proxy)
      GET  /api/recents             — known projects + their session summaries
      GET  /api/session?path=&id=   — one session's meta + replayable event stream
      POST /api/project  {path}     — register a project into the recents index
    Anything else is a static file from the web dir."""
    route, _, raw_query = path.partition("?")
    query = urllib.parse.parse_qs(raw_query)
    q = lambda k: (query.get(k, [""])[0])

    body = b""
    if method == "POST":
        n = int(headers.get("content-length", "0") or 0)
        if n:
            try:
                body = await reader.readexactly(n)
            except asyncio.IncompleteReadError:
                body = b""

    if route == "/api/models" and method == "GET":
        writer.write(_http_response("200 OK", _list_models(cfg), _CONTENT_TYPES[".json"]))
    elif route == "/api/recents" and method == "GET":
        writer.write(_json_ok(sessions.list_recents()))
    elif route == "/api/session" and method == "GET":
        writer.write(_json_ok(sessions.load_session(q("path"), q("id"))))
    elif route == "/api/project" and method == "POST":
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except ValueError:
            payload = {}
        proj = (payload.get("path") or "").strip()
        if proj:
            sessions.register_project(proj, (payload.get("name") or "").strip())
            writer.write(_json_ok({"ok": True}))
        else:
            writer.write(_http_response("400 Bad Request", b"path required", "text/plain"))
    elif method != "GET":
        writer.write(_http_response("405 Method Not Allowed", b"", "text/plain"))
    else:
        rel = "index.html" if route in ("/", "") else route.lstrip("/")
        target = (_WEB_DIR / rel).resolve()
        # Path-traversal guard: never serve outside the web dir.
        if _WEB_DIR in target.parents and target.is_file():
            ctype = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
            writer.write(_http_response("200 OK", target.read_bytes(), ctype))
        else:
            writer.write(_http_response("404 Not Found", b"not found", "text/plain"))
    await writer.drain()
    writer.close()


def _run_config(base_cfg, data: dict):
    """The per-run config for a run.start: load the SELECTED project's config (its
    cwd/achilles.toml), then apply UI overrides (model). Falls back to the server's
    own config when no cwd is given (protocol §4: run.start {…, cwd, config_overrides})."""
    from .config import load_config
    cwd = (data.get("cwd") or "").strip()
    cfg = load_config(cwd) if cwd else base_cfg
    overrides = data.get("config_overrides") or {}
    model = (overrides.get("model") or "").strip()
    if model:
        cfg.model = model
    return cfg


def _start_engine(cfg, goal: str, mode: str, channel: WebSocketChannel,
                  loop, out, session_id: str = "", project_path: str = "") -> threading.Thread:
    """Run one Harness.run() in a worker thread, driven by the WebSocketChannel.
    Emits the run.started/run.finished/error lifecycle (the driver owns it; the
    harness emits the domain events) and then signals the sender to close."""
    from .harness import Harness

    def run():
        try:
            channel.emit("run.started", {"goal": goal, "mode": mode,
                                         "session_id": session_id, "path": project_path})
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
    method, path, headers = await _read_request(reader)
    # A WebSocket handshake carries Upgrade: websocket; anything else is a plain
    # HTTP GET for the static UI or the /api/models endpoint.
    if headers.get("upgrade", "").lower() != "websocket":
        await _serve_http(reader, writer, method, path, headers, cfg)
        return
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
    store = None                                     # set on run.start; the sender persists through it

    async def sender():
        while True:
            env = await out.get()
            if env is None:                          # sentinel: run finished
                break
            if store is not None:                    # persist every outgoing event (single choke point)
                store.append(env)
                if env.get("type") == "run.finished":
                    store.finalize(env.get("data", {}).get("result", "success"))
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
                run_cfg = _run_config(cfg, data)
                proj_path = str(run_cfg.workspace_path)
                mode = data.get("mode", "autopilot")
                # Persist the session next to its project and register it in recents,
                # so the run's transcript survives and the left rail can list it.
                store = sessions.SessionStore(proj_path, run_id, {
                    "goal": data.get("goal", ""), "mode": mode, "model": run_cfg.model})
                sessions.register_project(proj_path)
                engine = _start_engine(run_cfg, data.get("goal", ""), mode,
                                       channel, loop, out,
                                       session_id=run_id, project_path=proj_path)
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
        log(f"Achilles web UI on http://{host}:{port}  ·  WebSocket ws://{host}:{port}  (Ctrl-C to stop)")
        async with server:
            await server.serve_forever()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("\nstopped.")
    return 0
