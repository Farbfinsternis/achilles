"""
comfy_client.py — the thin wire to a local ComfyUI. Dependency-free.

ComfyUI speaks plain HTTP + JSON, so — exactly like llm.py — we use urllib and
nothing else. This module knows the FIVE endpoints the image path needs and
nothing about workflows, slots or model-swapping (that lives in workflows.py and
comfy.py). It is the dumbest possible layer: send bytes, read bytes.

  GET  /object_info        every node type the server has installed (+ its input
                           spec, incl. the allowed values of a combo field)
  POST /prompt             queue a workflow graph for execution
  GET  /history/{id}       poll for the finished run's outputs
  GET  /view?…             fetch one produced image's bytes
  POST /free               unload ComfyUI's models to hand VRAM back

Progress here is deliberately COARSE (poll /history until the run appears). The
fine-grained WebSocket step-bar is a later polish; the MVP only needs "is it done
yet, and where is the picture".
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str, timeout: int = 30):
        # One trailing-slash-free base, so we can concatenate "/prompt" etc.
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # ---- low-level HTTP (stdlib only) ---------------------------------

    def _get(self, path: str) -> bytes:
        url = self.base + path
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.URLError as e:
            raise ComfyError(f"Could not reach ComfyUI at {url}: {e.reason}. "
                             "Is ComfyUI running?") from e

    def _get_json(self, path: str) -> dict:
        return json.loads(self._get(path).decode("utf-8"))

    def _post_json(self, path: str, payload: dict) -> dict:
        url = self.base + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:800]
            raise ComfyError(f"HTTP {e.code} from {url}: {detail}") from e
        except urllib.error.URLError as e:
            raise ComfyError(f"Could not reach ComfyUI at {url}: {e.reason}. "
                             "Is ComfyUI running?") from e

    # ---- the five endpoints -------------------------------------------

    def reachable(self) -> bool:
        try:
            self._get("/system_stats")
            return True
        except ComfyError:
            return False

    def object_info(self) -> dict:
        """Every installed node type keyed by class_type. Used to (a) verify a
        workflow's custom nodes are present and (b) read a combo input's real
        allowed values so we never guess an enum string."""
        return self._get_json("/object_info")

    def queue_prompt(self, graph: dict, client_id: Optional[str] = None) -> str:
        payload = {"prompt": graph}
        if client_id:
            payload["client_id"] = client_id
        resp = self._post_json("/prompt", payload)
        # ComfyUI validates the graph up front; a bad graph comes back as
        # node_errors rather than an exception, so surface it honestly.
        if resp.get("node_errors"):
            raise ComfyError(f"ComfyUI rejected the workflow: {resp['node_errors']}")
        pid = resp.get("prompt_id")
        if not pid:
            raise ComfyError(f"ComfyUI returned no prompt_id: {resp}")
        return pid

    def wait_for_outputs(self, prompt_id: str, timeout: int = 600,
                         poll: float = 1.0,
                         on_tick: Optional[Callable[[float], None]] = None) -> dict:
        """Poll /history until this run finishes. Returns the run's `outputs`
        (node_id -> {images: [...]}). Raises on timeout or an execution error.
        /history is empty while the job queues/runs, so we just wait for it to
        appear — coarse but correct."""
        deadline = time.monotonic() + timeout
        start = time.monotonic()
        while time.monotonic() < deadline:
            hist = self._get_json(f"/history/{prompt_id}")
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyError(f"ComfyUI run failed: "
                                     f"{_first_error(status)}")
                if entry.get("outputs"):
                    return entry["outputs"]
            if on_tick:
                on_tick(time.monotonic() - start)
            time.sleep(poll)
        raise ComfyError(f"ComfyUI run {prompt_id} did not finish within {timeout}s.")

    def image_bytes(self, image: dict) -> bytes:
        """Fetch one produced image. `image` is an entry from an output node's
        `images` list: {filename, subfolder, type}. Works for both SaveImage
        (type=output) and PreviewImage (type=temp), so any workflow yields a file."""
        q = urllib.parse.urlencode({
            "filename": image.get("filename", ""),
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        })
        return self._get(f"/view?{q}")

    def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        """Hand VRAM back: unload ComfyUI's checkpoints so the LM can reload.
        Best-effort — a failure here must not abort the caller's finally-block."""
        try:
            self._post_json("/free", {"unload_models": unload_models,
                                      "free_memory": free_memory})
        except ComfyError:
            pass


def first_image(outputs: dict) -> Optional[dict]:
    """Pull the first image entry out of a run's outputs, whichever node produced
    it (SaveImage, PreviewImage, …). None if the run made no image."""
    for node_out in outputs.values():
        for img in node_out.get("images", []) or []:
            return img
    return None


def _first_error(status: dict) -> str:
    for msg in status.get("messages", []) or []:
        # messages are [event_name, {details}] pairs; execution_error carries the text.
        if isinstance(msg, list) and len(msg) == 2 and "error" in str(msg[0]).lower():
            d = msg[1]
            return d.get("exception_message") or str(d)
    return "unknown error"
