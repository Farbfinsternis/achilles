"""
sessions.py — server-side persistence for the web UI's projects and sessions.

Two layers, mirroring Achilles' existing on-disk conventions (~/.achilles for the
global home, <workspace>/.achilles for per-project state):

  * A GLOBAL project index at ~/.achilles/projects.json — the registry of known
    project directories that feeds the "recent projects" rail. Small and fast.
  * PER-PROJECT session records under <workspace>/.achilles/sessions/:
      <id>.json   — the session meta (goal, mode, model, timestamps, result)
      <id>.jsonl  — the event stream, one envelope per line (append-only, so a
                    crash mid-run keeps the transcript so far; replay reads it back)
    These live next to the project they belong to and ride the existing
    .achilles/ gitignore, so history never pollutes the repo.

The global index is only the registry; a project's sessions are discovered by
scanning its own sessions dir, so the two never drift.

Stdlib only, in keeping with Achilles' zero-dependency rule.
"""

import json
import threading
import time
from pathlib import Path


def _home_index() -> Path:
    return Path.home() / ".achilles" / "projects.json"


def _sessions_dir(project_path: str) -> Path:
    return Path(project_path) / ".achilles" / "sessions"


_index_lock = threading.Lock()


# ---- global project index -------------------------------------------------

def _read_index() -> list:
    try:
        return json.loads(_home_index().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def register_project(path: str, name: str = "") -> None:
    """Upsert a project into the global index (creating ~/.achilles if needed) and
    stamp it as most-recently used. Idempotent: re-registering just bumps the time."""
    path = str(Path(path).resolve())
    name = name or Path(path).name or path
    with _index_lock:
        index = _read_index()
        entry = next((e for e in index if e.get("path") == path), None)
        if entry is None:
            entry = {"path": path, "name": name}
            index.append(entry)
        entry["name"] = name
        entry["last_used"] = int(time.time() * 1000)
        idx = _home_index()
        idx.parent.mkdir(parents=True, exist_ok=True)
        idx.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _session_summaries(project_path: str) -> list:
    """The meta of every session under a project, newest first (ignores the jsonl)."""
    out = []
    d = _sessions_dir(project_path)
    if not d.is_dir():
        return out
    for meta_file in d.glob("*.json"):
        try:
            out.append(json.loads(meta_file.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda s: s.get("started", 0), reverse=True)
    return out


def list_recents() -> dict:
    """The payload for GET /api/recents: known projects (most-recent first), each
    with its session summaries read from disk."""
    index = sorted(_read_index(), key=lambda e: e.get("last_used", 0), reverse=True)
    projects = []
    for e in index:
        path = e.get("path", "")
        if not path or not Path(path).is_dir():
            continue                             # a moved/deleted project drops off the rail
        projects.append({
            "path": path,
            "name": e.get("name") or Path(path).name,
            "last_used": e.get("last_used", 0),
            "sessions": _session_summaries(path),
        })
    return {"projects": projects}


def load_session(project_path: str, session_id: str) -> dict:
    """The full record for GET /api/session: meta + the replayed event stream."""
    d = _sessions_dir(project_path)
    meta_file = d / f"{session_id}.json"
    events_file = d / f"{session_id}.jsonl"
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    events = []
    if events_file.is_file():
        for line in events_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    return {"meta": meta, "events": events}


# ---- per-run store (written as the run streams) ---------------------------

class SessionStore:
    """Writes one session's meta + append-only event log as the run streams. The
    server appends every OUTGOING envelope (the single choke point is the sender
    loop), so the persisted transcript is exactly what the UI saw."""

    def __init__(self, project_path: str, session_id: str, meta: dict):
        self.dir = _sessions_dir(project_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.dir / f"{session_id}.json"
        self.events_file = self.dir / f"{session_id}.jsonl"
        self.meta = {
            "id": session_id,
            "goal": meta.get("goal", ""),
            "mode": meta.get("mode", "autopilot"),
            "model": meta.get("model", ""),
            "started": int(time.time() * 1000),
            "finished": None,
            "result": None,
        }
        self._lock = threading.Lock()
        self._write_meta()

    def _write_meta(self) -> None:
        self.meta_file.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def append(self, env: dict) -> None:
        """Append one event envelope to the transcript log."""
        try:
            with self._lock, self.events_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(env) + "\n")
        except OSError:
            pass                                 # a transcript write must never kill the run

    def finalize(self, result: str) -> None:
        self.meta["result"] = result
        self.meta["finished"] = int(time.time() * 1000)
        try:
            self._write_meta()
        except OSError:
            pass
