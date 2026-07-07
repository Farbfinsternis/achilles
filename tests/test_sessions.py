"""Tests for sessions.py — the web UI's server-side persistence.

The global project index and the per-project session records/transcripts are
exercised on a tmp HOME + tmp project dirs, so nothing touches the real
~/.achilles.
"""

from pathlib import Path

from achilles import sessions


def _isolate_home(monkeypatch, tmp_path):
    monkeypatch.setattr(sessions, "_home_index", lambda: tmp_path / "home" / "projects.json")


# ---- global project index -------------------------------------------------

def test_register_and_list_recents(tmp_path, monkeypatch):
    _isolate_home(monkeypatch, tmp_path)
    proj = tmp_path / "proj"; proj.mkdir()
    sessions.register_project(str(proj), "MyProj")
    rec = sessions.list_recents()
    assert len(rec["projects"]) == 1
    p = rec["projects"][0]
    assert p["name"] == "MyProj"
    assert Path(p["path"]) == proj.resolve()
    assert p["sessions"] == []


def test_register_is_idempotent_and_bumps_recency(tmp_path, monkeypatch):
    _isolate_home(monkeypatch, tmp_path)
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    sessions.register_project(str(a))
    sessions.register_project(str(b))
    sessions.register_project(str(a))                # re-register a → now most recent
    order = [p["name"] for p in sessions.list_recents()["projects"]]
    assert order == ["a", "b"]                       # most-recently-used first
    # only two entries despite three registrations
    assert len(sessions.list_recents()["projects"]) == 2


# ---- per-project session store --------------------------------------------

def test_session_store_roundtrip(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    st = sessions.SessionStore(str(proj), "run-1",
                               {"goal": "g", "mode": "interview", "model": "m"})
    st.append({"type": "log", "data": {"text": "hi"}})
    st.append({"type": "run.finished", "data": {"result": "success"}})
    st.finalize("success")

    rec = sessions.load_session(str(proj), "run-1")
    assert rec["meta"]["goal"] == "g" and rec["meta"]["mode"] == "interview"
    assert rec["meta"]["result"] == "success" and rec["meta"]["finished"] is not None
    assert [e["type"] for e in rec["events"]] == ["log", "run.finished"]


def test_recents_includes_session_summaries_newest_first(tmp_path, monkeypatch):
    _isolate_home(monkeypatch, tmp_path)
    proj = tmp_path / "proj"; proj.mkdir()
    sessions.register_project(str(proj))
    sessions.SessionStore(str(proj), "run-1", {"goal": "first", "mode": "autopilot"})
    sessions.SessionStore(str(proj), "run-2", {"goal": "second", "mode": "autopilot"})
    goals = [s["goal"] for s in sessions.list_recents()["projects"][0]["sessions"]]
    assert goals == ["second", "first"]              # newest first


def test_load_missing_session_returns_empty(tmp_path):
    assert sessions.load_session(str(tmp_path), "nope") == {}
