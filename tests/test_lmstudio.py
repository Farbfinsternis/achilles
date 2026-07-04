"""
loaded_llm() decides which model gets reloaded after an image render. If it
picks the wrong key — or the config placeholder "local-model" — the reload
fails with "Model not found" and Achilles is stranded brainless (the exact bug
this covers). These tests pin the `lms ps --json` parsing without a real CLI.
"""
import json
import types

import achilles.lmstudio as lms


def _cfg(model="google/gemma-4-12b", lms_command="lms"):
    return types.SimpleNamespace(model=model, lms_command=lms_command)


# A realistic `lms ps --json` entry (trimmed to the fields we read).
def _llm(model_key="google/gemma-4-12b", identifier=None):
    e = {"type": "llm", "modelKey": model_key}
    if identifier is not None:
        e["identifier"] = identifier
    return e


def _patch_ps(monkeypatch, output):
    # Intercept the CLI call: loaded_llm() shells out via _run("ps", "--json").
    def fake_run(lms_command, *args, timeout=300):
        assert args == ("ps", "--json")
        return output
    monkeypatch.setattr(lms, "_run", fake_run)


def test_returns_loaded_model_key(monkeypatch):
    _patch_ps(monkeypatch, json.dumps([_llm()]))
    assert lms.loaded_llm() == "google/gemma-4-12b"


def test_skips_embedding_models(monkeypatch):
    # An embedding model can be co-resident; we must reload the CHAT model, not it.
    entries = [{"type": "embedding", "modelKey": "nomic-embed"}, _llm()]
    _patch_ps(monkeypatch, json.dumps(entries))
    assert lms.loaded_llm() == "google/gemma-4-12b"


def test_falls_back_to_identifier_when_no_model_key(monkeypatch):
    entries = [{"type": "llm", "identifier": "custom-id"}]
    _patch_ps(monkeypatch, json.dumps(entries))
    assert lms.loaded_llm() == "custom-id"


def test_none_when_nothing_loaded(monkeypatch):
    _patch_ps(monkeypatch, "[]")
    assert lms.loaded_llm() is None


def test_none_on_cli_error(monkeypatch):
    # Detection is best-effort: a CLI failure must yield None so the caller can
    # fall back to config.model, never crash the render/reload swap.
    def boom(lms_command, *args, timeout=300):
        raise lms.LMStudioError("lms exploded")
    monkeypatch.setattr(lms, "_run", boom)
    assert lms.loaded_llm() is None


def test_none_on_garbage_json(monkeypatch):
    _patch_ps(monkeypatch, "not json at all")
    assert lms.loaded_llm() is None


# ---- remember / last_remembered (the last-model memory) -------------------

def _redirect_state(monkeypatch, tmp_path):
    monkeypatch.setattr(lms, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(lms, "_LAST_MODEL_FILE", tmp_path / "last_model")


def test_remember_roundtrip(monkeypatch, tmp_path):
    _redirect_state(monkeypatch, tmp_path)
    lms.remember_model("google/gemma-4-12b")
    assert lms.last_remembered() == "google/gemma-4-12b"


def test_remember_ignores_placeholder(monkeypatch, tmp_path):
    # "local-model" is the config placeholder — saving it would make a cold start
    # try `lms load local-model` and fail, the very bug we're fixing.
    _redirect_state(monkeypatch, tmp_path)
    lms.remember_model("local-model")
    assert lms.last_remembered() is None


def test_last_remembered_none_when_unset(monkeypatch, tmp_path):
    _redirect_state(monkeypatch, tmp_path)
    assert lms.last_remembered() is None


# ---- ensure_loaded (the startup guarantee) --------------------------------

def _patch_load_env(monkeypatch, tmp_path, *, loaded, available=True):
    """Wire up ensure_loaded's world: `lms` present or not, a given loaded model
    (None = nothing loaded), and a captured record of any `lms load` invocation."""
    _redirect_state(monkeypatch, tmp_path)
    monkeypatch.setattr(lms, "available", lambda cmd="lms": available)
    calls = {"loaded": []}

    def fake_run(lms_command, *args, timeout=300):
        if args[:1] == ("ps",):
            entries = [_llm(loaded)] if loaded else []
            return json.dumps(entries)
        if args[:1] == ("load",):
            calls["loaded"].append(args[1])
            return ""
        raise AssertionError(f"unexpected lms call: {args}")

    monkeypatch.setattr(lms, "_run", fake_run)
    return calls


def test_ensure_loaded_noop_when_model_present(monkeypatch, tmp_path):
    calls = _patch_load_env(monkeypatch, tmp_path, loaded="google/gemma-4-12b")
    lms.ensure_loaded(_cfg())
    assert calls["loaded"] == []                       # nothing (re)loaded
    assert lms.last_remembered() == "google/gemma-4-12b"  # but remembered


def test_ensure_loaded_restores_remembered(monkeypatch, tmp_path):
    calls = _patch_load_env(monkeypatch, tmp_path, loaded=None)
    lms.remember_model("google/gemma-4-12b")
    lms.ensure_loaded(_cfg(model="local-model"))       # config is a placeholder
    assert calls["loaded"] == ["google/gemma-4-12b"]   # loaded the remembered one


def test_ensure_loaded_falls_back_to_real_config_model(monkeypatch, tmp_path):
    calls = _patch_load_env(monkeypatch, tmp_path, loaded=None)
    lms.ensure_loaded(_cfg(model="qwen2.5-coder-7b-instruct"))
    assert calls["loaded"] == ["qwen2.5-coder-7b-instruct"]


def test_ensure_loaded_warns_when_nothing_to_load(monkeypatch, tmp_path):
    calls = _patch_load_env(monkeypatch, tmp_path, loaded=None)
    logs = []
    lms.ensure_loaded(_cfg(model="local-model"), log=logs.append)  # placeholder, none remembered
    assert calls["loaded"] == []
    assert any("no model loaded" in m for m in logs)


def test_ensure_loaded_noop_without_cli(monkeypatch, tmp_path):
    calls = _patch_load_env(monkeypatch, tmp_path, loaded=None, available=False)
    lms.ensure_loaded(_cfg())
    assert calls["loaded"] == []


# ---- config.model adoption (the request identifier, not just VRAM) --------

def test_ensure_loaded_adopts_loaded_key_over_placeholder(monkeypatch, tmp_path):
    # Loading only fills VRAM; the request still sends config.model. Newer LM Studio
    # rejects "local-model", so config.model must adopt the real loaded key.
    _patch_load_env(monkeypatch, tmp_path, loaded="google/gemma-4-12b")
    cfg = _cfg(model="local-model")
    lms.ensure_loaded(cfg)
    assert cfg.model == "google/gemma-4-12b"


def test_ensure_loaded_keeps_user_set_model_id(monkeypatch, tmp_path):
    # A real, user-chosen id is never overridden by whatever happens to be loaded.
    _patch_load_env(monkeypatch, tmp_path, loaded="google/gemma-4-12b")
    cfg = _cfg(model="qwen2.5-coder-7b-instruct")
    lms.ensure_loaded(cfg)
    assert cfg.model == "qwen2.5-coder-7b-instruct"


def test_ensure_loaded_adopts_key_after_restoring_remembered(monkeypatch, tmp_path):
    _patch_load_env(monkeypatch, tmp_path, loaded=None)
    lms.remember_model("google/gemma-4-12b")
    cfg = _cfg(model="local-model")
    lms.ensure_loaded(cfg)
    assert cfg.model == "google/gemma-4-12b"          # adopted the one it loaded
