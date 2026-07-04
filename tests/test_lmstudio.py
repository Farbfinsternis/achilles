"""
loaded_llm() decides which model gets reloaded after an image render. If it
picks the wrong key — or the config placeholder "local-model" — the reload
fails with "Model not found" and Achilles is stranded brainless (the exact bug
this covers). These tests pin the `lms ps --json` parsing without a real CLI.
"""
import json

import achilles.lmstudio as lms


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
