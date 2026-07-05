"""Tests for config.py env-override layering.

Pins the bug where the env-override set was a hand-kept list that drifted from the
Config fields, silently leaving comfy_url/lms_command/comfy_timeout un-overridable
despite the documented "any field" promise."""

import pytest

from achilles.config import load_config, _coerce_env


def test_env_overrides_comfy_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("ACHILLES_COMFY_URL", "http://127.0.0.1:8188")
    monkeypatch.setenv("ACHILLES_LMS_COMMAND", "lms")
    monkeypatch.setenv("ACHILLES_COMFY_TIMEOUT", "42")   # int coercion
    cfg = load_config(str(tmp_path))
    assert cfg.comfy_url == "http://127.0.0.1:8188"
    assert cfg.lms_command == "lms"
    assert cfg.comfy_timeout == 42
    assert isinstance(cfg.comfy_timeout, int)


def test_env_overrides_bool_and_float(monkeypatch, tmp_path):
    monkeypatch.setenv("ACHILLES_USE_GIT", "false")
    monkeypatch.setenv("ACHILLES_TEMPERATURE", "0.7")
    cfg = load_config(str(tmp_path))
    assert cfg.use_git is False
    assert cfg.temperature == pytest.approx(0.7)


def test_env_bad_int_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("ACHILLES_COMFY_TIMEOUT", "notanint")
    with pytest.raises(ValueError, match="ACHILLES_COMFY_TIMEOUT"):
        load_config(str(tmp_path))


def test_legacy_native_tools_maps_to_act_protocol(tmp_path):
    # A pre-migration config used native_tools (bool); it must still be meaningful —
    # native_tools=false meant the text protocol.
    (tmp_path / "achilles.toml").write_text("native_tools = false\n", encoding="utf-8")
    cfg = load_config(str(tmp_path))
    assert cfg.act_protocol == "text"
    assert not hasattr(cfg, "native_tools")        # the field is gone, only the shim


def test_act_protocol_default_is_native(tmp_path):
    (tmp_path / "achilles.toml").write_text("", encoding="utf-8")
    assert load_config(str(tmp_path)).act_protocol == "native"


def test_coerce_env_bool_forms():
    assert _coerce_env(True, "no") is False
    assert _coerce_env(False, "ON") is True
    assert _coerce_env(True, "1") is True
    with pytest.raises(ValueError):
        _coerce_env(True, "maybe")
