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


def test_coerce_env_bool_forms():
    assert _coerce_env(True, "no") is False
    assert _coerce_env(False, "ON") is True
    assert _coerce_env(True, "1") is True
    with pytest.raises(ValueError):
        _coerce_env(True, "maybe")
