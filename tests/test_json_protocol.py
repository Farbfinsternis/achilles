"""
Constrained content-JSON protocol (act_protocol="json"): the model emits ONE JSON
object per turn, grammar-enforced on the content channel via response_format=
json_schema — the channel LM Studio actually enforces (its tool-call arguments
channel is not). These tests pin the act schema, complete_json's parsing, and the
harness's json act loop: execute a call, stop on "finish", degrade to text if the
server rejects response_format, and fall back to act-parsing if it ignores it.
"""
import contextlib
import io
import json
import types

import pytest

from achilles import harness as H
from achilles import llm
from achilles.llm import complete_json, JsonReply, LLMError
from achilles.tools import Registry, BUILTINS, _manifest_tool


def _reg() -> Registry:
    r = Registry()
    for t in BUILTINS:
        r.register(t)
    return r


# ---- registry: the constrained act schema ---------------------------------

def test_content_json_schema_enum_and_fields():
    schema = _reg().content_json_schema()
    assert schema["required"] == ["tool"]
    assert schema["additionalProperties"] is False
    props = schema["properties"]
    enum = props["tool"]["enum"]
    assert {"read_file", "write_file", "list_dir", "run_command"} <= set(enum)
    assert "finish" in enum                     # the 'step done' sentinel
    assert "file_exists" not in enum            # check-only tools stay hidden
    # write_file's arguments surface as optional string fields.
    assert "path" in props and "content" in props
    assert props["content"]["type"] == "string"


def test_content_json_schema_includes_manifest_tool():
    r = _reg()
    r.register(_manifest_tool({"name": "deploy", "command": "deploy.sh {target} {body}"}))
    schema = r.content_json_schema()
    assert "deploy" in schema["properties"]["tool"]["enum"]
    assert "target" in schema["properties"]      # its {target} placeholder
    assert "body" in schema["properties"]        # its {body} freeform field


# ---- llm.complete_json -----------------------------------------------------

class _Cfg:
    base_url = "http://localhost:9/v1"
    api_key = "no-key"
    model = "m"
    request_timeout = 5
    max_tokens = 0


def _resp(body: dict):
    @contextlib.contextmanager
    def _cm(req, timeout=None):
        yield io.BytesIO(json.dumps(body).encode("utf-8"))
    return _cm


def test_complete_json_parses_object(monkeypatch):
    obj = {"tool": "write_file", "path": "a.py", "content": "x = 1"}
    body = {"choices": [{"message": {"content": json.dumps(obj)},
                         "finish_reason": "stop"}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _resp(body))
    jr = complete_json(_Cfg(), [{"role": "user", "content": "hi"}], schema={})
    assert jr.obj == obj
    assert jr.content == json.dumps(obj)


def test_complete_json_none_on_non_json(monkeypatch):
    body = {"choices": [{"message": {"content": "not json at all"},
                         "finish_reason": "stop"}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _resp(body))
    jr = complete_json(_Cfg(), [{"role": "user", "content": "hi"}], schema={})
    assert jr.obj is None
    assert jr.content == "not json at all"


def test_complete_json_length_guard(monkeypatch):
    body = {"choices": [{"message": {"content": '{"tool": "write_file", "cont'},
                         "finish_reason": "length"}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _resp(body))
    with pytest.raises(LLMError, match="truncated"):
        complete_json(_Cfg(), [{"role": "user", "content": "hi"}], schema={})


# ---- harness: the json act loop -------------------------------------------

def _cfg(tmp_path):
    return types.SimpleNamespace(
        workspace_path=tmp_path, act_protocol="json", tools=[], tools_dir="",
        comfy_url="", max_acts_per_step=6, temperature=0.2, max_tokens=0)


def test_json_executes_call_then_finishes(tmp_path, monkeypatch):
    h = H.Harness(_cfg(tmp_path), log=lambda *_: None)
    replies = iter([
        JsonReply(obj={"tool": "write_file", "path": "foo.txt", "content": "hi"},
                  content='{"tool":"write_file",...}'),
        JsonReply(obj={"tool": "finish"}, content='{"tool":"finish"}'),
    ])
    monkeypatch.setattr(H, "complete_json", lambda *a, **k: next(replies))
    ok = h._act_until_done([{"role": "system", "content": "s"},
                            {"role": "user", "content": "u"}])
    assert ok is True
    assert (tmp_path / "foo.txt").read_text(encoding="utf-8") == "hi"
    assert h._protocol == "json"                  # never had to degrade


def test_json_finish_first_nudges_once(tmp_path, monkeypatch):
    # A "finish" before any action looks like a lost step: nudge once, then accept.
    logs = []
    h = H.Harness(_cfg(tmp_path), log=logs.append)
    replies = iter([
        JsonReply(obj={"tool": "finish"}, content='{"tool":"finish"}'),
        JsonReply(obj={"tool": "finish"}, content='{"tool":"finish"}'),
    ])
    monkeypatch.setattr(H, "complete_json", lambda *a, **k: next(replies))
    ok = h._act_until_done([{"role": "system", "content": "s"},
                            {"role": "user", "content": "u"}])
    assert ok is True
    assert any("reminding the model" in m for m in logs)


def test_json_degrades_to_text_when_unsupported(tmp_path, monkeypatch):
    h = H.Harness(_cfg(tmp_path), log=lambda *_: None)

    def _reject(*a, **k):
        raise LLMError("response_format not supported")
    monkeypatch.setattr(H, "complete_json", _reject)
    monkeypatch.setattr(H, "chat", lambda *a, **k: "all done here")   # prose → done
    ok = h._act_until_done([{"role": "system", "content": "s"},
                            {"role": "user", "content": "u"}])
    assert ok is True
    assert h._protocol == "text"                  # degraded for the run


def test_json_falls_back_to_act_parse_when_ignored(tmp_path, monkeypatch):
    # Server accepted the request but ignored the schema, returning a fenced act
    # block as free text. We still parse and run it (obj is None), without degrading.
    h = H.Harness(_cfg(tmp_path), log=lambda *_: None)
    replies = iter([
        JsonReply(obj=None, content="```act\ntool: write_file\npath: a.txt\n---\nhi\n```"),
        JsonReply(obj={"tool": "finish"}, content='{"tool":"finish"}'),
    ])
    monkeypatch.setattr(H, "complete_json", lambda *a, **k: next(replies))
    ok = h._act_until_done([{"role": "system", "content": "s"},
                            {"role": "user", "content": "u"}])
    assert ok is True
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hi"
    assert h._protocol == "json"                  # obj=None does not degrade
