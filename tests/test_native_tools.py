"""
Native (OpenAI) tool-calling: Achilles offers the tools as a JSON schema and reads
structured tool_calls back, with the text `act` protocol as the fallback. These
tests pin the schema export, the tool_call → ToolCall mapping, complete_act's
parsing, and the harness's native-then-fallback act loop.
"""
import contextlib
import io
import json
import types

import pytest

from achilles import harness as H
from achilles import llm
from achilles.llm import complete_act, ActReply, LLMError
from achilles.tools import Registry, BUILTINS, ToolContext, _manifest_tool


def _reg() -> Registry:
    r = Registry()
    for t in BUILTINS:
        r.register(t)
    return r


# ---- registry: schema export + native-call mapping ------------------------

def test_tool_schemas_expose_act_tools_only():
    schemas = _reg().tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert {"read_file", "write_file", "list_dir", "run_command"} <= names
    assert "file_exists" not in names          # check-only tools are hidden
    for s in schemas:
        assert s["type"] == "function"
        assert s["function"]["parameters"]["type"] == "object"


def test_write_file_schema_requires_path_and_content():
    schemas = {s["function"]["name"]: s for s in _reg().tool_schemas()}
    params = schemas["write_file"]["function"]["parameters"]
    assert set(params["required"]) == {"path", "content"}


def test_build_call_routes_body_param_to_body():
    call = _reg().build_call("write_file", {"path": "a.py", "content": "x = 1"})
    assert call.name == "write_file"
    assert call.args == {"path": "a.py"}        # content pulled out
    assert call.body == "x = 1"


def test_build_call_plain_tool_has_no_body():
    call = _reg().build_call("read_file", {"path": "a.py"})
    assert call.args == {"path": "a.py"}
    assert call.body is None


def test_build_call_stringifies_nonstring_body():
    call = _reg().build_call("write_file", {"path": "a.py", "content": 123})
    assert call.body == "123"


def test_native_write_roundtrip(tmp_path):
    r = _reg()
    call = r.build_call("write_file", {"path": "out.txt", "content": "hello"})
    res = r.dispatch(call, ToolContext(tmp_path))
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hello"
    assert res.startswith("OK")


def test_describe_omits_usage_in_native_mode():
    d = _reg().describe(include_usage=False)
    assert "```act" not in d
    assert "read_file" in d          # names/descriptions still present


def test_manifest_tool_schema_from_placeholders():
    t = _manifest_tool({"name": "deploy", "command": "deploy.sh {target} {body}"})
    props = t.parameters["properties"]
    assert "target" in props and "body" in props
    assert t.body_param == "body"
    assert t.parameters["required"] == ["target"]   # {body} is not a required header


# ---- llm.complete_act -----------------------------------------------------

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


def test_complete_act_parses_tool_calls(monkeypatch):
    body = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {
            "name": "write_file",
            "arguments": json.dumps({"path": "a.py", "content": "x = 1"})}}]},
        "finish_reason": "tool_calls"}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _resp(body))
    reply = complete_act(_Cfg(), [{"role": "user", "content": "hi"}], tools=[])
    assert len(reply.tool_calls) == 1
    tc = reply.tool_calls[0]
    assert tc["id"] == "c1" and tc["name"] == "write_file"
    assert tc["arguments"] == {"path": "a.py", "content": "x = 1"}


def test_complete_act_text_reply_has_no_calls(monkeypatch):
    body = {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _resp(body))
    reply = complete_act(_Cfg(), [{"role": "user", "content": "hi"}], tools=[])
    assert reply.tool_calls == [] and reply.content == "done"


def test_complete_act_tolerates_bad_arguments_json(monkeypatch):
    body = {"choices": [{"message": {"tool_calls": [
        {"id": "c1", "function": {"name": "read_file", "arguments": "{not json"}}]},
        "finish_reason": "tool_calls"}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _resp(body))
    reply = complete_act(_Cfg(), [{"role": "user", "content": "hi"}], tools=[])
    assert reply.tool_calls[0]["arguments"] == {}


def test_complete_act_length_guard_only_without_calls(monkeypatch):
    body = {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _resp(body))
    with pytest.raises(LLMError, match="truncated"):
        complete_act(_Cfg(), [{"role": "user", "content": "hi"}], tools=[])


def test_complete_act_length_with_calls_is_ok(monkeypatch):
    # A complete set of tool_calls + finish_reason 'length' is not treated as an
    # unsafe fragment (the structured args are self-delimiting).
    body = {"choices": [{"message": {"tool_calls": [
        {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}]},
        "finish_reason": "length"}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", _resp(body))
    reply = complete_act(_Cfg(), [{"role": "user", "content": "hi"}], tools=[])
    assert reply.tool_calls


# ---- harness: native act loop + fallback ----------------------------------

def _harness_cfg(tmp_path):
    return types.SimpleNamespace(
        workspace_path=tmp_path, native_tools=True, tools=[], tools_dir="",
        comfy_url="", max_acts_per_step=6, temperature=0.2, max_tokens=0)


def test_harness_native_executes_tool_calls(tmp_path, monkeypatch):
    h = H.Harness(_harness_cfg(tmp_path), log=lambda *_: None)
    replies = iter([
        ActReply(content="", tool_calls=[
            {"id": "c1", "name": "write_file",
             "arguments": {"path": "foo.txt", "content": "hi"}}]),
        ActReply(content="done", tool_calls=[]),   # then it stops → step done
    ])
    monkeypatch.setattr(H, "complete_act", lambda *a, **k: next(replies))
    ok = h._act_until_done([{"role": "system", "content": "s"},
                            {"role": "user", "content": "u"}])
    assert ok is True
    assert (tmp_path / "foo.txt").read_text(encoding="utf-8") == "hi"
    assert h._native_tools is True                 # never had to fall back


def test_harness_falls_back_to_text_protocol(tmp_path, monkeypatch):
    h = H.Harness(_harness_cfg(tmp_path), log=lambda *_: None)

    def _reject_tools(*a, **k):
        raise LLMError("this server does not support tools")
    monkeypatch.setattr(H, "complete_act", _reject_tools)
    monkeypatch.setattr(H, "chat", lambda *a, **k: "all done here")   # prose → step done
    ok = h._act_until_done([{"role": "system", "content": "s"},
                            {"role": "user", "content": "u"}])
    assert ok is True
    assert h._native_tools is False                # flipped off for the run


def test_work_prompt_pins_dod_paths(tmp_path):
    # The executor is told the EXACT paths the Definition of Done checks, so it
    # can't invent styles.css when the contract says style.css.
    h = H.Harness(_harness_cfg(tmp_path), log=lambda *_: None)
    h._expected_paths = ["assets/hero.jpg", "style.css"]
    plan = [{"done": False, "text": "make the page"}]
    prompt = h._work_prompt("build it", plan, last_verify=None)
    assert "assets/hero.jpg" in prompt and "style.css" in prompt
    assert "exact path" in prompt.lower()


def test_work_prompt_omits_block_when_no_paths(tmp_path):
    h = H.Harness(_harness_cfg(tmp_path), log=lambda *_: None)
    h._expected_paths = []
    prompt = h._work_prompt("build it", [{"done": False, "text": "x"}], last_verify=None)
    assert "Required file paths" not in prompt


def test_log_result_prints_receipts(tmp_path):
    # The user must SEE what each tool call did, not just that one happened.
    from achilles.protocol import ToolCall
    logs = []
    h = H.Harness(_harness_cfg(tmp_path), log=logs.append)
    h._log_result(ToolCall("write_file", {"path": "a.py"}), "OK: wrote a.py (3 lines)")
    h._log_result(ToolCall("read_file", {"path": "b.py"}), "l1\nl2\nl3")
    h._log_result(ToolCall("run_command", {"command": "pytest"}), "exit=1\nboom")
    joined = "\n".join(logs)
    assert "OK: wrote a.py (3 lines)" in joined      # write shows its status line
    assert "read b.py (3 lines)" in joined            # read is summarised, not dumped
    assert "exit=1" in joined                          # a failure is surfaced


def test_native_execution_logs_a_receipt(tmp_path, monkeypatch):
    # End-to-end: the receipt appears on the real native act path, after dispatch.
    logs = []
    h = H.Harness(_harness_cfg(tmp_path), log=logs.append)
    replies = iter([
        ActReply(content="", tool_calls=[
            {"id": "c1", "name": "write_file",
             "arguments": {"path": "foo.txt", "content": "hi"}}]),
        ActReply(content="done", tool_calls=[]),
    ])
    monkeypatch.setattr(H, "complete_act", lambda *a, **k: next(replies))
    h._act_until_done([{"role": "system", "content": "s"},
                       {"role": "user", "content": "u"}])
    assert any("wrote foo.txt" in m for m in logs)


def test_system_prompt_permits_cdn_and_google_fonts(tmp_path):
    # The CDN/font permission must reach BOTH protocol variants — a web-capable
    # model shouldn't be stuck vendoring Tailwind or a font it could just link.
    h = H.Harness(_harness_cfg(tmp_path), log=lambda *_: None)
    for native in (True, False):
        h._native_tools = native
        prompt = h._system_prompt()
        assert "cdn.tailwindcss.com" in prompt
        assert "fonts.googleapis.com" in prompt
