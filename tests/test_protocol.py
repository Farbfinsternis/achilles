"""Tests for protocol.py — the text tool-call format the harness parses.

This is the single most fragile seam in Achilles (a weak model's format
compliance), and historically the source of silent work-loss bugs. These tests
pin the parser's behaviour so a fix elsewhere can't quietly regress it.
"""

from achilles.protocol import parse_tool_call, ToolCall


def test_basic_write_file_with_body():
    text = (
        "```act\n"
        "tool: write_file\n"
        "path: src/foo.py\n"
        "---\n"
        "def foo():\n"
        "    return 42\n"
        "```"
    )
    call = parse_tool_call(text)
    assert call is not None
    assert call.name == "write_file"
    assert call.args["path"] == "src/foo.py"
    assert call.body == "def foo():\n    return 42"


def test_headers_only_tool_has_no_body():
    call = parse_tool_call("```act\ntool: read_file\npath: a.txt\n```")
    assert call is not None
    assert call.name == "read_file"
    assert call.body is None


def test_name_alias_for_tool():
    call = parse_tool_call("```act\nname: list_dir\npath: .\n```")
    assert call is not None
    assert call.name == "list_dir"


def test_tool_and_action_fence_tags_accepted():
    for tag in ("act", "tool", "action"):
        call = parse_tool_call(f"```{tag}\ntool: read_file\npath: a.txt\n```")
        assert call is not None, tag
        assert call.name == "read_file"


def test_prose_without_fence_is_none():
    assert parse_tool_call("The step is complete. All tests pass.") is None


def test_empty_and_none_input():
    assert parse_tool_call("") is None
    assert parse_tool_call(None) is None


def test_missing_separator_treated_as_early_body():
    # Model forgot the `---`; the first non-header line starts the body.
    text = "```act\ntool: write_file\npath: a.txt\nhello world\n```"
    call = parse_tool_call(text)
    assert call is not None
    assert call.name == "write_file"
    assert call.body == "hello world"


def test_block_missing_tool_name_is_none():
    assert parse_tool_call("```act\npath: a.txt\n---\nbody\n```") is None


def test_only_first_block_is_honoured():
    text = (
        "```act\ntool: read_file\npath: first.txt\n```\n"
        "```act\ntool: read_file\npath: second.txt\n```"
    )
    call = parse_tool_call(text)
    assert call.args["path"] == "first.txt"
