from coder_review_benchmark.tool_parser import parse_json_object, parse_tool_calls


def test_native_tool_call():
    result = parse_tool_calls({"choices": [{"message": {"content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}]}}]})
    assert result.error is None
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "a.py"}


def test_qwen25_text_fallback():
    result = parse_tool_calls({"choices": [{"message": {"content": '<tool_call>{"name":"list_files","arguments":{}}</tool_call>'}}]}, "qwen25_text_fallback")
    assert result.tool_calls[0].source == "qwen25_text_fallback"


def test_qwen25_flattened_json_fallback():
    response = {"choices": [{"message": {"content": '```json\n{"tool":"list_files","path":"."}\n```'}}]}
    result = parse_tool_calls(response, "qwen25_text_fallback")
    assert result.error is None
    assert result.tool_calls[0].name == "list_files"
    assert result.tool_calls[0].arguments == {"path": "."}
    assert result.tool_calls[0].source == "qwen25_text_fallback"


def test_qwen25_command_fallback_ignores_hallucinated_result():
    response = {
        "choices": [
            {
                "message": {
                    "content": "```\n> list_files\n\ncurrent_directory_files.txt\nexample.py\nreport.pdf\n```"
                }
            }
        ]
    }
    result = parse_tool_calls(response, "qwen25_text_fallback")
    assert result.error is None
    assert result.tool_calls[0].name == "list_files"
    assert result.tool_calls[0].arguments == {}
    assert result.tool_calls[0].source == "qwen25_text_fallback"


def test_qwen25_python_call_fallback():
    response = {"choices": [{"message": {"content": "list_files(path='.')"}}]}
    result = parse_tool_calls(response, "qwen25_text_fallback")
    assert result.error is None
    assert result.tool_calls[0].name == "list_files"
    assert result.tool_calls[0].arguments == {"path": "."}
    assert result.tool_calls[0].source == "qwen25_text_fallback"


def test_qwen25_python_call_with_leading_explanation():
    response = {
        "choices": [{"message": {"content": "Inspect the relevant directory.\n\nlist_files(path='src/plugin/duration')"}}]
    }
    result = parse_tool_calls(response, "qwen25_text_fallback")
    assert result.error is None
    assert result.tool_calls[0].name == "list_files"
    assert result.tool_calls[0].arguments == {"path": "src/plugin/duration"}


def test_invalid_text_is_reported():
    result = parse_tool_calls({"choices": [{"message": {"content": '<tool_call>{bad}</tool_call>'}}]}, "qwen25_text_fallback")
    assert result.error


def test_review_json():
    assert parse_json_object('```json\n{"decision":"reject"}\n```')["decision"] == "reject"
