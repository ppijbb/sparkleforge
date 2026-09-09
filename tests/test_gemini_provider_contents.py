"""#1530: Gemini calls must carry prior tool calls/results as real turns,
not drop them -- gemini-flash-lite re-issues identical tool calls when it
has no memory of having already called them."""

from src.core.llm_manager.providers import ProviderAdaptersMixin


def _adapter():
    return ProviderAdaptersMixin()


def test_tool_call_and_result_round_trip_as_function_turns():
    history = [
        {"role": "user", "content": "list the files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "function": {"name": "filesystem", "arguments": '{"path": "."}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "name": "filesystem", "content": '{"success": true, "files": ["a.py"]}'},
    ]
    contents = _adapter()._build_gemini_contents(history, "what next?", system_message=None)

    roles = [c["role"] for c in contents]
    # trailing "what next?" prompt merges into the preceding function_response
    # turn since both are role "user" -- Gemini requires strict alternation.
    assert roles == ["user", "model", "user"]

    function_call_turn = contents[1]
    assert function_call_turn["parts"][0]["function_call"] == {
        "name": "filesystem",
        "args": {"path": "."},
    }

    function_response_turn = contents[2]
    assert function_response_turn["parts"][0]["function_response"]["name"] == "filesystem"
    assert function_response_turn["parts"][0]["function_response"]["response"]["files"] == ["a.py"]
    assert function_response_turn["parts"][1] == {"text": "what next?"}


def test_consecutive_same_role_turns_are_merged():
    history = [
        {"role": "tool", "name": "a", "content": "{}"},
        {"role": "system", "content": "hurry up"},
    ]
    contents = _adapter()._build_gemini_contents(history, "final message", system_message=None)

    # tool result (user/function_response) + system nudge (user/text) +
    # trailing prompt (user/text) must collapse into one alternating turn,
    # not three consecutive "user" turns -- Gemini's API rejects non-alternating roles.
    assert [c["role"] for c in contents] == ["user"]
    assert len(contents[0]["parts"]) == 3


def test_system_message_prefixed_onto_first_turn_only():
    contents = _adapter()._build_gemini_contents([], "hello", system_message="be terse")
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0] == {"text": "be terse\n\n"}
    assert contents[0]["parts"][1] == {"text": "hello"}


def test_malformed_tool_arguments_do_not_raise():
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_0", "function": {"name": "x", "arguments": "not json"}}],
        }
    ]
    contents = _adapter()._build_gemini_contents(history, "next", system_message=None)
    assert contents[0]["parts"][0]["function_call"] == {"name": "x", "args": {}}


if __name__ == "__main__":
    test_tool_call_and_result_round_trip_as_function_turns()
    test_consecutive_same_role_turns_are_merged()
    test_system_message_prefixed_onto_first_turn_only()
    test_malformed_tool_arguments_do_not_raise()
    print("ok")
