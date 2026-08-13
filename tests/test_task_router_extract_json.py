r"""Issue #1144: TaskRouter._extract_json used a non-greedy regex (r"\{.*?\}")
that stopped at the first closing brace, truncating any LLM routing response
whose JSON contained a nested object (e.g. structured fallback_agents
entries). It should find the first *complete* JSON object instead.
"""

from src.core.task_router import TaskRouter


def test_extract_json_handles_nested_braces():
    router = TaskRouter()
    text = (
        '{"route": "codebase_agent", '
        '"fallback_agents": [{"name": "a", "score": 0.9}, {"name": "b"}]}'
    )

    result = router._extract_json(text)

    assert result == {
        "route": "codebase_agent",
        "fallback_agents": [{"name": "a", "score": 0.9}, {"name": "b"}],
    }


def test_extract_json_strips_markdown_code_fence():
    router = TaskRouter()
    text = '```json\n{"route": "quantum_solver"}\n```'

    assert router._extract_json(text) == {"route": "quantum_solver"}


def test_extract_json_ignores_braces_inside_string_values():
    router = TaskRouter()
    text = '{"route": "codebase_agent", "note": "uses { and } in prose"}'

    assert router._extract_json(text) == {
        "route": "codebase_agent",
        "note": "uses { and } in prose",
    }


def test_extract_json_returns_empty_dict_for_no_json():
    router = TaskRouter()

    assert router._extract_json("no json here") == {}
