from src.core.mcp_integration import _normalize_mcp_call_params


def test_normalize_wraps_fastmcp_single_input_schema():
    tool_def = {
        "inputSchema": {
            "type": "object",
            "properties": {"input": {"$ref": "#/$defs/SearchInput"}},
            "required": ["input"],
        }
    }

    assert _normalize_mcp_call_params(
        tool_def, {"query": "current", "max_results": 5}
    ) == {"input": {"query": "current", "num_results": 5}}


def test_normalize_keeps_flat_schema_flat():
    tool_def = {
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer"},
            },
            "required": ["query"],
        }
    }

    assert _normalize_mcp_call_params(
        tool_def, {"query": "current", "max_results": 5}
    ) == {"query": "current", "num_results": 5}


def test_normalize_supports_input_schema_attribute_name():
    class ToolDef:
        input_schema = {
            "type": "object",
            "properties": {"input": {"$ref": "#/$defs/SearchInput"}},
            "required": ["input"],
        }

    assert _normalize_mcp_call_params(
        ToolDef(), {"query": "current", "max_results": 5}
    ) == {"input": {"query": "current", "num_results": 5}}
