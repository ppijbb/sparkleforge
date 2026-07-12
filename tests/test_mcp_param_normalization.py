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


def test_get_mcp_hub_loads_config_lazily():
    import src.core.mcp_integration as mcp_integration
    import src.core.researcher_config as researcher_config

    # get_mcp_hub()'s `global _mcp_hub` binds to its own defining module
    # (src.core.mcp_integration.tools), not the mcp_integration package
    # __init__.py -- that's the actual singleton to reset here.
    previous_config = researcher_config.config
    previous_hub = mcp_integration.tools._mcp_hub
    researcher_config.config = None
    mcp_integration.tools._mcp_hub = None

    try:
        hub = mcp_integration.get_mcp_hub()

        assert hub is not None
        assert researcher_config.config is not None
    finally:
        mcp_integration.tools._mcp_hub = previous_hub
        researcher_config.config = previous_config
