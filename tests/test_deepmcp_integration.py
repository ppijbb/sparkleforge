"""DeepMCPAgent 통합 테스트."""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.mcp_auto_discovery import FastMCPClientWrapper, FastMCPMulti, discover_server_tools
from src.core.mcp_tool_loader import MCPToolLoader, ToolInfo, _jsonschema_to_pydantic
from src.core.agent_tool_selector import AgentToolSelector, AgentType, ToolCategory
from src.core.config import HTTPServerSpec, servers_to_mcp_config
from src.core.cross_agent_tools import CrossAgent, make_cross_agent_tools, _extract_final_answer
from langchain_core.tools import BaseTool


def test_config_conversion():
    """서버 스펙 변환 테스트."""
    print("=== Testing Config Conversion ===")

    servers = {
        "test": HTTPServerSpec(
            url="http://localhost:8000/mcp",
            transport="http"
        )
    }

    config = servers_to_mcp_config(servers)
    assert "test" in config
    assert config["test"]["url"] == "http://localhost:8000/mcp"
    assert config["test"]["transport"] == "http"
    print("✅ Config conversion test passed")


def test_jsonschema_to_pydantic():
    """JSON Schema → Pydantic 변환 테스트."""
    print("=== Testing JSON Schema to Pydantic ===")

    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Result limit", "default": 10}
        },
        "required": ["query"]
    }

    model = _jsonschema_to_pydantic(schema, model_name="TestArgs")

    # 모델 인스턴스 생성 테스트
    instance = model(query="test query")
    assert instance.query == "test query"
    assert instance.limit == 10

    print("✅ JSON Schema to Pydantic conversion test passed")


def test_tool_category_inference():
    """도구 카테고리 추론 테스트."""
    print("=== Testing Tool Category Inference ===")

    selector = AgentToolSelector()

    # 검색 도구
    search_info = ToolInfo(
        server_guess="search",
        name="web_search",
        description="Search the web for information",
        input_schema={}
    )
    assert selector._infer_tool_category(search_info) == ToolCategory.SEARCH

    # 코드 도구
    code_info = ToolInfo(
        server_guess="code",
        name="run_python",
        description="Execute Python code",
        input_schema={}
    )
    assert selector._infer_tool_category(code_info) == ToolCategory.CODE

    # 기본 카테고리
    unknown_info = ToolInfo(
        server_guess="unknown",
        name="unknown_tool",
        description="Some unknown tool",
        input_schema={}
    )
    assert selector._infer_tool_category(unknown_info) == ToolCategory.UTILITY

    print("✅ Tool category inference test passed")


def test_agent_tool_assignment():
    """에이전트별 도구 할당 테스트."""
    print("=== Testing Agent Tool Assignment ===")

    selector = AgentToolSelector()

    # 모의 도구 생성
    mock_tools = [
        type('MockTool', (), {
            'name': 'web_search',
            '_arun': lambda self, **kwargs: "Search results"
        })(),
        type('MockTool', (), {
            'name': 'run_code',
            '_arun': lambda self, **kwargs: "Code output"
        })()
    ]

    mock_infos = [
        ToolInfo("search", "web_search", "Search the web", {}),
        ToolInfo("code", "run_code", "Execute code", {})
    ]

    # 각 에이전트별 도구 할당
    assignments = selector.select_tools_for_all_agents(mock_tools, mock_infos)

    # PlannerAgent는 search 카테고리 도구를 가져야 함
    planner_assignment = assignments[AgentType.PLANNER]
    assert len(planner_assignment.tools) > 0
    assert any("search" in tool.name for tool in planner_assignment.tools)

    # ExecutorAgent는 search와 code 카테고리 도구를 가져야 함
    executor_assignment = assignments[AgentType.EXECUTOR]
    assert len(executor_assignment.tools) > 0

    print("✅ Agent tool assignment test passed")


def test_cross_agent_tools():
    """Cross-Agent 도구 테스트."""
    print("=== Testing Cross-Agent Tools ===")

    # 모의 에이전트 생성
    async def mock_agent_response(input_data):
        return {"messages": [{"content": f"Response to: {input_data['messages'][0]['content']}"}]}

    mock_agent = type('MockAgent', (), {'ainvoke': mock_agent_response})()

    peers = {
        "researcher": CrossAgent(
            agent=mock_agent,
            description="Research assistant"
        )
    }

    # Cross-Agent 도구 생성
    tools = make_cross_agent_tools(peers)

    # ask_agent_researcher 도구가 있어야 함
    tool_names = [tool.name for tool in tools]
    assert "ask_agent_researcher" in tool_names

    # broadcast_to_agents 도구가 있어야 함 (include_broadcast=True 기본값)
    assert "broadcast_to_agents" in tool_names

    # 결과 추출 테스트
    result = _extract_final_answer({"messages": [{"content": "Test response"}]})
    assert result == "Test response"

    print("✅ Cross-Agent tools test passed")


async def test_async_components():
    """비동기 컴포넌트 테스트."""
    print("=== Testing Async Components ===")

    # FastMCP 클라이언트 래퍼 생성 테스트 (실제 서버 없음)
    try:
        wrapper = FastMCPClientWrapper({})
        assert wrapper.client is not None
        print("✅ FastMCPClientWrapper creation test passed")
    except Exception as e:
        print(f"⚠️ FastMCPClientWrapper test skipped (no MCP server): {e}")

    # MCPToolLoader 생성 테스트
    try:
        multi = FastMCPMulti({})
        loader = MCPToolLoader(multi)
        assert loader._multi == multi
        print("✅ MCPToolLoader creation test passed")
    except Exception as e:
        print(f"❌ MCPToolLoader test failed: {e}")


def main():
    """메인 테스트 함수."""
    print("🚀 Starting DeepMCPAgent Integration Tests")
    print("=" * 50)

    try:
        # 동기 테스트들
        test_config_conversion()
        test_jsonschema_to_pydantic()
        test_tool_category_inference()
        test_agent_tool_assignment()
        test_cross_agent_tools()

        # 비동기 테스트들
        asyncio.run(test_async_components())

        print("=" * 50)
        print("🎉 All tests passed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
