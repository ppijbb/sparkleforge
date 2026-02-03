"""
Production Upgrade Tests

이 테스트는 production-level upgrade의 모든 변경사항을 검증합니다:
1. Fallback 코드 제거 확인
2. MCP-only 아키텍처 검증
3. OpenRouter + Gemini 2.5 Flash Lite 통합 확인
4. Health check 강화 검증
5. API direct call 제거 확인
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import patch, AsyncMock, MagicMock
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.llm_manager import MultiModelOrchestrator, TaskType
from src.research.tools.web_search import WebSearchTool
from src.research.tools.academic_search import AcademicSearchTool
from src.research.workflow_manager import ResearchWorkflowManager
from src.automation.browser_manager import BrowserManager
from src.core.mcp_integration import UniversalMCPHub, health_check
from src.core.researcher_config import ResearcherSystemConfig


class TestProductionUpgrade:
    """Production upgrade 검증 테스트."""
    
    @pytest.fixture
    def mock_env_vars(self):
        """환경 변수 모킹."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test_key',
            'LLM_PROVIDER': 'openrouter',
            'LLM_MODEL': 'google/gemini-2.5-flash-lite',
            'MCP_ENABLED': 'true',
            'ENABLE_AUTO_FALLBACK': 'false'
        }):
            yield
    
    @pytest.fixture
    def llm_orchestrator(self, mock_env_vars):
        """LLM 오케스트레이터 인스턴스."""
        return MultiModelOrchestrator()
    
    @pytest.fixture
    def web_search_tool(self, mock_env_vars):
        """웹 검색 도구 인스턴스."""
        return WebSearchTool({
            'name': 'test_web_search',
            'enabled': True,
            'timeout': 30,
            'max_results': 10
        })
    
    @pytest.fixture
    def academic_search_tool(self, mock_env_vars):
        """학술 검색 도구 인스턴스."""
        return AcademicSearchTool({
            'name': 'test_academic_search',
            'enabled': True,
            'timeout': 30,
            'max_results': 10
        })
    
    @pytest.fixture
    def browser_manager(self, mock_env_vars):
        """브라우저 매니저 인스턴스."""
        return BrowserManager()
    
    @pytest.fixture
    def mcp_hub(self, mock_env_vars):
        """MCP 허브 인스턴스."""
        return UniversalMCPHub()
    
    def test_no_fallback_models_loaded(self, llm_orchestrator):
        """Fallback 모델이 로드되지 않았는지 확인."""
        # Fallback 모델들이 로드되지 않았는지 확인
        fallback_models = ['qwen2.5-vl-72b', 'deepseek-chat-v3', 'llama-3.3-70b']
        for model in fallback_models:
            assert model not in llm_orchestrator.models, f"Fallback model {model} should not be loaded"
        
        # OpenRouter 모델만 로드되었는지 확인
        openrouter_models = [name for name, config in llm_orchestrator.models.items() 
                           if config.provider == 'openrouter']
        assert len(openrouter_models) > 0, "No OpenRouter models loaded"
    
    def test_mcp_only_architecture(self, web_search_tool, academic_search_tool):
        """MCP-only 아키텍처 확인."""
        # 웹 검색 도구가 MCP만 사용하는지 확인
        assert hasattr(web_search_tool, 'execute_with_mcp'), "WebSearchTool should have execute_with_mcp method"
        assert not hasattr(web_search_tool, 'tavily_api_key'), "WebSearchTool should not have API keys"
        assert not hasattr(web_search_tool, 'exa_api_key'), "WebSearchTool should not have API keys"
        assert not hasattr(web_search_tool, 'google_api_key'), "WebSearchTool should not have API keys"
        
        # 학술 검색 도구가 MCP만 사용하는지 확인
        assert hasattr(academic_search_tool, 'execute_with_mcp'), "AcademicSearchTool should have execute_with_mcp method"
        assert academic_search_tool.enable_mcp_priority, "AcademicSearchTool should prioritize MCP"
    
    def test_openrouter_gemini_integration(self, llm_orchestrator):
        """OpenRouter + Gemini 2.5 Flash Lite 통합 확인."""
        # Gemini 모델들이 로드되었는지 확인
        gemini_models = [name for name, config in llm_orchestrator.models.items() 
                        if config.provider == 'google' and 'gemini' in config.model_id]
        assert len(gemini_models) > 0, "No Gemini models loaded"
        
        # Gemini 2.5 Flash Lite가 기본 모델인지 확인
        assert 'gemini-2.5-flash-lite' in [config.model_id for config in llm_orchestrator.models.values()], \
            "Gemini 2.5 Flash Lite should be loaded"
    
    @pytest.mark.asyncio
    async def test_enhanced_health_check(self, mcp_hub):
        """강화된 헬스 체크 테스트."""
        # MCP 허브 초기화 모킹
        with patch.object(mcp_hub, 'openrouter_client') as mock_client:
            mock_client.generate_response = AsyncMock(return_value={'choices': [{'message': {'content': 'OK'}}]})
            
            # execute_tool 모킹
            with patch('mcp_integration.execute_tool') as mock_execute_tool:
                mock_execute_tool.return_value = {'success': True, 'data': {'results': []}}
                
                health_status = await mcp_hub.health_check()
                
                # 헬스 체크 결과 검증
                assert 'overall_health' in health_status, "Health check should include overall_health"
                assert 'tool_health' in health_status, "Health check should include tool_health"
                assert 'essential_tools_healthy' in health_status, "Health check should include essential_tools_healthy"
                assert 'critical_error' in health_status or health_status['overall_health'] == 'healthy', \
                    "Health check should include critical_error or be healthy"
    
    @pytest.mark.asyncio
    async def test_essential_tools_validation(self, mcp_hub):
        """필수 도구 검증 테스트."""
        with patch('mcp_integration.execute_tool') as mock_execute_tool:
            # 성공적인 도구 실행 모킹
            mock_execute_tool.return_value = {'success': True, 'data': {'results': []}}
            
            # 필수 도구 검증이 성공하는지 확인
            try:
                await mcp_hub._validate_essential_tools()
                # 예외가 발생하지 않으면 성공
                assert True
            except RuntimeError as e:
                pytest.fail(f"Essential tools validation should not fail: {e}")
    
    @pytest.mark.asyncio
    async def test_essential_tools_validation_failure(self, mcp_hub):
        """필수 도구 검증 실패 테스트."""
        with patch('mcp_integration.execute_tool') as mock_execute_tool:
            # 실패하는 도구 실행 모킹
            mock_execute_tool.return_value = {'success': False, 'error': 'Tool failed'}
            
            # 필수 도구 검증이 실패하는지 확인
            with pytest.raises(RuntimeError, match="Essential MCP tools failed validation"):
                await mcp_hub._validate_essential_tools()
    
    def test_no_direct_api_calls(self):
        """직접 API 호출이 제거되었는지 확인."""
        # 웹 검색 도구에서 직접 API 호출 제거 확인
        web_search_file = os.path.join(project_root, 'src', 'research', 'tools', 'web_search.py')
        with open(web_search_file, 'r') as f:
            content = f.read()
            assert 'requests.get' not in content, "WebSearchTool should not use direct requests.get"
            assert 'requests.post' not in content, "WebSearchTool should not use direct requests.post"
            assert 'aiohttp.ClientSession' not in content, "WebSearchTool should not use direct aiohttp"
        
        # 브라우저 매니저에서 직접 API 호출 제거 확인
        browser_file = os.path.join(project_root, 'src', 'automation', 'browser_manager.py')
        with open(browser_file, 'r') as f:
            content = f.read()
            assert 'requests.get' not in content, "BrowserManager should not use direct requests.get"
            assert 'requests.post' not in content, "BrowserManager should not use direct requests.post"
    
    def test_workflow_manager_mcp_integration(self):
        """워크플로우 매니저 MCP 통합 확인."""
        workflow_file = os.path.join(project_root, 'src', 'research', 'workflow_manager.py')
        with open(workflow_file, 'r') as f:
            content = f.read()
            assert 'execute_tool' in content, "WorkflowManager should use MCP tools"
            assert 'requests.get' not in content, "WorkflowManager should not use direct requests"
    
    def test_research_agent_mcp_integration(self):
        """연구 에이전트 MCP 통합 확인."""
        agent_file = os.path.join(project_root, 'src', 'agents', 'research_agent.py')
        with open(agent_file, 'r') as f:
            content = f.read()
            assert 'execute_tool' in content, "ResearchAgent should use MCP tools"
            # 직접 API 호출이 MCP 도구로 대체되었는지 확인
            assert 'from src.core.mcp_integration import execute_tool' in content, \
                "ResearchAgent should import execute_tool from MCP integration"
    
    def test_configuration_updates(self):
        """설정 파일 업데이트 확인."""
        # env.example 파일 확인
        env_file = os.path.join(project_root, 'env.example')
        with open(env_file, 'r') as f:
            content = f.read()
            assert 'ENABLE_AUTO_FALLBACK=false' in content, "env.example should disable auto fallback"
            assert 'OPENROUTER_API_KEY' in content, "env.example should include OpenRouter API key"
            assert 'google/gemini-2.5-flash-lite' in content, "env.example should use Gemini 2.5 Flash Lite"
        
        # run.sh 파일 확인
        run_file = os.path.join(project_root, 'run.sh')
        with open(run_file, 'r') as f:
            content = f.read()
            assert 'ENABLE_AUTO_FALLBACK=false' in content, "run.sh should disable auto fallback"
            assert 'MCP Only - No Fallbacks' in content, "run.sh should indicate MCP-only approach"
    
    def test_base_tool_mcp_integration(self):
        """베이스 도구 MCP 통합 확인."""
        base_tool_file = os.path.join(project_root, 'src', 'research', 'tools', 'base_tool.py')
        with open(base_tool_file, 'r') as f:
            content = f.read()
            assert 'execute_with_mcp' in content, "BaseTool should have execute_with_mcp method"
            assert 'alternative_tools' in content, "BaseTool should support alternative tools"
            # 직접 HTTP 요청이 MCP 도구로 대체되었는지 확인
            assert 'execute_tool("fetch"' in content, "BaseTool should use MCP fetch tool"
    
    @pytest.mark.asyncio
    async def test_mcp_tool_execution(self, web_search_tool):
        """MCP 도구 실행 테스트."""
        with patch('src.research.tools.base_tool.execute_tool') as mock_execute_tool:
            # 성공적인 MCP 도구 실행 모킹 - ToolResult 객체 반환
            from src.research.tools.base_tool import ToolResult
            mock_execute_tool.return_value = ToolResult(
                success=True,
                data={
                    'results': [
                        {
                            'title': 'Test Result',
                            'url': 'https://example.com',
                            'content': 'Test content'
                        }
                    ],
                    'provider': 'mcp_gsearch'
                },
                execution_time=0.5,
                confidence=0.9
            )
            
            # 웹 검색 실행
            results = await web_search_tool.search("test query")
            
            # 결과 검증
            assert len(results) > 0, "Search should return results"
            assert results[0].title == 'Test Result', "Result should have correct title"
            assert results[0].url == 'https://example.com', "Result should have correct URL"
            assert results[0].mcp_tool_used is not None, "Result should indicate MCP tool used"
    
    @pytest.mark.asyncio
    async def test_mcp_tool_failure_handling(self, web_search_tool):
        """MCP 도구 실패 처리 테스트."""
        with patch('src.research.tools.base_tool.execute_tool') as mock_execute_tool:
            # 실패하는 MCP 도구 실행 모킹 - ToolResult 객체 반환
            from src.research.tools.base_tool import ToolResult
            mock_execute_tool.return_value = ToolResult(
                success=False,
                error='MCP tool failed'
            )
            
            # 웹 검색 실행 시 예외 발생 확인
            with pytest.raises(RuntimeError, match="All MCP tools failed"):
                await web_search_tool.search("test query")
    
    def test_production_grade_reliability(self, web_search_tool):
        """Production-grade 신뢰성 기능 확인."""
        # Circuit breaker 기능 확인
        assert hasattr(web_search_tool, 'performance'), "Tool should have performance tracking"
        assert hasattr(web_search_tool.performance, 'circuit_breaker_status'), \
            "Performance should include circuit breaker status"
        
        # 신뢰성 설정 확인
        assert hasattr(web_search_tool, 'reliability_config'), "Tool should have reliability config"
        assert web_search_tool.reliability_config.enable_circuit_breaker, \
            "Circuit breaker should be enabled"
    
    def test_structured_logging(self):
        """구조화된 로깅 확인."""
        # 로깅 설정 파일 확인
        logging_file = os.path.join(project_root, 'src', 'core', 'logging_config.py')
        if os.path.exists(logging_file):
            with open(logging_file, 'r') as f:
                content = f.read()
                assert 'structlog' in content, "Logging should use structlog"
                assert 'JSONRenderer' in content, "Logging should use JSON format"
                assert 'SensitiveDataMasker' in content, "Logging should mask sensitive data"
    
    def test_error_handling_improvements(self):
        """에러 처리 개선 확인."""
        # 커스텀 예외 클래스 확인
        exceptions_file = os.path.join(project_root, 'src', 'core', 'exceptions.py')
        if os.path.exists(exceptions_file):
            with open(exceptions_file, 'r') as f:
                content = f.read()
                assert 'class BaseResearcherException' in content, "Should have base BaseResearcherException"
                assert 'class ConfigurationError' in content, "Should have ConfigurationError"
                assert 'class ExecutionError' in content, "Should have ExecutionError"
                assert 'class TaskExecutionError' in content, "Should have TaskExecutionError"


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"])