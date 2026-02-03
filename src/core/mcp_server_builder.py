"""
MCP Server Builder - 자동 MCP 서버 생성 시스템

도구가 없거나 실행 실패 시, LLM을 활용하여 필요한 MCP 서버를 Python FastMCP로
자동 빌드하고 동적으로 등록하여 사용할 수 있도록 합니다.

참고: MCP Builder Skill Guide
https://github.com/anthropics/skills/blob/main/skills/mcp-builder/SKILL.md
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import hashlib

from src.core.llm_manager import execute_llm_task, TaskType
from src.core.researcher_config import get_llm_config, get_agent_config
from src.core.process_manager import get_process_manager

logger = logging.getLogger(__name__)

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent.parent


class MCPServerBuilder:
    """
    MCP 서버 자동 빌더
    
    도구 요구사항을 분석하고 FastMCP 기반 MCP 서버를 자동으로 생성합니다.
    """
    
    def __init__(self):
        """MCPServerBuilder 초기화"""
        self.llm_config = get_llm_config()
        self.agent_config = get_agent_config()
        
        # MCP 설정 로드
        try:
            from src.core.researcher_config import get_mcp_config
            self.config = get_mcp_config()
        except Exception as e:
            logger.warning(f"Failed to load MCP config: {e}")
            self.config = None
        
        # 빌드된 서버 저장 디렉토리
        if self.config and hasattr(self.config, 'builder_temp_dir'):
            server_dir_str = self.config.builder_temp_dir
        else:
            server_dir_str = "temp/mcp_servers"
        
        self.server_dir = project_root / server_dir_str
        self.server_dir.mkdir(parents=True, exist_ok=True)
        
        # 빌드 캐시 (동일 요구사항 재사용)
        self.build_cache: Dict[str, Dict[str, Any]] = {}
        
        # 빌드 메타데이터 저장
        self.metadata_file = self.server_dir / "build_metadata.json"
        self._load_metadata()
        
        logger.info(f"MCP Server Builder initialized (server_dir: {self.server_dir})")
        
        # ProcessManager에 cleanup callback 등록
        try:
            from src.core.process_manager import get_process_manager
            pm = get_process_manager()
            pm.cleanup_callbacks.append(self._cleanup_built_servers)
        except Exception as e:
            logger.debug(f"Failed to register cleanup callback: {e}")
    
    def _cleanup_built_servers(self):
        """빌드된 서버 정리 (ProcessManager cleanup callback)"""
        try:
            if self.config and hasattr(self.config, 'builder_auto_cleanup') and not self.config.builder_auto_cleanup:
                return  # auto_cleanup이 비활성화된 경우 스킵
            
            # 빌드된 서버 디렉토리 정리 (선택적)
            # 실제 서버 프로세스는 ProcessManager가 종료하므로 여기서는 메타데이터만 정리
            logger.debug("[MCP][builder.cleanup] Cleaning up build metadata")
        except Exception as e:
            logger.debug(f"[MCP][builder.cleanup] Cleanup error (ignored): {e}")
    
    def _load_metadata(self):
        """빌드 메타데이터 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.build_cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load build metadata: {e}")
                self.build_cache = {}
        else:
            self.build_cache = {}
    
    def _save_metadata(self):
        """빌드 메타데이터 저장"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.build_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save build metadata: {e}")
    
    def _get_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        key_data = {
            "tool_name": tool_name,
            "parameters": sorted(parameters.items())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def analyze_tool_requirements(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        error_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        LLM을 활용하여 도구 요구사항 분석
        
        Args:
            tool_name: 도구 이름
            parameters: 도구 파라미터
            error_context: 에러 컨텍스트 (있는 경우)
            
        Returns:
            요구사항 분석 결과 (API 엔드포인트, 인증 방식, 스키마 등)
        """
        logger.info(f"Analyzing requirements for tool: {tool_name}")
        
        # MCP Best Practices 참조 프롬프트
        prompt = f"""You are an expert MCP server developer. Analyze the requirements for implementing an MCP tool.

Tool Name: {tool_name}
Parameters: {json.dumps(parameters, indent=2, ensure_ascii=False)}
Error Context: {error_context or "None"}

Based on the MCP Best Practices and FastMCP patterns, determine:

1. **API Endpoints or Data Sources**:
   - What API endpoints or data sources are needed?
   - What HTTP methods (GET, POST, PUT, DELETE) are required?
   - What is the base URL or service?

2. **Authentication Method**:
   - API Key (Bearer token, query parameter, header)?
   - OAuth?
   - No authentication?
   - Environment variable name for API key?

3. **Input Schema**:
   - Pydantic model structure for tool input
   - Field types, descriptions, defaults, constraints
   - Required vs optional fields

4. **Output Format**:
   - What format should the tool return? (JSON, text, structured data)
   - Response structure

5. **Error Handling**:
   - Common error scenarios
   - Error message format

6. **Dependencies**:
   - Required Python packages (httpx, requests, etc.)
   - Any special libraries needed?

7. **Server Name**:
   - Appropriate server name (lowercase, no spaces, descriptive)

Respond in JSON format:
{{
    "server_name": "service_name",
    "api_base_url": "https://api.example.com",
    "auth_method": "api_key|oauth|none",
    "api_key_env": "SERVICE_API_KEY",
    "tools": [
        {{
            "name": "{tool_name}",
            "method": "GET|POST|PUT|DELETE",
            "endpoint": "/path/to/endpoint",
            "input_schema": {{
                "field_name": {{
                    "type": "string|int|float|bool|list|dict",
                    "description": "...",
                    "required": true|false,
                    "default": null
                }}
            }},
            "output_format": "json|text|structured"
        }}
    ],
    "dependencies": ["httpx", "pydantic"],
    "error_handling": {{
        "common_errors": ["rate_limit", "auth_failed"],
        "retry_strategy": "exponential_backoff"
    }}
}}
"""
        
        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.ANALYSIS,
                system_message="You are an expert MCP server developer. Analyze tool requirements and provide structured JSON responses."
            )
            
            # JSON 응답 파싱
            response_text = result.content if hasattr(result, 'content') else str(result)
            
            # JSON 추출 (코드 블록 제거)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                requirements = json.loads(json_match.group())
            else:
                # JSON이 없으면 기본 구조 생성
                logger.warning(f"Could not parse JSON from LLM response, using defaults")
                requirements = {
                    "server_name": tool_name.lower().replace(" ", "_").replace("-", "_"),
                    "api_base_url": "https://api.example.com",
                    "auth_method": "api_key",
                    "api_key_env": f"{tool_name.upper()}_API_KEY",
                    "tools": [{
                        "name": tool_name,
                        "method": "GET",
                        "endpoint": "/api/v1/endpoint",
                        "input_schema": {k: {"type": "string", "description": str(v), "required": True} 
                                        for k, v in parameters.items()},
                        "output_format": "json"
                    }],
                    "dependencies": ["httpx"],
                    "error_handling": {
                        "common_errors": [],
                        "retry_strategy": "none"
                    }
                }
            
            logger.info(f"Requirements analyzed: server_name={requirements.get('server_name')}, tools={len(requirements.get('tools', []))}")
            return requirements
            
        except Exception as e:
            logger.error(f"Failed to analyze tool requirements: {e}", exc_info=True)
            # 기본 요구사항 반환
            return {
                "server_name": tool_name.lower().replace(" ", "_").replace("-", "_"),
                "api_base_url": "https://api.example.com",
                "auth_method": "api_key",
                "api_key_env": f"{tool_name.upper()}_API_KEY",
                "tools": [{
                    "name": tool_name,
                    "method": "GET",
                    "endpoint": "/api/v1/endpoint",
                    "input_schema": {k: {"type": "string", "description": str(v), "required": True} 
                                    for k, v in parameters.items()},
                    "output_format": "json"
                }],
                "dependencies": ["httpx"],
                "error_handling": {
                    "common_errors": [],
                    "retry_strategy": "none"
                }
            }
    
    def generate_server_code(self, requirements: Dict[str, Any]) -> str:
        """
        FastMCP 서버 코드 생성
        
        Args:
            requirements: 요구사항 분석 결과
            
        Returns:
            생성된 FastMCP 서버 코드
        """
        server_name = requirements.get("server_name", "mcp_server")
        tools = requirements.get("tools", [])
        api_base_url = requirements.get("api_base_url", "https://api.example.com")
        auth_method = requirements.get("auth_method", "api_key")
        api_key_env = requirements.get("api_key_env", "API_KEY")
        dependencies = requirements.get("dependencies", ["httpx"])
        
        # FastMCP 서버 코드 템플릿
        code_parts = [
            '"""',
            f"Auto-generated MCP server for {server_name}",
            f"Generated at: {datetime.now().isoformat()}",
            '"""',
            '',
            'import os',
            'import json',
            'import logging',
            'from typing import Any, Dict, Optional',
            'from pathlib import Path',
            '',
            'try:',
            '    from fastmcp import FastMCP',
            '    from pydantic import BaseModel, Field',
            '    import httpx',
            'except ImportError as e:',
            '    raise ImportError(f"Required packages not installed: {e}. Install with: pip install fastmcp pydantic httpx")',
            '',
            'logger = logging.getLogger(__name__)',
            '',
            f'mcp = FastMCP("{server_name}")',
            '',
        ]
        
        # 각 도구에 대한 Pydantic 모델 및 함수 생성
        for tool in tools:
            tool_name = tool.get("name", "tool")
            input_schema = tool.get("input_schema", {})
            method = tool.get("method", "GET")
            endpoint = tool.get("endpoint", "/api/v1/endpoint")
            output_format = tool.get("output_format", "json")
            
            # Pydantic 모델 생성
            model_name = f"{tool_name.title().replace('_', '')}Input"
            code_parts.append(f'class {model_name}(BaseModel):')
            code_parts.append(f'    """Input schema for {tool_name}"""')
            
            for field_name, field_info in input_schema.items():
                field_type = field_info.get("type", "str")
                field_desc = field_info.get("description", "")
                field_required = field_info.get("required", True)
                field_default = field_info.get("default")
                
                # Python 타입 매핑
                type_map = {
                    "string": "str",
                    "int": "int",
                    "float": "float",
                    "bool": "bool",
                    "list": "List[str]",
                    "dict": "Dict[str, Any]"
                }
                python_type = type_map.get(field_type, "str")
                
                if field_required and field_default is None:
                    code_parts.append(f'    {field_name}: {python_type} = Field(..., description="{field_desc}")')
                else:
                    default_val = json.dumps(field_default) if field_default is not None else "None"
                    code_parts.append(f'    {field_name}: Optional[{python_type}] = Field(default={default_val}, description="{field_desc}")')
            
            code_parts.append('')
            
            # 도구 함수 생성
            code_parts.append(f'@mcp.tool()')
            code_parts.append(f'async def {tool_name}(input: {model_name}) -> str:')
            code_parts.append(f'    """')
            code_parts.append(f'    {tool_name} tool')
            code_parts.append(f'    """')
            code_parts.append(f'    try:')
            
            # 인증 헤더 설정
            if auth_method == "api_key":
                code_parts.append(f'        api_key = os.getenv("{api_key_env}")')
                code_parts.append(f'        if not api_key:')
                code_parts.append(f'            return json.dumps({{"error": "API key not found. Set {api_key_env} environment variable."}})')
                code_parts.append(f'        headers = {{"Authorization": f"Bearer {{api_key}}"}}')
            elif auth_method == "oauth":
                code_parts.append(f'        # OAuth token retrieval (implement based on service)')
                code_parts.append(f'        headers = {{}}')
            else:
                code_parts.append(f'        headers = {{}}')
            
            code_parts.append('')
            
            # HTTP 요청 생성
            if method.upper() == "GET":
                code_parts.append(f'        url = f"{api_base_url}{endpoint}"')
                code_parts.append(f'        params = input.model_dump(exclude_none=True)')
                code_parts.append(f'        async with httpx.AsyncClient(timeout=30.0) as client:')
                code_parts.append(f'            response = await client.get(url, params=params, headers=headers)')
            elif method.upper() == "POST":
                code_parts.append(f'        url = f"{api_base_url}{endpoint}"')
                code_parts.append(f'        data = input.model_dump(exclude_none=True)')
                code_parts.append(f'        async with httpx.AsyncClient(timeout=30.0) as client:')
                code_parts.append(f'            response = await client.post(url, json=data, headers=headers)')
            elif method.upper() == "PUT":
                code_parts.append(f'        url = f"{api_base_url}{endpoint}"')
                code_parts.append(f'        data = input.model_dump(exclude_none=True)')
                code_parts.append(f'        async with httpx.AsyncClient(timeout=30.0) as client:')
                code_parts.append(f'            response = await client.put(url, json=data, headers=headers)')
            elif method.upper() == "DELETE":
                code_parts.append(f'        url = f"{api_base_url}{endpoint}"')
                code_parts.append(f'        async with httpx.AsyncClient(timeout=30.0) as client:')
                code_parts.append(f'            response = await client.delete(url, headers=headers)')
            else:
                code_parts.append(f'        url = f"{api_base_url}{endpoint}"')
                code_parts.append(f'        async with httpx.AsyncClient(timeout=30.0) as client:')
                code_parts.append(f'            response = await client.request("{method.upper()}", url, headers=headers)')
            
            code_parts.append('            response.raise_for_status()')
            code_parts.append('')
            
            # 응답 처리
            if output_format == "json":
                code_parts.append('            result = response.json()')
                code_parts.append('            return json.dumps(result, ensure_ascii=False, indent=2)')
            else:
                code_parts.append('            return response.text')
            
            code_parts.append('')
            code_parts.append('    except httpx.HTTPStatusError as e:')
            code_parts.append('        error_msg = f"HTTP {e.response.status_code}: {e.response.text}"')
            code_parts.append('        logger.error(error_msg)')
            code_parts.append('        return json.dumps({"error": error_msg})')
            code_parts.append('    except Exception as e:')
            code_parts.append(f'        error_msg = f"Error in {tool_name}: {{str(e)}}"')
            code_parts.append('        logger.error(error_msg)')
            code_parts.append('        return json.dumps({"error": error_msg})')
            code_parts.append('')
        
        # 메인 실행 부분
        code_parts.extend([
            'if __name__ == "__main__":',
            '    mcp.run()',
            ''
        ])
        
        server_code = '\n'.join(code_parts)
        logger.debug(f"Generated server code for {server_name} ({len(server_code)} chars)")
        return server_code
    
    async def build_server(
        self,
        server_name: str,
        server_code: str,
        requirements: Dict[str, Any]
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        서버 빌드 및 검증
        
        Args:
            server_name: 서버 이름
            server_code: 생성된 서버 코드
            requirements: 요구사항
            
        Returns:
            (서버 파일 경로, 메타데이터)
        """
        logger.info(f"Building MCP server: {server_name}")
        
        # 서버 디렉토리 생성
        server_dir = self.server_dir / server_name
        server_dir.mkdir(parents=True, exist_ok=True)
        
        # 서버 파일 저장
        server_file = server_dir / "server.py"
        server_file.write_text(server_code, encoding='utf-8')
        logger.info(f"Server code saved to: {server_file}")
        
        # requirements.txt 생성
        dependencies = requirements.get("dependencies", ["httpx", "fastmcp", "pydantic"])
        requirements_txt = server_dir / "requirements.txt"
        requirements_txt.write_text('\n'.join(dependencies) + '\n', encoding='utf-8')
        
        # 메타데이터 저장
        metadata = {
            "server_name": server_name,
            "created_at": datetime.now().isoformat(),
            "requirements": requirements,
            "server_file": str(server_file),
            "dependencies": dependencies
        }
        metadata_file = server_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        
        # 의존성 설치 확인 (선택적)
        try:
            # fastmcp가 설치되어 있는지 확인
            import fastmcp
            logger.debug("fastmcp is available")
        except ImportError:
            logger.warning("fastmcp not installed. Server may not work without: pip install fastmcp")
        
        # 서버 코드 검증 (기본 문법 체크)
        try:
            compile(server_code, str(server_file), 'exec')
            logger.info(f"Server code validation passed: {server_name}")
        except SyntaxError as e:
            logger.error(f"Server code syntax error: {e}")
            raise ValueError(f"Generated server code has syntax errors: {e}")
        
        return server_file, metadata
    
    async def build_mcp_server(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        error_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        전체 MCP 서버 빌드 프로세스
        
        Args:
            tool_name: 도구 이름
            parameters: 도구 파라미터
            error_context: 에러 컨텍스트
            
        Returns:
            빌드 결과 (success, server_path, server_name, metadata 등)
        """
        try:
            # 캐시 확인
            cache_key = self._get_cache_key(tool_name, parameters)
            if cache_key in self.build_cache:
                cached = self.build_cache[cache_key]
                server_path = Path(cached["server_path"])
                if server_path.exists():
                    logger.info(f"Using cached server: {cached['server_name']}")
                    return {
                        "success": True,
                        "server_path": server_path,
                        "server_name": cached["server_name"],
                        "metadata": cached.get("metadata", {}),
                        "cached": True
                    }
            
            # 요구사항 분석
            requirements = await self.analyze_tool_requirements(tool_name, parameters, error_context)
            server_name = requirements.get("server_name", tool_name.lower().replace(" ", "_"))
            
            # 서버 코드 생성
            server_code = self.generate_server_code(requirements)
            
            # 서버 빌드
            server_path, metadata = await self.build_server(server_name, server_code, requirements)
            
            # 캐시에 저장
            self.build_cache[cache_key] = {
                "server_name": server_name,
                "server_path": str(server_path),
                "metadata": metadata,
                "created_at": datetime.now().isoformat()
            }
            self._save_metadata()
            
            logger.info(f"✅ MCP server built successfully: {server_name} at {server_path}")
            
            return {
                "success": True,
                "server_path": server_path,
                "server_name": server_name,
                "metadata": metadata,
                "cached": False
            }
            
        except Exception as e:
            logger.error(f"Failed to build MCP server: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "server_path": None,
                "server_name": None
            }


# 싱글톤 인스턴스
_builder_instance: Optional[MCPServerBuilder] = None


def get_mcp_server_builder() -> MCPServerBuilder:
    """MCPServerBuilder 싱글톤 인스턴스 반환"""
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = MCPServerBuilder()
    return _builder_instance

