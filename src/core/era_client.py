"""
ERA Agent HTTP API Client

ERA Agent의 HTTP API를 통해 안전한 코드 실행을 제공하는 클라이언트.
microVM 기반 격리 실행 환경을 제공합니다.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """코드 실행 결과"""
    exit_code: int
    stdout: str
    stderr: str
    duration: str
    vm_id: Optional[str] = None


class ERAClient:
    """ERA Agent HTTP API 클라이언트"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: float = 60.0
    ):
        """
        ERA 클라이언트 초기화
        
        Args:
            base_url: ERA Agent 서버 URL
            api_key: API 키 (선택사항)
            timeout: 요청 타임아웃 (초)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 가져오기 (지연 초기화)"""
        if self._client is None:
            headers = {
                'Content-Type': 'application/json',
            }
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout
            )
        return self._client
    
    async def close(self):
        """클라이언트 종료"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _make_request(
        self,
        endpoint: str,
        method: str = 'POST',
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        HTTP 요청 실행
        
        Args:
            endpoint: API 엔드포인트
            method: HTTP 메서드
            data: 요청 데이터
            
        Returns:
            API 응답 데이터
            
        Raises:
            ConnectionError: 서버 연결 실패
            ValueError: API 응답 오류
        """
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = await client.request(
                method=method,
                url=url,
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error')
                raise ValueError(f"ERA API error: {error_msg}")
            
            return result.get('data', {})
            
        except httpx.TimeoutException as e:
            raise ConnectionError(f"ERA server timeout: {e}")
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to ERA server at {self.base_url}: {e}")
        except httpx.HTTPStatusError as e:
            error_msg = "Unknown error"
            try:
                error_data = e.response.json()
                error_msg = error_data.get('error', str(e))
            except:
                error_msg = str(e)
            raise ValueError(f"ERA API HTTP error ({e.response.status_code}): {error_msg}")
        except Exception as e:
            raise ConnectionError(f"ERA API request failed: {e}")
    
    async def check_health(self) -> bool:
        """
        ERA 서버 상태 확인
        
        Returns:
            서버가 정상 동작 중이면 True
        """
        try:
            client = await self._get_client()
            # /api/vm/list 엔드포인트로 health check
            response = await client.get(f"{self.base_url}/api/vm/list", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"ERA health check failed: {e}")
            return False
    
    async def run_temp(
        self,
        language: str,
        command: str,
        cpu: int = 1,
        memory: int = 256,
        network: str = "none",
        timeout: int = 30,
        file: Optional[str] = None,
        code_content: Optional[str] = None
    ) -> ExecutionResult:
        """
        임시 VM에서 코드 실행
        
        Args:
            language: 프로그래밍 언어 (python, javascript, node, ruby, golang)
            command: 실행할 명령어
            cpu: CPU 수
            memory: 메모리 (MB)
            network: 네트워크 정책 (none, allow_all)
            timeout: 실행 타임아웃 (초)
            file: 업로드할 파일 경로 (선택사항, 로컬 파일 경로)
            code_content: 코드 내용 (선택사항, file이 없을 때 사용)
            
        Returns:
            실행 결과
        """
        req_data = {
            "language": language,
            "command": command,
            "cpu": cpu,
            "memory": memory,
            "network": network,
            "timeout": timeout
        }
        
        # file 파라미터는 로컬 파일 경로 (서버가 접근 가능한 경우)
        # HTTP API를 통해 전송할 때는 파일 내용을 base64로 인코딩하여 전송해야 할 수도 있음
        # 현재는 로컬 파일 경로를 전송 (서버가 같은 머신에서 실행되는 경우)
        if file:
            req_data["file"] = file
        elif code_content:
            # 코드 내용을 base64로 인코딩하여 전송 (향후 개선)
            # 현재는 file 파라미터만 지원하므로 임시 파일 사용
            import base64
            import tempfile
            import os
            temp_file = None
            try:
                # 임시 파일에 코드 저장
                file_ext = ".py" if language.lower() in ["python", "py"] else ".js"
                temp_file = os.path.join(tempfile.gettempdir(), f"era_code_{os.getpid()}_{id(code_content)}{file_ext}")
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(code_content)
                req_data["file"] = temp_file
            except Exception as e:
                logger.warning(f"Failed to create temp file for code: {e}")
                # fallback: code_content를 사용하지 않음
        
        try:
            result = await self._make_request('/api/vm/temp', 'POST', req_data)
            
            return ExecutionResult(
                exit_code=result.get('exit_code', -1),
                stdout=result.get('stdout', ''),
                stderr=result.get('stderr', ''),
                duration=result.get('duration', ''),
                vm_id=result.get('vm_id')
            )
        except Exception as e:
            logger.error(f"ERA run_temp failed: {e}")
            raise
    
    async def create_vm(
        self,
        language: str = "python",
        image: Optional[str] = None,
        cpu: int = 1,
        memory: int = 256,
        network: str = "none",
        persist: bool = False
    ) -> Dict[str, Any]:
        """
        영구 VM 생성
        
        Args:
            language: 프로그래밍 언어
            image: 커스텀 이미지 (선택사항)
            cpu: CPU 수
            memory: 메모리 (MB)
            network: 네트워크 정책
            persist: 영구 저장소 사용 여부
            
        Returns:
            VM 정보
        """
        req_data = {
            "language": language,
            "cpu": cpu,
            "memory": memory,
            "network": network,
            "persist": persist
        }
        
        if image:
            req_data["image"] = image
        
        return await self._make_request('/api/vm/create', 'POST', req_data)
    
    async def execute_in_vm(
        self,
        vm_id: str,
        command: str,
        file: Optional[str] = None,
        timeout: int = 30
    ) -> ExecutionResult:
        """
        기존 VM에서 명령 실행
        
        Args:
            vm_id: VM ID
            command: 실행할 명령어
            file: 업로드할 파일 경로 (선택사항)
            timeout: 실행 타임아웃 (초)
            
        Returns:
            실행 결과
        """
        req_data = {
            "vm_id": vm_id,
            "command": command,
            "timeout": timeout
        }
        
        if file:
            req_data["file"] = file
        
        try:
            result = await self._make_request('/api/vm/execute', 'POST', req_data)
            
            return ExecutionResult(
                exit_code=result.get('exit_code', -1),
                stdout=result.get('stdout', ''),
                stderr=result.get('stderr', ''),
                duration=result.get('duration', ''),
                vm_id=vm_id
            )
        except Exception as e:
            logger.error(f"ERA execute_in_vm failed: {e}")
            raise
    
    async def list_vms(self, status: Optional[str] = None, all_vms: bool = False) -> List[Dict[str, Any]]:
        """
        VM 목록 조회
        
        Args:
            status: 상태 필터 (선택사항)
            all_vms: 모든 VM 포함 여부
            
        Returns:
            VM 목록
        """
        params = {}
        if status:
            params['status'] = status
        if all_vms:
            params['all'] = 'true'
        
        query_string = '&'.join(f"{k}={v}" for k, v in params.items())
        endpoint = f'/api/vm/list'
        if query_string:
            endpoint += f'?{query_string}'
        
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            result = response.json()
            
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error')
                raise ValueError(f"ERA API error: {error_msg}")
            
            return result.get('data', [])
        except Exception as e:
            logger.error(f"ERA list_vms failed: {e}")
            raise
    
    async def stop_vm(self, vm_id: str):
        """VM 중지"""
        req_data = {"vm_id": vm_id}
        await self._make_request('/api/vm/stop', 'POST', req_data)
    
    async def clean_vm(self, vm_id: str, keep_persist: bool = False):
        """VM 정리"""
        req_data = {
            "vm_id": vm_id,
            "keep_persist": keep_persist
        }
        await self._make_request('/api/vm/clean', 'POST', req_data)

