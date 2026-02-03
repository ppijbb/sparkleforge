"""
Docker 기반 코드 실행 샌드박스

안전한 코드 실행을 위한 Docker 컨테이너 기반 샌드박스 환경.
실제 SparkleForge 도구에서 사용되는 코드 실행을 담당.
"""

import asyncio
import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """샌드박스 설정"""
    image: str = "python:3.11-slim"
    timeout: int = 30
    memory_limit: str = "512m"
    cpu_limit: float = 0.5
    network_disabled: bool = True
    read_only: bool = False
    tmpfs_size: str = "100m"


@dataclass
class ExecutionResult:
    """실행 결과"""
    success: bool
    output: str
    error: str
    exit_code: int
    execution_time: float
    container_id: Optional[str] = None


class DockerSandbox:
    """Docker 기반 코드 실행 샌드박스"""

    def __init__(self, config: Optional[SandboxConfig] = None):
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker is not available. Please install docker package.")

        self.config = config or SandboxConfig()
        self.docker_client = docker.from_env()

        # 컨테이너 풀 관리
        self.container_pool = []
        self.max_pool_size = 3

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        input_data: Optional[str] = None
    ) -> ExecutionResult:
        """
        코드를 안전하게 실행

        Args:
            code: 실행할 코드
            language: 프로그래밍 언어 (python, javascript, bash 등)
            input_data: 표준 입력 데이터

        Returns:
            ExecutionResult: 실행 결과
        """
        import time
        start_time = time.time()

        try:
            # 컨테이너 풀에서 컨테이너 가져오기 또는 생성
            container = await self._get_container()

            # 언어별 실행 명령 생성
            cmd, file_path = self._prepare_execution(code, language)

            # 파일을 컨테이너에 복사
            if file_path:
                with open(file_path, 'rb') as f:
                    container.put_archive('/tmp', f.read())

            # 실행
            exec_result = container.exec_run(
                cmd=cmd,
                stdin=input_data.encode() if input_data else None,
                timeout=self.config.timeout,
                demux=True
            )

            stdout, stderr = exec_result.output
            exit_code = exec_result.exit_code

            # 결과 정리
            output = stdout.decode('utf-8') if stdout else ""
            error = stderr.decode('utf-8') if stderr else ""

            # 임시 파일 정리
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=exit_code == 0,
                output=output,
                error=error,
                exit_code=exit_code,
                execution_time=execution_time,
                container_id=container.id
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Sandbox execution failed: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=execution_time
            )
        finally:
            # 컨테이너 풀로 반환
            if 'container' in locals():
                await self._return_container(container)

    async def execute_file(
        self,
        file_path: str,
        language: str = "python",
        input_data: Optional[str] = None
    ) -> ExecutionResult:
        """
        파일을 안전하게 실행

        Args:
            file_path: 실행할 파일 경로
            language: 프로그래밍 언어
            input_data: 표준 입력 데이터

        Returns:
            ExecutionResult: 실행 결과
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            return await self.execute_code(code, language, input_data)
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Failed to read file: {e}",
                exit_code=-1,
                execution_time=0.0
            )

    def _prepare_execution(self, code: str, language: str) -> Tuple[str, Optional[str]]:
        """
        언어별 실행 준비

        Returns:
            Tuple[str, Optional[str]]: (실행 명령, 임시 파일 경로)
        """
        if language.lower() in ["python", "py"]:
            # 긴 코드는 임시 파일로 실행
            if len(code) > 1000 or code.count('\n') > 10:
                file_path = self._create_temp_file(code, ".py")
                cmd = f"python {file_path}"
                return cmd, file_path
            else:
                # 짧은 코드는 stdin으로 실행
                return ["python", "-c", code], None

        elif language.lower() in ["javascript", "js", "node", "nodejs"]:
            if len(code) > 1000 or code.count('\n') > 10:
                file_path = self._create_temp_file(code, ".js")
                cmd = f"node {file_path}"
                return cmd, file_path
            else:
                return ["node", "-e", code], None

        elif language.lower() in ["bash", "sh"]:
            # 쉘 스크립트는 항상 파일로 실행
            file_path = self._create_temp_file(code, ".sh")
            cmd = f"bash {file_path}"
            return cmd, file_path

        else:
            raise ValueError(f"Unsupported language: {language}")

    def _create_temp_file(self, content: str, extension: str) -> str:
        """임시 파일 생성"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as f:
            f.write(content)
            return f.name

    async def _get_container(self):
        """컨테이너 풀에서 컨테이너 가져오기"""
        # 풀에서 사용 가능한 컨테이너 찾기
        for container in self.container_pool:
            if container.status == 'running':
                return container

        # 새 컨테이너 생성
        if len(self.container_pool) < self.max_pool_size:
            container = self.docker_client.containers.run(
                self.config.image,
                command="tail -f /dev/null",  # 무한 실행
                detach=True,
                mem_limit=self.config.memory_limit,
                cpu_quota=int(self.config.cpu_limit * 100000),
                network_mode='none' if self.config.network_disabled else 'bridge',
                read_only=self.config.read_only,
                tmpfs={'/tmp': f'size={self.config.tmpfs_size}'} if not self.config.read_only else None
            )
            self.container_pool.append(container)
            return container

        # 풀에서 가장 오래된 컨테이너 재사용
        container = self.container_pool.pop(0)
        try:
            container.restart()
        except:
            # 재시작 실패 시 새 컨테이너 생성
            container = self.docker_client.containers.run(
                self.config.image,
                command="tail -f /dev/null",
                detach=True,
                mem_limit=self.config.memory_limit,
                cpu_quota=int(self.config.cpu_limit * 100000),
                network_mode='none' if self.config.network_disabled else 'bridge',
                read_only=self.config.read_only
            )
        self.container_pool.append(container)
        return container

    async def _return_container(self, container):
        """컨테이너를 풀로 반환"""
        # 컨테이너 정리 (임시 파일 삭제 등)
        try:
            container.exec_run("rm -rf /tmp/*")
        except:
            pass  # 정리 실패는 무시

        # 풀에 추가 (이미 있다면 무시)
        if container not in self.container_pool:
            self.container_pool.append(container)

    async def cleanup(self):
        """샌드박스 정리"""
        for container in self.container_pool:
            try:
                container.stop()
                container.remove()
            except:
                pass
        self.container_pool.clear()

    async def health_check(self) -> bool:
        """샌드박스 상태 확인"""
        try:
            # 간단한 코드 실행으로 테스트
            result = await self.execute_code("print('sandbox ready')", "python")
            return result.success
        except:
            return False


# 전역 샌드박스 인스턴스
_sandbox_instance = None

def get_sandbox() -> DockerSandbox:
    """전역 샌드박스 인스턴스 가져오기"""
    global _sandbox_instance
    if _sandbox_instance is None:
        config = SandboxConfig()
        _sandbox_instance = DockerSandbox(config)
    return _sandbox_instance