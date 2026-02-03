"""
Base CLI Agent - CLI 기반 에이전트들의 공통 추상화 레이어

모든 CLI 기반 에이전트(claudecode, opencode, gemini cli, cline cli 등)의
공통 인터페이스를 정의하고 실행/결과 파싱을 표준화.
"""

import asyncio
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CLIExecutionResult:
    """CLI 실행 결과"""
    success: bool
    output: str
    error: str
    exit_code: int
    execution_time: float
    metadata: Dict[str, Any] = None


@dataclass
class CLIAgentConfig:
    """CLI 에이전트 설정"""
    name: str
    command: Union[str, List[str]]
    args: List[str] = None
    env: Dict[str, str] = None
    timeout: int = 300
    working_dir: Optional[Path] = None
    parse_output: bool = True
    output_format: str = "text"  # text, json, markdown 등


class BaseCLIAgent(ABC):
    """
    CLI 기반 에이전트의 추상 기본 클래스

    모든 CLI 에이전트는 이 클래스를 상속받아 구현해야 함.
    """

    def __init__(self, config: CLIAgentConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")

    @abstractmethod
    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        쿼리를 실행하고 결과를 반환

        Args:
            query: 실행할 쿼리
            **kwargs: 추가 파라미터

        Returns:
            표준화된 결과 딕셔너리
        """
        pass

    @abstractmethod
    def parse_output(self, result: CLIExecutionResult) -> Dict[str, Any]:
        """
        CLI 출력을 파싱하여 표준화된 형식으로 변환

        Args:
            result: CLI 실행 결과

        Returns:
            파싱된 결과
        """
        pass

    async def _execute_command(self, command: Union[str, List[str]],
                              input_text: Optional[str] = None) -> CLIExecutionResult:
        """
        CLI 명령을 실제로 실행

        Args:
            command: 실행할 명령어
            input_text: stdin으로 전달할 텍스트

        Returns:
            실행 결과
        """
        start_time = time.time()

        try:
            # 명령어 준비
            if isinstance(command, str):
                cmd = command.split()
            else:
                cmd = command.copy()

            # 추가 인자 추가
            if self.config.args:
                cmd.extend(self.config.args)

            self.logger.debug(f"Executing CLI command: {' '.join(cmd)}")

            # 환경 변수 설정
            env = {**self.config.env} if self.config.env else {}
            env.update(os.environ)

            # 프로세스 실행
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if input_text else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.working_dir,
                timeout=self.config.timeout
            )

            # 입력 전송 및 출력 수신
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=input_text.encode() if input_text else None),
                timeout=self.config.timeout
            )

            execution_time = time.time() - start_time

            # 결과 생성
            result = CLIExecutionResult(
                success=process.returncode == 0,
                output=stdout.decode('utf-8', errors='replace'),
                error=stderr.decode('utf-8', errors='replace'),
                exit_code=process.returncode or 0,
                execution_time=execution_time,
                metadata={
                    'command': cmd,
                    'working_dir': str(self.config.working_dir) if self.config.working_dir else None
                }
            )

            self.logger.debug(f"CLI execution completed in {execution_time:.2f}s with exit code {result.exit_code}")

            return result

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            self.logger.error(f"CLI command timed out after {execution_time:.2f}s")
            return CLIExecutionResult(
                success=False,
                output="",
                error=f"Command timed out after {self.config.timeout} seconds",
                exit_code=-1,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"CLI execution failed: {e}")
            return CLIExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=execution_time
            )

    def _validate_result(self, result: CLIExecutionResult) -> bool:
        """
        실행 결과를 검증

        Args:
            result: 실행 결과

        Returns:
            유효성 여부
        """
        # 기본 검증: 성공 여부
        if not result.success:
            return False

        # 출력 존재 여부
        if not result.output.strip():
            return False

        return True

    async def health_check(self) -> bool:
        """
        에이전트 상태 확인

        Returns:
            건강 상태
        """
        try:
            # 간단한 테스트 쿼리 실행
            result = await self.execute_query("echo 'test'")
            return result.get('success', False)
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        에이전트 정보 반환

        Returns:
            에이전트 정보
        """
        return {
            'name': self.config.name,
            'type': 'cli_agent',
            'command': self.config.command,
            'timeout': self.config.timeout,
            'output_format': self.config.output_format
        }