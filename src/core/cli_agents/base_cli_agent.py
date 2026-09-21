from typing import Any, List

"""Base CLI Agent - CLI 기반 에이전트들의 공통 추상화 레이어

모든 CLI 기반 에이전트(claudecode, opencode, gemini cli, cline cli 등)의
공통 인터페이스를 정의하고 실행/결과 파싱을 표준화.
"""

import asyncio
import fcntl
import logging
import os
import pty
import struct
import termios
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

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
    working_dir: Path | None = None
    parse_output: bool = True
    output_format: str = "text"  # text, json, markdown 등


class BaseCLIAgent(ABC):
    """CLI 기반 에이전트의 추상 기본 클래스

    모든 CLI 에이전트는 이 클래스를 상속받아 구현해야 함.
    """

    def __init__(self, config: CLIAgentConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")

    @abstractmethod
    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """쿼리를 실행하고 결과를 반환

        Args:
            query: 실행할 쿼리
            **kwargs: 추가 파라미터

        Returns:
            표준화된 결과 딕셔너리
        """

    @abstractmethod
    def parse_output(self, result: CLIExecutionResult) -> Dict[str, Any]:
        """CLI 출력을 파싱하여 표준화된 형식으로 변환

        Args:
            result: CLI 실행 결과

        Returns:
            파싱된 결과
        """

    async def _execute_command(
        self, command: Union[str, List[str]], input_text: str | None = None
    ) -> CLIExecutionResult:
        """CLI 명령을 실제로 실행

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

            # 프로세스 실행 (timeout은 wait_for(communicate())에만 적용)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if input_text else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.working_dir,
            )

            # 입력 전송 및 출력 수신
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=input_text.encode() if input_text else None),
                timeout=self.config.timeout,
            )

            execution_time = time.time() - start_time

            # 결과 생성
            result = CLIExecutionResult(
                success=process.returncode == 0,
                output=stdout.decode("utf-8", errors="replace"),
                error=stderr.decode("utf-8", errors="replace"),
                exit_code=process.returncode or 0,
                execution_time=execution_time,
                metadata={
                    "command": cmd,
                    "working_dir": (
                        str(self.config.working_dir) if self.config.working_dir else None
                    ),
                },
            )

            self.logger.debug(
                f"CLI execution completed in {execution_time:.2f}s with exit code {result.exit_code}"
            )

            return result

        except TimeoutError:
            execution_time = time.time() - start_time
            self.logger.error(f"CLI command timed out after {execution_time:.2f}s")
            return CLIExecutionResult(
                success=False,
                output="",
                error=f"Command timed out after {self.config.timeout} seconds",
                exit_code=-1,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"CLI execution failed: {e}")
            return CLIExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=execution_time,
            )

    async def _execute_command_pty(
        self, command: Union[str, List[str]], timeout: int | None = None
    ) -> CLIExecutionResult:
        """Run a command under a real pty instead of a plain pipe.

        Plain ``subprocess.PIPE`` capture makes ``isatty()`` False in the
        child, so TTY-gated rendering (colors, spinners, width-based
        wrapping) never appears in the captured output -- a self-judge
        reading that capture is judging a different program than what a
        human sees. This gives it the same view a human terminal would.
        """
        start_time = time.time()
        timeout = timeout or self.config.timeout
        cmd = command.split() if isinstance(command, str) else list(command)

        master_fd, slave_fd = pty.openpty()
        try:
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        except OSError:
            pass

        env = {**self.config.env} if self.config.env else {}
        env.update(os.environ)
        env.setdefault("TERM", "xterm-256color")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env,
                cwd=self.config.working_dir,
            )
        finally:
            os.close(slave_fd)

        async def _drain() -> bytes:
            loop = asyncio.get_event_loop()
            chunks = []
            while True:
                try:
                    data = await loop.run_in_executor(None, os.read, master_fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                chunks.append(data)
            return b"".join(chunks)

        drain_task = asyncio.create_task(_drain())
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
            output = await asyncio.wait_for(drain_task, timeout=5)
        except TimeoutError:
            drain_task.cancel()
            process.kill()
            await process.wait()
            execution_time = time.time() - start_time
            self.logger.error(f"PTY command timed out after {execution_time:.2f}s")
            return CLIExecutionResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds",
                exit_code=-1,
                execution_time=execution_time,
            )
        finally:
            os.close(master_fd)

        execution_time = time.time() - start_time
        return CLIExecutionResult(
            success=process.returncode == 0,
            output=output.decode("utf-8", errors="replace"),
            error="",
            exit_code=process.returncode or 0,
            execution_time=execution_time,
            metadata={"command": cmd, "pty": True},
        )

    def _validate_result(self, result: CLIExecutionResult) -> bool:
        """실행 결과를 검증

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
        """에이전트 상태 확인

        Returns:
            건강 상태
        """
        try:
            # 간단한 테스트 쿼리 실행
            result = await self.execute_query("echo 'test'")
            return result.get("success", False)
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """에이전트 정보 반환

        Returns:
            에이전트 정보
        """
        return {
            "name": self.config.name,
            "type": "cli_agent",
            "command": self.config.command,
            "timeout": self.config.timeout,
            "output_format": self.config.output_format,
        }

    def context_window(self) -> int:
        """Best-known input+output context window for the active model.

        Real agentic CLIs (claude, codex) manage their own repo/file context
        internally and don't need this -- callers that pre-budget prompt
        content (fix_issue.py) still need a safe number to call regardless of
        which concrete agent they got, so this generic default exists for
        agents that don't override it with a precise, model-aware value
        (see OpenCodeAgent.context_window).
        """
        return 128_000

    def prompt_context_budget(self, reserve_output_tokens: int = 4_096) -> int:
        """Conservative prompt-input token budget for one agent call."""
        return max(1_000, self.context_window() - reserve_output_tokens - 4_000)
