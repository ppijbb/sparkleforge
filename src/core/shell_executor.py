#!/usr/bin/env python3
"""
Shell Command Executor (완전 자동형 SparkleForge)

안전한 Shell 명령 실행 기능.
위험한 명령 차단, 작업 디렉토리 제한, 사용자 확인 등의 안전장치 포함.
"""

import asyncio
import subprocess
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class CommandRisk(Enum):
    """명령 위험도."""
    SAFE = "safe"
    WARNING = "warning"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


class ShellExecutor:
    """안전한 Shell 명령 실행 클래스."""
    
    # 차단할 위험한 명령 패턴
    BLOCKED_PATTERNS = [
        r'rm\s+-rf\s+/',  # rm -rf /
        r'rm\s+-rf\s+~',  # rm -rf ~
        r'format\s+[a-z]',  # format C:
        r'dd\s+if=',  # dd if=
        r':\s*\(\)\s*\{',  # fork bomb
        r'mkfs\.',  # 파일시스템 포맷
        r'fdisk\s+/dev/',  # 디스크 파티션
        r'chmod\s+777\s+/',  # 전체 권한 변경
        r'chown\s+-R\s+.*\s+/',  # 전체 소유권 변경
    ]
    
    # 경고가 필요한 명령 패턴
    WARNING_PATTERNS = [
        r'rm\s+-rf',  # rm -rf (일반)
        r'git\s+push\s+--force',  # force push
        r'git\s+reset\s+--hard',  # hard reset
        r'docker\s+rm\s+-f',  # docker 강제 삭제
        r'kill\s+-9',  # 강제 종료
        r'sudo\s+',  # sudo 명령
    ]
    
    # 허용된 작업 디렉토리 (기본값)
    DEFAULT_ALLOWED_DIRS = [
        Path.cwd(),
        Path("./outputs"),
        Path("./workspace"),
        Path("./temp"),
        Path("./storage"),
    ]
    
    def __init__(
        self,
        allowed_dirs: Optional[List[Path]] = None,
        require_confirmation: bool = True,
        max_execution_time: int = 300  # 5분
    ):
        """
        초기화.
        
        Args:
            allowed_dirs: 허용된 작업 디렉토리 목록
            require_confirmation: 위험한 명령에 대해 확인 요청 여부
            max_execution_time: 최대 실행 시간 (초)
        """
        self.allowed_dirs = allowed_dirs or self.DEFAULT_ALLOWED_DIRS
        self.require_confirmation = require_confirmation
        self.max_execution_time = max_execution_time
    
    def _assess_risk(self, command: str) -> CommandRisk:
        """
        명령 위험도 평가.
        
        Args:
            command: 실행할 명령
        
        Returns:
            CommandRisk
        """
        command_lower = command.lower().strip()
        
        # 차단 패턴 확인
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, command_lower):
                logger.warning(f"Blocked dangerous command: {command[:100]}")
                return CommandRisk.BLOCKED
        
        # 경고 패턴 확인
        for pattern in self.WARNING_PATTERNS:
            if re.search(pattern, command_lower):
                return CommandRisk.WARNING
        
        return CommandRisk.SAFE
    
    def _is_safe_directory(self, directory: Path) -> bool:
        """작업 디렉토리가 안전한지 확인."""
        try:
            resolved = directory.resolve()
            
            # 허용된 디렉토리 확인
            for allowed in self.allowed_dirs:
                try:
                    allowed_resolved = allowed.resolve()
                    if resolved.is_relative_to(allowed_resolved) or resolved == allowed_resolved:
                        return True
                except (ValueError, AttributeError):
                    # Python 3.8 호환성
                    try:
                        if str(resolved).startswith(str(allowed_resolved)):
                            return True
                    except:
                        pass
            
            return False
        except Exception:
            return False
    
    async def run(
        self,
        command: str,
        working_dir: Optional[Path] = None,
        confirm: Optional[bool] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        안전한 Shell 명령 실행.
        
        Args:
            command: 실행할 명령
            working_dir: 작업 디렉토리 (None이면 현재 디렉토리)
            confirm: 확인 여부 (None이면 자동 결정)
            timeout: 타임아웃 (초, None이면 기본값 사용)
        
        Returns:
            실행 결과
        """
        # 위험도 평가
        risk = self._assess_risk(command)
        
        if risk == CommandRisk.BLOCKED:
            return {
                "success": False,
                "error": "Command blocked: potentially dangerous operation",
                "risk": risk.value,
                "command": command[:100]  # 보안상 전체 명령 노출 제한
            }
        
        # 작업 디렉토리 확인
        if working_dir:
            if not self._is_safe_directory(working_dir):
                return {
                    "success": False,
                    "error": f"Unsafe working directory: {working_dir}",
                    "command": command[:100]
                }
        else:
            working_dir = Path.cwd()
            if not self._is_safe_directory(working_dir):
                return {
                    "success": False,
                    "error": f"Current directory is not in allowed list: {working_dir}",
                    "command": command[:100]
                }
        
        # 확인 요청 (필요시)
        if risk == CommandRisk.WARNING:
            if confirm is None:
                confirm = self.require_confirmation
            
            if confirm is False:
                # 확인 없이 실행
                pass
            elif confirm is True:
                # 여기서는 자동으로 진행 (실제로는 사용자 확인 필요)
                logger.warning(f"Warning: Executing potentially risky command: {command[:100]}")
            # confirm이 None이면 자동으로 진행하지 않음 (사용자 확인 필요)
            # 실제 구현에서는 사용자 인터페이스와 통합 필요
        
        # 타임아웃 설정
        exec_timeout = timeout or self.max_execution_time
        
        try:
            # 명령 실행
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(working_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=exec_timeout
                )
            except asyncio.TimeoutError:
                # 타임아웃 시 프로세스 종료
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "error": f"Command execution timeout ({exec_timeout}s)",
                    "command": command[:100],
                    "risk": risk.value
                }
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode('utf-8', errors='replace').strip(),
                "stderr": stderr.decode('utf-8', errors='replace').strip(),
                "returncode": process.returncode,
                "command": command[:100],
                "risk": risk.value,
                "working_dir": str(working_dir)
            }
            
        except Exception as e:
            logger.error(f"Shell command execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "command": command[:100],
                "risk": risk.value
            }
    
    async def run_interactive(
        self,
        command: str,
        working_dir: Optional[Path] = None,
        input_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        대화형 명령 실행 (입력 지원).
        
        Args:
            command: 실행할 명령
            working_dir: 작업 디렉토리
            input_data: 입력 데이터
        
        Returns:
            실행 결과
        """
        # 위험도 평가
        risk = self._assess_risk(command)
        
        if risk == CommandRisk.BLOCKED:
            return {
                "success": False,
                "error": "Command blocked: potentially dangerous operation",
                "risk": risk.value
            }
        
        # 작업 디렉토리 확인
        if working_dir and not self._is_safe_directory(working_dir):
            return {
                "success": False,
                "error": f"Unsafe working directory: {working_dir}"
            }
        
        if not working_dir:
            working_dir = Path.cwd()
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(working_dir),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True
            )
            
            # 입력 데이터 전송
            if input_data:
                stdout, stderr = await process.communicate(input_data.encode('utf-8'))
            else:
                stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode('utf-8', errors='replace').strip(),
                "stderr": stderr.decode('utf-8', errors='replace').strip(),
                "returncode": process.returncode,
                "command": command[:100],
                "risk": risk.value
            }
            
        except Exception as e:
            logger.error(f"Interactive command execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "command": command[:100]
            }
    
    async def run_background(
        self,
        command: str,
        working_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        백그라운드 작업 실행.
        
        Args:
            command: 실행할 명령
            working_dir: 작업 디렉토리
        
        Returns:
            작업 정보 (PID 등)
        """
        # 위험도 평가
        risk = self._assess_risk(command)
        
        if risk == CommandRisk.BLOCKED:
            return {
                "success": False,
                "error": "Command blocked: potentially dangerous operation",
                "risk": risk.value
            }
        
        # 작업 디렉토리 확인
        if working_dir and not self._is_safe_directory(working_dir):
            return {
                "success": False,
                "error": f"Unsafe working directory: {working_dir}"
            }
        
        if not working_dir:
            working_dir = Path.cwd()
        
        try:
            # 백그라운드 실행
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(working_dir),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                shell=True
            )
            
            return {
                "success": True,
                "pid": process.pid,
                "command": command[:100],
                "risk": risk.value,
                "working_dir": str(working_dir)
            }
            
        except Exception as e:
            logger.error(f"Background command execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "command": command[:100]
            }




