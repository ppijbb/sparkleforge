"""Workflow Audit Logger.

하네스의 상태 전이, 도구 사용 이력, 서브에이전트 실행 등을 파일 시스템 내에 Audit Trail로 기록합니다.
어떤 권한(TrustGate)으로 어떤 도구(ToolGovernor)를 호출했는지 증명하기 위해 사용됩니다.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any

from src.core.harness_state import HarnessState

logger = logging.getLogger(__name__)

class WorkflowAuditLogger:
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log_state_transition(self, session_id: str, old_phase: str, new_phase: str, state_subset: Dict[str, Any]):
        """상태 전이 기록"""
        entry = {
            "timestamp": time.time(),
            "type": "state_transition",
            "session_id": session_id,
            "old_phase": old_phase,
            "new_phase": new_phase,
            "state_metadata": state_subset
        }
        self._write_log(session_id, entry)

    def log_tool_execution(self, session_id: str, agent_id: str, tool_name: str, parameters: Dict[str, Any], success: bool, duration: float):
        """도구 사용 기록 (Governance 증명)"""
        entry = {
            "timestamp": time.time(),
            "type": "tool_execution",
            "session_id": session_id,
            "agent_id": agent_id,
            "tool_name": tool_name,
            # 주의: 보안상 민감 파라미터는 마스킹 처리할 수 있음
            "parameters": parameters,
            "success": success,
            "duration": duration
        }
        self._write_log(session_id, entry)
        
    def _write_log(self, session_id: str, entry: Dict[str, Any]):
        try:
            log_file = self.log_dir / f"audit_{session_id}.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

_audit_logger = None

def get_audit_logger() -> WorkflowAuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = WorkflowAuditLogger()
    return _audit_logger
