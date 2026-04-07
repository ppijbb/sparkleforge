"""Verifier Agent.

실행된 태스크 결과를 검증하고, 재시도 여부를 판단합니다.
"""

import logging
from typing import Dict, Any, List

from src.core.harness_state import HarnessState
from src.core.llm_manager import TaskType, execute_llm_task
from src.core.skills.agent_loader import get_prompt

logger = logging.getLogger(__name__)

class VerifierAgent:
    """결과 검증 에이전트"""
    
    def __init__(self):
        self.name = "verifier_agent"
        self.instruction = "You are an expert at verifying research results against task requirements."

    async def verify_results(self, state: HarnessState, completed_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"[{self.name}] 🔍 Verifying results for {len(completed_tasks)} tasks...")
        
        verified_results = []
        failed_tasks = []
        
        for task in completed_tasks:
             # 임시 검증 로직 / LLM 검증 호출 가능
             result_data = task.get("result", {})
             if result_data:
                  logger.info(f"[{self.name}] Task {task.get('task_id')} verified successfully.")
                  verified_results.append(task)
             else:
                  logger.warning(f"[{self.name}] Task {task.get('task_id')} validation failed.")
                  failed_tasks.append(task)
                  
        return {
            "verified_results": verified_results,
            "failed_tasks": failed_tasks
        }
