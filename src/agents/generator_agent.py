"""Generator Agent (Synthesizer).

수집된 모든 정보와 검증된 결과를 종합하여 최종 리포트 및 결과물을 생성합니다.
"""

import logging
from typing import Dict, Any

from src.core.harness_state import HarnessState
from src.core.llm_manager import TaskType, execute_llm_task

logger = logging.getLogger(__name__)

class GeneratorAgent:
    """최종 결과물 생성 및 종합 에이전트"""
    
    def __init__(self):
        self.name = "generator_agent"
        
    async def synthesize(self, state: HarnessState) -> Dict[str, Any]:
        logger.info(f"[{self.name}] 📝 Synthesizing final results...")
        
        # 실제로는 state의 context나 results에서 내용을 읽어와 LLM 프롬프트 생성 
        final_output = state["workflow"].get("final_output", "Final compiled report based on tasks.")
        
        return {
            "final_output": final_output
        }
