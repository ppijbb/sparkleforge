"""Planner Agent.

요청을 분석하고 연구 계획 및 하위 작업을 기획합니다.
기존 agent_orchestrator에서 추려낸 풀 버전 (Council, Task Decomposition 포함).
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List

from src.core.harness_state import HarnessState
from src.core.llm_manager import TaskType, execute_llm_task
from src.core.skills.agent_loader import get_prompt

logger = logging.getLogger(__name__)

class PlannerAgent:
    """계획 및 작업 분할 에이전트"""
    
    def __init__(self):
        self.name = "planner_agent"
        self.instruction = "You are an expert research planning agent."
        
    async def create_plan(self, state: HarnessState) -> Dict[str, Any]:
        logger.info(f"[{self.name}] 🔍 Starting research planning...")
        
        user_query = state["workflow"]["user_query"]
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")
        
        # 1. 초기 컨텍스트 가져오기
        domain_analysis = state["context"].get("domain_analysis")
        financial_analysis = state["context"].get("financial_analysis")
        
        domain_context = ""
        if domain_analysis:
             domain_context = f"Domain Analysis Results:\n{json.dumps(domain_analysis, ensure_ascii=False, indent=2)}\n"
             
        financial_context = ""
        if financial_analysis:
             financial_context = f"Financial Context:\n{json.dumps(financial_analysis, ensure_ascii=False, indent=2)}\n"

        # 2. Planning 프롬프트 
        prompt = get_prompt(
            "planner",
            "planning",
            instruction=self.instruction,
            user_query=user_query,
            previous_plans="No previous plans context available.",
            current_time=current_time,
            sparkle_ideas=""
        )
        
        full_prompt = f"{domain_context}\n{financial_context}\n\n{prompt}"
        
        logger.info(f"[{self.name}] Calling LLM for base plan...")
        model_result = await execute_llm_task(
            prompt=full_prompt,
            task_type=TaskType.PLANNING,
            model_name=None,
            system_message=self.instruction,
        )
        plan = model_result.content or "No plan generated"
        
        # 3. Council 검토 (수동 또는 자동 활성화 확인)
        # 예시로 항상 검토 안하는 것으로 폴백을 주지만 실환경에선 Council 켤 수 있음.
        use_council = False
        try:
            from src.core.council_activator import get_council_activator
            activator = get_council_activator()
            decision = activator.should_activate(
                process_type="planning",
                query=user_query,
                context={}
            )
            use_council = decision.should_activate
        except Exception:
            pass
            
        if use_council:
            try:
                from src.core.llm_council import run_full_council
                logger.info(f"[{self.name}] 🏛️ Running Council review for research plan...")
                council_query = f"Review and improve research plan for: {user_query}\n\nPlan:\n{plan}"
                _, _, stage3_result, _ = await run_full_council(council_query)
                plan = stage3_result.get("response", plan)
                logger.info(f"[{self.name}] ✅ Council review completed.")
            except Exception as e:
                logger.warning(f"[{self.name}] Council review failed: {e}")

        # 4. Task Decomposition (기획된 계획을 서브 태스크로 분할)
        logger.info(f"[{self.name}] Splitting plan into tasks...")
        
        task_split_prompt = get_prompt(
            "planner",
            "task_decomposition",
            plan=plan,
            query=user_query,
            domain_analysis=json.dumps(domain_analysis or {}, ensure_ascii=False),
            current_time=current_time,
        )
        
        task_split_result = await execute_llm_task(
            prompt=task_split_prompt,
            task_type=TaskType.PLANNING,
            model_name=None,
            system_message="You are a task decomposition agent.",
        )
        
        tasks = self._parse_tasks(task_split_result.content, user_query)
        logger.info(f"[{self.name}] ✅ Genereated {len(tasks)} tasks.")
        
        # 구조 맞추기
        formatted_tasks = []
        for i, t in enumerate(tasks):
            formatted_tasks.append({
                "task_id": t.get("task_id", f"task_{i+1}"),
                "name": t.get("name", t.get("description", user_query)[:50]),
                "description": t.get("description", user_query),
                "task_type": t.get("task_type", "general"),
                "status": "pending"
            })
             
        return {
            "plan": plan,
            "tasks": formatted_tasks
        }

    def _parse_tasks(self, content: str, user_query: str) -> List[Dict[str, Any]]:
        if not content:
            return [{"task_id": "task_1", "description": user_query}]
            
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                return json.loads(json_match.group()).get("tasks", [])
            except json.JSONDecodeError:
                pass
                
        # 기본 폴백
        return [{"task_id": "task_1", "description": user_query}]

