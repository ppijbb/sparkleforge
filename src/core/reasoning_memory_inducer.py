"""Reasoning Memory Inducer

성공, 실패, 병렬 궤적(Trajectory)으로부터 일반화 가능한 추론 메모리를 추출(Induce)하는 엔진입니다.
ReasoningBank의 induce_memory.py 및 induce_scaling.py를 기반으로 구현되었습니다.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, List

from src.core.llm_manager import TaskType, execute_llm_task
from src.core.prompts.reasoning_memory_prompts import (
    FAILED_TRAJECTORY_PROMPT,
    PARALLEL_CONTRAST_PROMPT,
    SUCCESSFUL_TRAJECTORY_PROMPT,
    TRAJECTORY_EVALUATION_PROMPT,
)
from src.core.reasoning_memory import ReasoningMemoryItem

logger = logging.getLogger(__name__)


class TrajectoryRecord:
    """단일 연구/작업 궤적 기록"""

    def __init__(self, query: str, domain: str = "general"):
        self.query = query
        self.domain = domain
        self.steps = []  # think, action, observation 쌍
        self.final_result = None
        self.status = "unknown"  # success, fail, unknown

    def add_step(self, think: str, action: Any, observation: Any = None):
        self.steps.append(
            {
                "think": think,
                "action": str(action),
                "observation": str(observation) if observation else "",
            }
        )

    def format_for_prompt(self) -> str:
        """프롬프트 주입을 위한 마크다운 포맷팅"""
        formatted = f"**Task Query:** {self.query}\n\n**Trajectory:**\n"
        for i, step in enumerate(self.steps):
            formatted += f"### Step {i+1}\n"
            formatted += f"<think>\n{step['think']}\n</think>\n"
            formatted += f"<action>\n{step['action']}\n</action>\n"
            if step["observation"]:
                formatted += f"<observation>\n{step['observation'][:500]}...\n</observation>\n"
            formatted += "\n"

        if self.final_result:
            formatted += f"**Final Result:**\n{self.final_result}\n"

        return formatted


class ReasoningMemoryInducer:
    """궤적으로부터 메모리를 추출하고 평가하는 유틸리티 클래스."""
    """
    Self-Reflective Knowledge Pruning & Distillation Daemon.

    Background daemon that distills raw research logs (trajectories) into
    compact, reusable procedural skills without human intervention. It prunes
    redundant/low-value induced memories and consolidates overlapping items
    into distilled procedural skills.
    """

    @staticmethod
    def _parse_memory_items(
        markdown_text: str, source_query: str, status: str, domain: str
    ) -> List[ReasoningMemoryItem]:
        """LLM이 생성한 마크다운을 ReasoningMemoryItem 리스트로 파싱합니다."""
        items = []
        current_item = {}

        for line in markdown_text.split("\n"):
            line = line.strip()
            if line.startswith("# Memory Item"):
                if current_item and "title" in current_item and "content" in current_item:
                    items.append(current_item)
                current_item = {"title": "", "description": "", "content": ""}
            elif line.startswith("## Title"):
                current_item["title"] = line.replace("## Title", "").strip()
            elif line.startswith("## Description"):
                current_item["description"] = line.replace("## Description", "").strip()
            elif line.startswith("## Content"):
                current_item["content"] = line.replace("## Content", "").strip()
            elif current_item and "content" in current_item and line and not line.startswith("#"):
                # Handle multi-line content
                current_item["content"] += " " + line

        # Add the last item
        if current_item and "title" in current_item and "content" in current_item:
            items.append(current_item)

        # Convert to ReasoningMemoryItem objects
        memory_objects = []
        for item in items:
            mem = ReasoningMemoryItem(
                memory_id=f"mem_reasoning_{uuid.uuid4().hex[:8]}",
                title=item["title"],
                description=item["description"],
                content=item["content"],
                trajectory_status=status,
                task_query=source_query,
                domain=domain,
                created_at=datetime.now().isoformat(),
            )
            memory_objects.append(mem)

        return memory_objects

    async def auto_evaluate_trajectory(self, trajectory: TrajectoryRecord) -> str:
        """LLM-as-a-judge 방식을 사용하여 궤적이 성공했는지 실패했는지 평가합니다."""
        prompt = f"**User Query:** {trajectory.query}\n\n**Trajectory:**\n{trajectory.format_for_prompt()}"

        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.SYNTHESIS,
                system_message=TRAJECTORY_EVALUATION_PROMPT,
                temperature=0.1,
            )

            # JSON 파싱 시도
            try:
                import re

                json_match = re.search(r"\{.*\}", result.content, re.DOTALL)
                if json_match:
                    eval_data = json.loads(json_match.group(0))
                    status = eval_data.get("status", "unknown")
                    if status in ["success", "fail"]:
                        trajectory.status = status
                        return status
            except Exception as json_e:
                logger.warning(f"Failed to parse AutoEval JSON: {json_e}")

            # Fallback text matching
            content_lower = result.content.lower()
            if "success" in content_lower and "fail" not in content_lower:
                trajectory.status = "success"
                return "success"
            elif "fail" in content_lower:
                trajectory.status = "fail"
                return "fail"

        except Exception as e:
            logger.error(f"Auto evaluation failed: {e}")

        return "unknown"

    async def induce_from_success(self, trajectory: TrajectoryRecord) -> List[ReasoningMemoryItem]:
        """성공한 궤적에서 재활용 가능한 전략을 추출합니다."""
        if trajectory.status == "unknown":
            trajectory.status = await self.auto_evaluate_trajectory(trajectory)

        # 실패했다면 fail 로직으로 폴백
        if trajectory.status == "fail":
            return await self.induce_from_failure(trajectory)

        user_prompt = f"Extract memory items from the following successful trajectory:\n\n{trajectory.format_for_prompt()}"

        try:
            result = await execute_llm_task(
                prompt=user_prompt,
                task_type=TaskType.DEEP_REASONING,
                system_message=SUCCESSFUL_TRAJECTORY_PROMPT,
                temperature=0.7,  # 창의성을 위해 약간 높임
            )

            items = self._parse_memory_items(
                result.content, trajectory.query, "success", trajectory.domain
            )
            logger.info(
                f"Induced {len(items)} successful memories for query: '{trajectory.query[:30]}...'"
            )
            return items

        except Exception as e:
            logger.error(f"Failed to induce successful memory: {e}")
            return []

    async def induce_from_failure(self, trajectory: TrajectoryRecord) -> List[ReasoningMemoryItem]:
        """실패한 궤적에서 회피/복구 전략을 추출합니다."""
        if trajectory.status == "unknown":
            trajectory.status = await self.auto_evaluate_trajectory(trajectory)

        # 성공했다면 success 로직으로 폴백
        if trajectory.status == "success":
            return await self.induce_from_success(trajectory)

        user_prompt = f"Extract memory items from the following failed trajectory:\n\n{trajectory.format_for_prompt()}"

        try:
            result = await execute_llm_task(
                prompt=user_prompt,
                task_type=TaskType.DEEP_REASONING,
                system_message=FAILED_TRAJECTORY_PROMPT,
                temperature=0.7,
            )

            items = self._parse_memory_items(
                result.content, trajectory.query, "fail", trajectory.domain
            )
            logger.info(
                f"Induced {len(items)} failure-avoidance memories for query: '{trajectory.query[:30]}...'"
            )
            return items

        except Exception as e:
            logger.error(f"Failed to induce failure memory: {e}")
            return []

    async def induce_from_parallel(
        self, trajectories: List[TrajectoryRecord], query: str, domain: str = "general"
    ) -> List[ReasoningMemoryItem]:
        """성공/실패가 섞인 여러 병렬 궤적을 대조(Self-Contrast)하여 인사이트를 추출합니다."""
        if len(trajectories) < 2:
            if len(trajectories) == 1:
                if trajectories[0].status == "success":
                    return await self.induce_from_success(trajectories[0])
                else:
                    return await self.induce_from_failure(trajectories[0])
            return []

        combined_prompt = f"**User Query:** {query}\n\n"

        for i, traj in enumerate(trajectories):
            status_text = traj.status.upper() if traj.status != "unknown" else "UNKNOWN STATUS"
            combined_prompt += f"## Trajectory {i+1} ({status_text})\n"
            combined_prompt += traj.format_for_prompt() + "\n\n"

        try:
            result = await execute_llm_task(
                prompt=combined_prompt,
                task_type=TaskType.DEEP_REASONING,
                system_message=PARALLEL_CONTRAST_PROMPT,
                temperature=0.7,
            )

            items = self._parse_memory_items(result.content, query, "parallel_contrast", domain)
            logger.info(
                f"Induced {len(items)} parallel contrast memories for query: '{query[:30]}...'"
            )
            return items

        except Exception as e:
            logger.error(f"Failed to induce parallel contrast memory: {e}")
            return []


# 전역 싱글톤 인스턴스
_reasoning_memory_inducer = None


def get_reasoning_memory_inducer() -> ReasoningMemoryInducer:
    global _reasoning_memory_inducer
    if _reasoning_memory_inducer is None:
        _reasoning_memory_inducer = ReasoningMemoryInducer()
    return _reasoning_memory_inducer
