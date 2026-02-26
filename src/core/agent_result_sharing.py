#!/usr/bin/env python3
"""Agent Result Sharing System

병렬 실행 중 agent들이 결과를 공유하고, 결과에 대해 토론할 수 있는 시스템.
Filesystem Context: 대규모 결과는 파일 시스템에 저장하고 참조만 공유 (옵션).
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.core.llm_manager import TaskType, execute_llm_task
from src.core.researcher_config import get_agent_config, get_llm_config

logger = logging.getLogger(__name__)

# workspace/agents 디렉터리 (Sub-Agent Communication via Filesystem)
def _get_shared_results_workspace_root() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    return root / "workspace" / "shared_results"


def _serialize_result_size(result: Any) -> int:
    """결과 직렬화 시 대략적인 바이트 크기."""
    try:
        return len(json.dumps(result, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(str(result))


@dataclass
class SharedResult:
    """공유된 결과."""

    task_id: str
    agent_id: str
    result: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0


@dataclass
class DiscussionMessage:
    """토론 메시지."""

    agent_id: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    related_task_id: str | None = None
    related_result_id: str | None = None


class SharedResultsManager:
    """Agent 간 결과 공유 관리자.

    use_filesystem_for_large_results=True 이면 대규모 result를 파일에 저장하고
    메모리에는 참조(_fs_ref)와 요약만 보관 (Sub-Agent Communication via Filesystem).
    """

    def __init__(
        self,
        objective_id: str,
        use_filesystem_for_large_results: bool | None = None,
        large_result_threshold_chars: int = 10000,
    ):
        """초기화.

        Args:
            objective_id: 목표 ID
            use_filesystem_for_large_results: None이면 env ENABLE_SHARED_RESULTS_FS 사용
            large_result_threshold_chars: 이 크기 초과 시 파일로 저장
        """
        self.objective_id = objective_id
        self.shared_results: Dict[str, SharedResult] = {}
        self.task_results: Dict[str, List[str]] = defaultdict(list)
        self.agent_results: Dict[str, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()

        if use_filesystem_for_large_results is None:
            use_filesystem_for_large_results = (
                os.getenv("ENABLE_SHARED_RESULTS_FS", "false").lower() == "true"
            )
        self.use_filesystem_for_large_results = use_filesystem_for_large_results
        self.large_result_threshold_chars = large_result_threshold_chars

        logger.info(
            f"SharedResultsManager initialized for objective: {objective_id} "
            f"(fs_large_results={use_filesystem_for_large_results})"
        )

    def _write_result_to_fs(self, result_id: str, agent_id: str, result: Any) -> Dict[str, Any]:
        """대규모 결과를 workspace/shared_results에 쓰고 참조 dict 반환."""
        try:
            root = _get_shared_results_workspace_root()
            root.mkdir(parents=True, exist_ok=True)
            safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in result_id)
            path = root / f"{self.objective_id}_{safe_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            summary = json.dumps(result, ensure_ascii=False)[:500] + ("..." if _serialize_result_size(result) > 500 else "")
            return {
                "_fs_ref": str(path),
                "summary": summary,
                "instruction": "Full result stored in file. Use read_file to load when needed.",
            }
        except Exception as e:
            logger.warning("SharedResultsManager: failed to write result to fs: %s", e)
            return result

    async def share_result(
        self,
        task_id: str,
        agent_id: str,
        result: Any,
        metadata: Dict[str, Any] | None = None,
        confidence: float = 0.0,
    ) -> str:
        """결과 공유. 대규모 결과는 옵션에 따라 파일 시스템에 저장."""
        async with self._lock:
            result_id = f"{task_id}_{agent_id}_{datetime.now().timestamp()}"

            payload = result
            if self.use_filesystem_for_large_results and _serialize_result_size(result) > self.large_result_threshold_chars:
                payload = self._write_result_to_fs(result_id, agent_id, result)

            shared_result = SharedResult(
                task_id=task_id,
                agent_id=agent_id,
                result=payload,
                metadata=metadata or {},
                confidence=confidence,
            )

            self.shared_results[result_id] = shared_result
            self.task_results[task_id].append(result_id)
            self.agent_results[agent_id].append(result_id)

            logger.info(
                f"Result shared: {result_id} by agent {agent_id} for task {task_id}"
            )

            return result_id

    @staticmethod
    def load_result_from_fs(shared_result: "SharedResult") -> Any:
        """SharedResult.result가 _fs_ref이면 파일에서 전체 로드하여 반환."""
        r = shared_result.result
        if isinstance(r, dict) and r.get("_fs_ref"):
            try:
                with open(r["_fs_ref"], encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load result from fs: %s", e)
        return r

    async def get_shared_results(
        self,
        task_id: str | None = None,
        agent_id: str | None = None,
        exclude_agent_id: str | None = None,
    ) -> List[SharedResult]:
        """공유된 결과 조회."""
        async with self._lock:
            results = []

            if task_id:
                # 특정 작업의 결과만
                result_ids = self.task_results.get(task_id, [])
                for result_id in result_ids:
                    if result_id in self.shared_results:
                        result = self.shared_results[result_id]
                        if exclude_agent_id and result.agent_id == exclude_agent_id:
                            continue
                        results.append(result)
            elif agent_id:
                # 특정 agent의 결과만
                result_ids = self.agent_results.get(agent_id, [])
                for result_id in result_ids:
                    if result_id in self.shared_results:
                        results.append(self.shared_results[result_id])
            else:
                # 모든 결과
                for result in self.shared_results.values():
                    if exclude_agent_id and result.agent_id == exclude_agent_id:
                        continue
                    results.append(result)

            return results

    async def get_result_summary(self, task_id: str | None = None) -> Dict[str, Any]:
        """결과 요약."""
        results = await self.get_shared_results(task_id=task_id)

        if not results:
            return {
                "total_results": 0,
                "agents_count": 0,
                "average_confidence": 0.0,
                "results": [],
            }

        agents = set(r.agent_id for r in results)
        avg_confidence = (
            sum(r.confidence for r in results) / len(results) if results else 0.0
        )

        return {
            "total_results": len(results),
            "agents_count": len(agents),
            "average_confidence": avg_confidence,
            "results": [
                {
                    "result_id": result_id,
                    "agent_id": r.agent_id,
                    "task_id": r.task_id,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp.isoformat(),
                }
                for result_id, r in self.shared_results.items()
                if r in results
            ],
        }


class AgentDiscussionManager:
    """Agent 간 토론 관리자."""

    def __init__(self, objective_id: str, shared_results_manager: SharedResultsManager):
        """초기화."""
        self.objective_id = objective_id
        self.shared_results_manager = shared_results_manager
        self.discussions: Dict[str, List[DiscussionMessage]] = defaultdict(
            list
        )  # topic -> [messages]
        self.agent_config = get_agent_config()
        self.llm_config = get_llm_config()
        self._lock = asyncio.Lock()

        logger.info(f"AgentDiscussionManager initialized for objective: {objective_id}")

    async def start_discussion(
        self, topic: str, initial_result_id: str, agent_id: str, initial_message: str
    ) -> str:
        """토론 시작."""
        async with self._lock:
            message = DiscussionMessage(
                agent_id=agent_id,
                message=initial_message,
                related_result_id=initial_result_id,
            )

            self.discussions[topic].append(message)

            logger.info(f"Discussion started on topic '{topic}' by agent {agent_id}")

            return topic

    async def add_message(
        self,
        topic: str,
        agent_id: str,
        message: str,
        related_result_id: str | None = None,
    ) -> None:
        """토론에 메시지 추가."""
        async with self._lock:
            discussion_message = DiscussionMessage(
                agent_id=agent_id, message=message, related_result_id=related_result_id
            )

            self.discussions[topic].append(discussion_message)

            logger.info(f"Message added to discussion '{topic}' by agent {agent_id}")

    async def get_discussion(self, topic: str) -> List[DiscussionMessage]:
        """토론 내용 조회."""
        async with self._lock:
            return self.discussions.get(topic, []).copy()

    @staticmethod
    def _weighted_consensus(
        shared_result: SharedResult,
        other_agent_results: List[SharedResult],
    ) -> Dict[str, Any]:
        """Confidence 기반 Weighted Voting으로 명시적 합의 도출 (Multi-Agent Patterns)."""
        all_results = [shared_result] + list(other_agent_results)
        if not all_results:
            return {"consensus_score": 0.0, "weighted_summary": "", "participant_weights": []}

        total_weight = sum(max(0.0, r.confidence) for r in all_results) or 1.0
        participant_weights = [
            {"agent_id": r.agent_id, "confidence": r.confidence, "weight": max(0.0, r.confidence) / total_weight}
            for r in all_results
        ]
        consensus_score = sum(r.confidence for r in all_results) / len(all_results) if all_results else 0.0
        best = max(all_results, key=lambda r: r.confidence)
        try:
            weighted_summary = json.dumps(best.result, ensure_ascii=False)[:500]
        except (TypeError, ValueError):
            weighted_summary = str(best.result)[:500]
        return {
            "consensus_score": round(consensus_score, 4),
            "weighted_summary": weighted_summary,
            "participant_weights": participant_weights,
            "leading_agent_id": best.agent_id,
            "leading_confidence": best.confidence,
        }

    async def agent_discuss_result(
        self,
        result_id: str,
        agent_id: str,
        other_agent_results: List[SharedResult],
        discussion_type: str = "general",
    ) -> Dict[str, Any]:
        """Agent가 다른 agent의 결과에 대해 논박 (debate).

        Returns:
            Dict with 'message', 'consistency_check', 'logical_validity', 'consensus'
        """
        if not self.agent_config.enable_agent_communication:
            logger.debug("Agent communication is disabled")
            return {}

        if not other_agent_results:
            logger.debug("No other agent results to discuss")
            return {}

        # 공유된 결과 가져오기
        shared_result = None
        for result_id_key, result in self.shared_results_manager.shared_results.items():
            if result_id_key == result_id:
                shared_result = result
                break

        if not shared_result:
            logger.warning(f"Result {result_id} not found for discussion")
            return {}

        # 논박 타입에 따른 프롬프트 선택
        if discussion_type == "verification":
            system_message = "You are a verification agent that critically evaluates research results for accuracy, consistency, and logical validity."
            discussion_prompt = f"""
You are a verification agent participating in a critical debate about research results.

Your agent ID: {agent_id}
Result to verify: {json.dumps(shared_result.result, ensure_ascii=False, indent=2)[:1000]}
Your confidence: {shared_result.confidence}

Other agents' results and perspectives:
{json.dumps([{"agent_id": r.agent_id, "result": r.result, "confidence": r.confidence} for r in other_agent_results[:5]], ensure_ascii=False, indent=2)[:2000]}

**CRITICAL DEBATE REQUIREMENTS:**

1. **Consistency Check (일관성 검증)**:
   - Are the results consistent across different agents?
   - Are there contradictions or conflicts?
   - What are the points of agreement and disagreement?

2. **Logical Validity (논리적 올바름 검증)**:
   - Is the logic sound? Are there logical fallacies?
   - Do the conclusions follow from the evidence?
   - Are the arguments valid and well-reasoned?

3. **Critical Analysis (비판적 분석)**:
   - What are the strengths and weaknesses of each perspective?
   - What assumptions underlie each result?
   - What evidence supports or contradicts each view?

4. **Consensus Building (합의 도출)**:
   - Based on the debate, what is the consensus?
   - What points are agreed upon?
   - What points remain in dispute?

Provide a structured response:
1. Consistency assessment (consistent/inconsistent/partially_consistent)
2. Logical validity assessment (valid/invalid/partially_valid)
3. Critical analysis of each perspective
4. Consensus summary
5. Remaining disagreements (if any)

Be thorough and critical - this is a debate, not just a comparison.
"""
        elif discussion_type == "evaluation":
            system_message = "You are an evaluation agent that assesses research quality, completeness, and value through critical debate."
            discussion_prompt = f"""
You are an evaluation agent participating in a critical debate about research results quality.

Your agent ID: {agent_id}
Result to evaluate: {json.dumps(shared_result.result, ensure_ascii=False, indent=2)[:1000]}
Your confidence: {shared_result.confidence}

Other agents' evaluations:
{json.dumps([{"agent_id": r.agent_id, "result": r.result, "confidence": r.confidence} for r in other_agent_results[:5]], ensure_ascii=False, indent=2)[:2000]}

**CRITICAL EVALUATION DEBATE:**

1. **Quality Assessment (품질 평가)**:
   - How does the quality compare across different evaluations?
   - Are there quality gaps or inconsistencies?
   - What are the quality strengths and weaknesses?

2. **Completeness Check (완전성 검증)**:
   - Is the evaluation complete? What's missing?
   - Are all important aspects covered?
   - What additional perspectives are needed?

3. **Value Assessment (가치 평가)**:
   - What is the value of each evaluation?
   - Which insights are most valuable?
   - What recommendations are most actionable?

4. **Consensus on Quality (품질 합의)**:
   - What is the consensus on overall quality?
   - What are the agreed-upon strengths?
   - What are the agreed-upon weaknesses?

Provide a structured response with quality assessment, completeness check, value assessment, and consensus.
"""
        else:
            system_message = "You are a collaborative research agent that provides constructive feedback through critical debate."
        discussion_prompt = f"""
You are an AI research agent participating in a collaborative debate about research results.

Your agent ID: {agent_id}
Your result: {json.dumps(shared_result.result, ensure_ascii=False, indent=2)[:1000]}
Your confidence: {shared_result.confidence}

Other agents' results:
{json.dumps([{"agent_id": r.agent_id, "result": r.result, "confidence": r.confidence} for r in other_agent_results[:5]], ensure_ascii=False, indent=2)[:2000]}

**DEBATE REQUIREMENTS:**

1. **Critical Comparison (비판적 비교)**:
   - How does your result compare to other agents' results?
   - What are the key differences and similarities?
   - What are the strengths and weaknesses of each perspective?

2. **Consistency Analysis (일관성 분석)**:
   - Are the results consistent? Where do they agree/disagree?
   - What explains the differences?

3. **Logical Validity (논리적 올바름)**:
   - Are the arguments logically sound?
   - Do conclusions follow from evidence?

4. **Insights and Recommendations (통찰 및 권장사항)**:
   - What insights emerge from the debate?
   - What improvements or clarifications are needed?

Provide a structured response with comparison, consistency analysis, logical validity check, and insights.
"""

        try:
            llm_result = await execute_llm_task(
                prompt=discussion_prompt,
                task_type=TaskType.ANALYSIS,
                system_message=system_message,
            )

            discussion_message = (
                llm_result.content
                if hasattr(llm_result, "content")
                else str(llm_result)
            )

            # Weighted Voting / 명시적 합의 도출
            weighted = self._weighted_consensus(shared_result, other_agent_results)

            # 논박 결과 파싱 (구조화된 응답 추출 시도)
            discussion_result = {
                "message": discussion_message,
                "agent_id": agent_id,
                "result_id": result_id,
                "discussion_type": discussion_type,
                "timestamp": datetime.now().isoformat(),
                "weighted_consensus": weighted,
                "consensus_score": weighted.get("consensus_score", 0.0),
            }

            # 일관성 및 논리적 올바름 추출 시도
            consistency_keywords = [
                "consistent",
                "inconsistent",
                "contradiction",
                "agreement",
                "disagreement",
            ]
            logical_keywords = [
                "logical",
                "valid",
                "invalid",
                "fallacy",
                "sound",
                "reasoning",
            ]

            discussion_lower = discussion_message.lower()
            consistency_check = "unknown"
            logical_validity = "unknown"

            if any(
                kw in discussion_lower for kw in ["consistent", "agreement", "agree"]
            ):
                if any(
                    kw in discussion_lower
                    for kw in ["inconsistent", "contradiction", "disagreement"]
                ):
                    consistency_check = "partially_consistent"
                else:
                    consistency_check = "consistent"
            elif any(
                kw in discussion_lower
                for kw in ["inconsistent", "contradiction", "disagreement"]
            ):
                consistency_check = "inconsistent"

            if any(
                kw in discussion_lower
                for kw in ["logical", "valid", "sound", "reasoning"]
            ):
                if any(
                    kw in discussion_lower for kw in ["invalid", "fallacy", "illogical"]
                ):
                    logical_validity = "partially_valid"
                else:
                    logical_validity = "valid"
            elif any(
                kw in discussion_lower for kw in ["invalid", "fallacy", "illogical"]
            ):
                logical_validity = "invalid"

            discussion_result["consistency_check"] = consistency_check
            discussion_result["logical_validity"] = logical_validity

            # 토론에 메시지 추가
            topic = f"result_debate_{result_id}_{discussion_type}"
            await self.add_message(
                topic=topic,
                agent_id=agent_id,
                message=discussion_message,
                related_result_id=result_id,
            )

            logger.info(
                f"Agent {agent_id} debated result {result_id} (type: {discussion_type}, consistency: {consistency_check}, validity: {logical_validity})"
            )

            return discussion_result

        except Exception as e:
            logger.error(f"Failed to generate discussion message: {e}")
            return {}

    async def get_all_discussions_for_result(
        self, result_id: str
    ) -> List[Dict[str, Any]]:
        """특정 결과에 대한 모든 논박 내용 조회."""
        async with self._lock:
            all_discussions = []
            for topic, messages in self.discussions.items():
                if result_id in topic:
                    for msg in messages:
                        all_discussions.append(
                            {
                                "agent_id": msg.agent_id,
                                "message": msg.message,
                                "timestamp": msg.timestamp.isoformat(),
                                "topic": topic,
                            }
                        )
            return all_discussions

    async def get_discussion_summary(self, topic: str | None = None) -> Dict[str, Any]:
        """토론 요약."""
        async with self._lock:
            if topic:
                discussions = {topic: self.discussions.get(topic, [])}
            else:
                discussions = dict(self.discussions)

            summary = {"total_topics": len(discussions), "topics": {}}

            for topic_name, messages in discussions.items():
                agents = set(msg.agent_id for msg in messages)
                summary["topics"][topic_name] = {
                    "message_count": len(messages),
                    "participating_agents": list(agents),
                    "last_message_time": messages[-1].timestamp.isoformat()
                    if messages
                    else None,
                }

            return summary
