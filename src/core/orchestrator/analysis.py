import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List

from src.core.llm_manager import TaskType, execute_llm_task
from src.core.orchestrator.base_node import BaseNode
from src.core.orchestrator.state import ResearchState
from src.core.streaming_manager import EventType

logger = logging.getLogger(__name__)


class AnalysisNode(BaseNode):
    """Handler for research objective analysis."""

    def __init__(self, creativity_agent, context_manager, streaming_manager, hybrid_storage):
        self.creativity_agent = creativity_agent
        self.context_manager = context_manager
        self.streaming_manager = streaming_manager
        self.hybrid_storage = hybrid_storage

    async def analyze_objectives(self, state: ResearchState) -> ResearchState:
        """목표 분석 (Multi-Model Orchestration + 재귀적 컨텍스트)."""
        # 입력 로깅
        self._log_node_input("analyze_objectives", state)

        logger.info("🔍 Thinking: Analyzing research objectives and requirements")
        logger.info(f"📝 Research Request: {state['user_request']}")

        # Sparkle 시드 아이디어 생성
        try:
            seed_insights = await self.creativity_agent.generate_seed_ideas(
                (state.get("user_request") or "").strip()
            )
            state["seed_ideas"] = [
                {
                    "insight_id": getattr(i, "insight_id", ""),
                    "type": getattr(getattr(i, "type", None), "value", "unknown"),
                    "title": getattr(i, "title", ""),
                    "description": getattr(i, "description", ""),
                    "reasoning": getattr(i, "reasoning", ""),
                    "related_concepts": getattr(i, "related_concepts", []),
                }
                for i in seed_insights
            ]
            logger.info(f"✨ Seed ideas (sparkle) generated: {len(seed_insights)}")
        except Exception as e:
            logger.warning(f"Sparkle seed ideas failed: {e}")
            state["seed_ideas"] = []

        # 초기 컨텍스트 생성 (재귀적 컨텍스트 사용)
        initial_context_data = {
            "user_request": state["user_request"],
            "context": state.get("context", {}),
            "objective_id": state.get("objective_id", ""),
            "stage": "analysis",
        }
        context_id = self.context_manager.push_context(
            context_data=initial_context_data,
            depth=0,
            parent_id=None,
            metadata={
                "node": "analyze_objectives",
                "timestamp": datetime.now().isoformat(),
            },
        )
        logger.debug(f"Initial context created: {context_id}")

        # 스트리밍 이벤트: 분석 시작
        await self.streaming_manager.stream_event(
            event_type=EventType.WORKFLOW_START,
            agent_id="orchestrator",
            workflow_id=state["objective_id"],
            data={
                "stage": "analysis",
                "message": "Starting objective analysis",
                "request": (
                    state["user_request"][:100] + "..."
                    if len(state["user_request"]) > 100
                    else state["user_request"]
                ),
            },
            priority=1,
        )

        analysis_prompt = f"""
        Analyze the following research request comprehensively:
        
        Request: {state["user_request"]}
        Context: {state.get("context", {})}
        
        Provide detailed analysis including:
        1. Intent analysis (what the user wants to achieve)
        2. Domain analysis (relevant fields and expertise areas)
        3. Scope analysis (breadth and depth of research needed)
        4. Complexity assessment (1-10 scale)
        5. Resource requirements and constraints
        6. Success criteria and quality metrics
        
        Use production-level analysis with specific, actionable insights.
        7. Categorize ideas into Pareto tiers: "Quick Wins", "High Impact", "Radical Innovation".
        Return the result in JSON format with the following structure:
        {{
            "objectives": [{{"id": "obj_1", "description": "Research objective", "priority": "high"}}],
            "intent": {{"primary": "research", "secondary": "analysis"}},
            "domain": {{"fields": ["technology", "research"], "expertise": "general"}},
            "scope": {{"breadth": "comprehensive", "depth": "detailed"}},
            "complexity": 7.0
            "pareto_frontier": {{"quick_wins": [], "high_impact": [], "radical_innovation": []}}
        }}
        """

        try:
            # Multi-Model Orchestration으로 분석 (JSON 파싱 실패 시 재시도)
            analysis_data = None
            last_error = None

            for attempt in range(3):
                try:
                    retry_hint = ""
                    if attempt > 0:
                        retry_hint = (
                            "\n\nIMPORTANT: Your previous response was not valid JSON. "
                            "You MUST return a valid JSON object with the exact structure specified above. "
                            "Do NOT include any text outside the JSON object.\n"
                        )

                    result = await execute_llm_task(
                        prompt=analysis_prompt + retry_hint,
                        task_type=TaskType.ANALYSIS,
                        system_message="You are an expert research analyst. Always respond with valid JSON.",
                    )

                    analysis_data = self._parse_analysis_result(result.content)
                    if analysis_data is not None:
                        logger.info(f"✅ Analysis completed using model: {result.model_used}")
                        logger.info(f"📊 Analysis confidence: {result.confidence}")
                        break

                    last_error = "LLM returned unparseable response"
                    logger.warning(f"Analysis parse attempt {attempt + 1}/3 failed, retrying...")

                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Analysis attempt {attempt + 1}/3 failed: {e}")

                if attempt < 2:
                    await asyncio.sleep(1.0 * (2**attempt))

            if analysis_data is None:
                raise RuntimeError(f"Analysis failed after 3 attempts. Last error: {last_error}")

            logger.info(f"🎯 Identified objectives: {len(analysis_data.get('objectives', []))}")
            logger.info(f"🧠 Complexity score: {analysis_data.get('complexity', 5.0)}")
            logger.info(f"🏷️ Domain: {analysis_data.get('domain', {}).get('fields', [])}")

            # 유사 연구 검색
            similar_research = await self._search_similar_research(
                state["user_request"], state.get("user_id", "default_user")
            )

            state.update(
                {
                    "analyzed_objectives": analysis_data.get("objectives", []),
                    "intent_analysis": analysis_data.get("intent", {}),
                    "domain_analysis": analysis_data.get("domain", {}),
                    "scope_analysis": analysis_data.get("scope", {}),
                    "complexity_score": analysis_data.get("complexity", 5.0),
                    "pareto_frontier": analysis_data.get("pareto_frontier", {}),
                    "current_step": "planning_agent",
                    "similar_research": similar_research,
                    "innovation_stats": {
                        "analysis_model": result.model_used,
                        "analysis_confidence": result.confidence,
                        "analysis_time": result.execution_time,
                    },
                }
            )

            # 스트리밍 이벤트: 분석 완료
            await self.streaming_manager.stream_event(
                event_type=EventType.AGENT_ACTION,
                agent_id="orchestrator",
                workflow_id=state["objective_id"],
                data={
                    "action": "analysis_completed",
                    "status": "completed",
                    "objectives_count": len(analysis_data.get("objectives", [])),
                    "complexity_score": analysis_data.get("complexity", 5.0),
                    "model_used": result.model_used,
                    "confidence": result.confidence,
                },
                priority=1,
            )

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            state["error_message"] = str(e)
            state["should_continue"] = False
            raise

        # 출력 로깅
        key_changes = {
            "analyzed_objectives": len(analysis_data.get("objectives", [])),
            "complexity_score": analysis_data.get("complexity", 5.0),
            "intent_analysis": analysis_data.get("intent", {}),
            "domain_analysis": analysis_data.get("domain", {}),
            "pareto_frontier": analysis_data.get("pareto_frontier", {}),
        }
        self._log_node_output("analyze_objectives", state, key_changes)

        return state

    def _parse_analysis_result(self, content: str) -> Dict[str, Any] | None:
        """분석 결과 파싱 — JSON이면 파싱, 실패 시 None 반환 (retry 유도)."""
        cleaned = (content or "").strip()

        # Markdown 코드 블록 제거
        md_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if md_match:
            cleaned = md_match.group(1).strip()

        # JSON 파싱 시도
        if cleaned.startswith("{"):
            try:
                parsed = json.loads(cleaned)
                # 최소 필수 필드 검증
                if "objectives" in parsed or "intent" in parsed or "complexity" in parsed or "pareto_frontier" in parsed:
                    return parsed
                logger.warning("Parsed JSON missing required fields")
                return None
            except json.JSONDecodeError:
                logger.warning("JSON decode failed in analysis result")
                return None

        logger.warning("Analysis result is not JSON")
        return None

    async def _search_similar_research(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """유사한 과거 연구를 검색합니다."""
        try:
            similar_research = await self.hybrid_storage.search_similar_research(
                query=query, user_id=user_id, limit=5, similarity_threshold=0.3
            )

            formatted_results = []
            for research in similar_research:
                formatted_results.append(
                    {
                        "research_id": research.research_id,
                        "topic": research.metadata.get("topic", ""),
                        "summary": research.summary,
                        "similarity_score": research.similarity_score,
                        "timestamp": research.timestamp.isoformat(),
                        "confidence_score": research.metadata.get("confidence_score", 0.0),
                    }
                )

            logger.info(f"Found {len(formatted_results)} similar research results")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search similar research: {e}")
            return []
