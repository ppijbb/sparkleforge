"""Core ResearchAgent class: init, LLM plumbing, and task lifecycle.

Split out of the former monolithic research_agent.py (issue #582, mirroring
the Sigma-1 split of mcp_integration.py -- module by responsibility, facade
re-export kept at src/agents/research_agent/__init__.py). Composes the other
mixins in this package to form the full ResearchAgent surface.
"""

import asyncio
import json
import os
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.research_agent.browser import BrowserAutomationMixin
from src.agents.research_agent.data_collection import DataCollectionMixin
from src.agents.research_agent.quality_metrics import QualityMetricsMixin
from src.agents.research_agent.search_providers import SearchProvidersMixin
from src.agents.research_agent.task_pipelines import TaskPipelinesMixin
from src.core.researcher_config import get_llm_config, get_research_config
from src.core.skills.prompts.agents.research_agent import research_agent_prompts
from src.utils.logger import setup_logger

logger = setup_logger("research_agent", log_level="INFO")


class ResearchAgent(
    SearchProvidersMixin,
    DataCollectionMixin,
    QualityMetricsMixin,
    TaskPipelinesMixin,
    BrowserAutomationMixin,
):
    """Autonomous research agent for data collection and analysis."""

    def __init__(self):
        """Initialize the research agent."""
        # Load configurations
        self.llm_config = get_llm_config()
        self.research_config = get_research_config()

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Active research tasks
        self.active_tasks: Dict[str, Dict[str, Any]] = {}

        # Prompt registry consumed by the analysis/synthesis/validation
        # pipeline methods in task_pipelines.py (issue #790).
        self.config = type("ResearchAgentConfig", (), {"prompts": research_agent_prompts})

        # Adaptive strategy state mutated by update_capabilities() based on
        # evaluation feedback (issue #790).
        self.adaptive_strategies = {
            "search_depth": 3,
            "content_length": 5000,
            "source_diversity": 0.7,
        }
        self.analysis_methods = self._load_analysis_methods()

        # Learning capabilities
        self.learning_data = []
        self.research_history = []

        # Enhanced browser automation (optional, initialized when needed)
        self.browser_manager = None

        logger.info("Research Agent initialized with LLM-based research capabilities")


    def _initialize_llm(self):
        """Initialize the LLM client."""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("Gemini API key not found. Research functionality will be limited.")
                return None

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-3.5-flash-lite")
            logger.info(
                "LLM initialized for ResearchAgent with model: gemini-3.5-flash-lite"
            )
            return model
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise


    async def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with retry logic and rate limiting."""
        if not self.llm:
            return "LLM not available"

        for attempt in range(max_retries):
            try:
                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = random.uniform(2, 5) * (attempt + 1)
                    logger.info(
                        f"Retrying LLM call after {delay:.2f}s delay (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                else:
                    # Small delay even on first attempt
                    await asyncio.sleep(random.uniform(0.5, 1.5))

                response = self.llm.generate_content(prompt)
                return response.text.strip()

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Extract retry delay from error if available
                        retry_delay = 30 + (attempt * 10)  # Progressive delay
                        logger.warning(
                            f"API quota exceeded, retrying in {retry_delay}s (attempt {attempt + 1})"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                        return f"LLM call failed: {e}"
                else:
                    logger.error(f"LLM call failed: {e}")
                    return f"LLM call failed: {e}"

        return "LLM call failed after all retries"


    def _load_research_tools(self) -> Dict[str, Any]:
        """Load available research tools.

        Returns:
            Dictionary of research tools
        """
        return {
            "web_search": {
                "capabilities": ["web_search", "content_extraction", "url_analysis"],
                "max_concurrent": 5,
                "rate_limit": 10,  # requests per minute
            },
            "academic_search": {
                "capabilities": [
                    "academic_papers",
                    "citation_analysis",
                    "literature_review",
                ],
                "max_concurrent": 3,
                "rate_limit": 5,
            },
            "data_analysis": {
                "capabilities": [
                    "statistical_analysis",
                    "data_visualization",
                    "trend_analysis",
                ],
                "max_concurrent": 2,
                "rate_limit": 0,  # no rate limit
            },
            "content_analysis": {
                "capabilities": [
                    "text_analysis",
                    "sentiment_analysis",
                    "topic_modeling",
                ],
                "max_concurrent": 3,
                "rate_limit": 0,
            },
        }


    def _load_data_sources(self) -> Dict[str, Any]:
        """Load available data sources.

        Returns:
            Dictionary of data sources
        """
        return {
            "web_sources": {
                "types": ["news_articles", "blog_posts", "reports", "websites"],
                "reliability": 0.7,
                "accessibility": 0.9,
            },
            "academic_sources": {
                "types": [
                    "research_papers",
                    "conference_proceedings",
                    "theses",
                    "books",
                ],
                "reliability": 0.9,
                "accessibility": 0.6,
            },
            "data_sources": {
                "types": ["datasets", "databases", "apis", "statistics"],
                "reliability": 0.8,
                "accessibility": 0.7,
            },
            "expert_sources": {
                "types": ["interviews", "expert_opinions", "case_studies"],
                "reliability": 0.85,
                "accessibility": 0.5,
            },
        }


    def _load_analysis_methods(self) -> Dict[str, Any]:
        """Load available analysis methods.

        Returns:
            Dictionary of analysis methods
        """
        return {
            "quantitative": {
                "methods": [
                    "statistical_analysis",
                    "regression_analysis",
                    "correlation_analysis",
                ],
                "suitable_for": ["numerical_data", "surveys", "experiments"],
                "outputs": ["statistics", "charts", "models"],
            },
            "qualitative": {
                "methods": [
                    "content_analysis",
                    "thematic_analysis",
                    "discourse_analysis",
                ],
                "suitable_for": ["text_data", "interviews", "observations"],
                "outputs": ["themes", "insights", "narratives"],
            },
            "mixed_methods": {
                "methods": [
                    "triangulation",
                    "convergent_analysis",
                    "explanatory_analysis",
                ],
                "suitable_for": ["complex_research", "comprehensive_studies"],
                "outputs": ["integrated_findings", "comprehensive_reports"],
            },
        }


    async def conduct_research(
        self,
        tasks: List[Dict[str, Any]],
        context: Dict[str, Any] | None = None,
        objective_id: str = None,
    ) -> Dict[str, Any]:
        """Conduct comprehensive research for multiple tasks.

        Args:
            tasks: List of research tasks
            context: Additional context
            objective_id: Objective ID for tracking

        Returns:
            Comprehensive research result
        """
        try:
            logger.info(f"Conducting research for {len(tasks)} tasks")

            research_results = []
            successful_tasks = 0
            failed_tasks = 0

            # Execute each task
            for task in tasks:
                try:
                    result = await self.execute_task(task, objective_id)
                    research_results.append(result)

                    if result.get("success", False):
                        successful_tasks += 1
                    else:
                        failed_tasks += 1

                except Exception as e:
                    logger.error(f"Task execution failed: {e}")
                    research_results.append(
                        {
                            "success": False,
                            "error": str(e),
                            "task_id": task.get("task_id"),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    failed_tasks += 1

            # Generate comprehensive research summary
            summary = await self._generate_research_summary(
                research_results,
                {
                    "total_tasks": len(tasks),
                    "successful_tasks": successful_tasks,
                    "failed_tasks": failed_tasks,
                },
            )

            return {
                "success": successful_tasks > 0,
                "research_results": research_results,
                "summary": summary,
                "statistics": {
                    "total_tasks": len(tasks),
                    "successful_tasks": successful_tasks,
                    "failed_tasks": failed_tasks,
                    "success_rate": successful_tasks / len(tasks) if tasks else 0,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Research conduction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "research_results": [],
                "timestamp": datetime.now().isoformat(),
            }


    async def execute_task(
        self,
        task: Dict[str, Any],
        objective_id: str,
        is_refinement: bool = False,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Execute a research task with LLM-based research.

        Args:
            task: Task to execute
            objective_id: Objective ID for tracking
            is_refinement: Whether this is a refinement task

        Returns:
            Task execution result
        """
        try:
            task_id = task.get("task_id", str(uuid.uuid4()))
            task_type = task.get("task_type", "general")

            logger.info(f"Executing LLM-based research task: {task_id} (type: {task_type})")

            # Track active task
            self.active_tasks[task_id] = {
                "task": task,
                "objective_id": objective_id,
                "started_at": datetime.now(),
                "status": "running",
            }

            # Use LLM to plan and execute research
            result = await self._llm_conduct_research(task, objective_id)

            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = datetime.now()

            # Add metadata
            result.update(
                {
                    "task_id": task_id,
                    "objective_id": objective_id,
                    "agent": "researcher",
                    "task_type": task_type,
                    "is_refinement": is_refinement,
                    "execution_time": (
                        datetime.now() - self.active_tasks[task_id]["started_at"]
                    ).total_seconds(),
                    "status": "completed",
                }
            )

            # Store in research history
            self.research_history.append(
                {
                    "task_id": task_id,
                    "objective_id": objective_id,
                    "task_type": task_type,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger.info(f"LLM-based research task completed: {task_id}")
            return result

        except Exception as e:
            logger.error(f"LLM-based research task execution failed: {e}")

            # Update task status
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["error"] = str(e)

            return {
                "task_id": task_id,
                "objective_id": objective_id,
                "agent": "researcher",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def execute_research_task(
        self,
        task: Dict[str, Any],
        objective_id: str,
        is_refinement: bool = False,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Alias for execute_task(), the name ResearchOperator calls (issue #790)."""
        return await self.execute_task(task, objective_id, is_refinement, context)


    async def _llm_conduct_research(
        self, task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Use LLM to conduct comprehensive research."""
        try:
            # Get LLM research plan
            research_plan = await self._get_llm_research_plan(task)

            # Execute research based on plan
            research_results = []
            research_steps = research_plan.get("steps", [])
            if not isinstance(research_steps, list):
                research_steps = []

            for research_step in research_steps:
                if isinstance(research_step, dict):
                    step_result = await self._execute_research_step(research_step, task)
                    if step_result:
                        research_results.append(step_result)

            # Use LLM to analyze and synthesize results
            analysis_result = await self._llm_analyze_research_results(research_results, task)

            # Ensure analysis_result is a dict
            if not isinstance(analysis_result, dict):
                analysis_result = {"quality_score": 0.0, "key_findings": []}

            return {
                "research_plan": research_plan,
                "research_results": research_results,
                "analysis_result": analysis_result,
                "total_sources": len(research_results),
                "research_quality": analysis_result.get("quality_score", 0.0),
                "key_findings": analysis_result.get("key_findings", []),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"LLM research failed: {e}")
            # Return error result instead of fallback
            return {
                "research_plan": {},
                "research_results": [],
                "analysis_result": {"quality_score": 0.0, "key_findings": []},
                "total_sources": 0,
                "research_quality": 0.0,
                "key_findings": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


    async def _get_llm_research_plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Get research plan from LLM."""
        try:
            prompt = f"""
            다음 연구 작업을 위한 구체적인 연구 계획을 수립하세요.
            
            작업: {task.get("description", "")}
            작업 유형: {task.get("task_type", "general")}
            
            다음을 포함한 연구 계획을 수립하세요:
            1. 필요한 정보 유형 식별
            2. 적절한 정보원 선택
            3. 검색 전략 수립
            4. 데이터 수집 방법 결정
            5. 분석 방법 선택
            
            JSON 형태로 응답하세요:
            {{
                "research_strategy": "전체 연구 전략",
                "steps": [
                    {{
                        "step_id": "step_1",
                        "description": "단계 설명",
                        "method": "web_search|academic_search|data_analysis|content_analysis",
                        "query": "검색 쿼리",
                        "sources": ["웹", "학술", "데이터베이스"],
                        "expected_output": "예상 결과"
                    }}
                ],
                "quality_criteria": ["품질 기준1", "품질 기준2"],
                "success_metrics": ["성공 지표1", "성공 지표2"]
            }}
            """

            response = await asyncio.to_thread(self.llm.generate_content, prompt)
            # Parse Gemini response properly
            response_text = response.text.strip()
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")

        except Exception as e:
            logger.error(f"LLM research planning failed: {e}")
            # Return basic research plan structure
            return {
                "research_strategy": f"Research task: {task.get('description', 'Unknown')}",
                "steps": [
                    {
                        "step_id": "step_1",
                        "description": "Basic web search",
                        "method": "web_search",
                        "query": task.get("description", ""),
                        "sources": ["웹"],
                        "expected_output": "Basic information",
                    }
                ],
                "quality_criteria": ["정보의 정확성", "출처의 신뢰성"],
                "success_metrics": ["정보 수집 완료", "기본 분석 완료"],
            }


    async def _generate_basic_research_plan(self, task_description: str) -> Dict[str, Any]:
        """Generate basic research plan without LLM."""
        try:
            # Create basic research plan with actual tool calls
            basic_plan = {
                "research_strategy": f"Systematic research approach for: {task_description}",
                "tool_calls": [
                    {
                        "tool": "web_search",
                        "query": task_description,
                        "purpose": "Initial research and overview",
                    },
                    {
                        "tool": "web_search",
                        "query": f"{task_description} benefits advantages",
                        "purpose": "Identify benefits and advantages",
                    },
                    {
                        "tool": "web_search",
                        "query": f"{task_description} challenges limitations",
                        "purpose": "Understand challenges and limitations",
                    },
                    {
                        "tool": "web_search",
                        "query": f"{task_description} case studies examples",
                        "purpose": "Find real-world applications and examples",
                    },
                ],
                "analysis_plan": [
                    {
                        "step": "Synthesize findings from multiple sources",
                        "method": "Cross-reference and validate information",
                    },
                    {
                        "step": "Identify key patterns and trends",
                        "method": "Pattern analysis and trend identification",
                    },
                ],
                "validation_criteria": [
                    "Source credibility",
                    "Information recency",
                    "Cross-source validation",
                ],
                "expected_outcomes": [
                    "Comprehensive overview of the topic",
                    "Key benefits and challenges identified",
                    "Real-world examples and case studies",
                ],
            }

            # Execute the basic research plan
            research_results = await self._execute_llm_research_plan(basic_plan, task_description)
            return research_results

        except Exception as e:
            logger.error(f"Basic research plan generation failed: {e}")
            return {"method": "basic_research", "error": str(e)}


    async def _execute_research_step(
        self, step: Dict[str, Any], task: Dict[str, Any] = None
    ) -> Dict[str, Any] | None:
        """Execute a single research step."""
        try:
            # Ensure step is a dict
            if not isinstance(step, dict):
                step = {"method": "web_search", "query": ""}

            # Ensure task is a dict
            if not isinstance(task, dict):
                task = {"description": ""}

            method = step.get("method", "web_search")
            query = step.get("query", task.get("description", ""))

            if method == "web_search":
                return await self._perform_web_search(query)
            elif method == "academic_search":
                return await self._perform_academic_search(query)
            elif method == "data_analysis":
                return await self._perform_data_analysis(query, task)
            elif method == "content_analysis":
                return await self._perform_content_analysis(query, task)
            else:
                # Create a task object for general research
                general_task = {"description": query, "task_type": "general"}
                return await self._perform_general_research(general_task, "unknown")

        except Exception as e:
            logger.error(f"Research step execution failed: {e}")
            return None


    async def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if agent can handle a specific task.

        Args:
            task: Task to check

        Returns:
            True if agent can handle the task
        """
        try:
            task.get("task_type", "general")
            required_skills = task.get("required_skills", [])

            # Check if agent has required skills
            agent_skills = [
                "research",
                "data_collection",
                "web_search",
                "academic_research",
                "analysis",
            ]

            for skill in required_skills:
                if skill not in agent_skills:
                    return False

            return True

        except Exception as e:
            logger.error(f"Task capability check failed: {e}")
            return False


    async def cancel_tasks(self, objective_id: str) -> bool:
        """Cancel tasks for a specific objective.

        Args:
            objective_id: Objective ID to cancel tasks for

        Returns:
            True if tasks were cancelled successfully
        """
        try:
            cancelled_count = 0

            for task_id, task_info in self.active_tasks.items():
                if task_info.get("objective_id") == objective_id:
                    task_info["status"] = "cancelled"
                    task_info["cancelled_at"] = datetime.now()
                    cancelled_count += 1

            logger.info(f"Cancelled {cancelled_count} tasks for objective: {objective_id}")
            return True

        except Exception as e:
            logger.error(f"Task cancellation failed: {e}")
            return False


    async def cleanup(self):
        """Cleanup agent resources."""
        try:
            # Cancel all active tasks
            for task_id in list(self.active_tasks.keys()):
                self.active_tasks[task_id]["status"] = "cancelled"
                self.active_tasks[task_id]["cancelled_at"] = datetime.now()

            self.active_tasks.clear()
            logger.info("Research Agent cleanup completed")

        except Exception as e:
            logger.error(f"Research Agent cleanup failed: {e}")


    async def update_capabilities(self, evaluation_result: Dict[str, Any], iteration: int) -> None:
        """Update agent capabilities based on evaluation feedback.

        Args:
            evaluation_result: Evaluation results from current iteration
            iteration: Current iteration number
        """
        try:
            feedback = evaluation_result.get("feedback", [])
            quality_metrics = evaluation_result.get("quality_metrics", {})

            # Store learning data
            learning_entry = {
                "iteration": iteration,
                "feedback": feedback,
                "quality_metrics": quality_metrics,
                "timestamp": datetime.now().isoformat(),
            }
            self.learning_data.append(learning_entry)

            # Update adaptive strategies based on feedback
            if "insufficient_data" in str(feedback):
                self.adaptive_strategies["search_depth"] = min(
                    5, self.adaptive_strategies["search_depth"] + 1
                )
                self.adaptive_strategies["content_length"] = min(
                    10000, self.adaptive_strategies["content_length"] + 1000
                )
            elif "excessive_data" in str(feedback):
                self.adaptive_strategies["search_depth"] = max(
                    1, self.adaptive_strategies["search_depth"] - 1
                )
                self.adaptive_strategies["content_length"] = max(
                    2000, self.adaptive_strategies["content_length"] - 1000
                )

            if "insufficient_diversity" in str(feedback):
                self.adaptive_strategies["source_diversity"] = min(
                    1.0, self.adaptive_strategies["source_diversity"] + 0.1
                )
            elif "excessive_diversity" in str(feedback):
                self.adaptive_strategies["source_diversity"] = max(
                    0.5, self.adaptive_strategies["source_diversity"] - 0.1
                )

            # Update research patterns based on successful patterns
            if quality_metrics.get("overall_score", 0) > 0.8:
                await self._update_successful_research_patterns(evaluation_result)

            logger.info(f"ResearchAgent capabilities updated for iteration {iteration}")

        except Exception as e:
            logger.error(f"Research capability update failed: {e}")


    async def _update_successful_research_patterns(self, evaluation_result: Dict[str, Any]) -> None:
        """Update research patterns based on successful evaluations."""
        try:
            # Extract successful patterns from evaluation
            quality_metrics = evaluation_result.get("quality_metrics", {})

            # Update research method weights based on success
            for method_name in self.analysis_methods:
                if quality_metrics.get(f"{method_name}_score", 0) > 0.8:
                    if "weight" not in self.analysis_methods[method_name]:
                        self.analysis_methods[method_name]["weight"] = 1.0
                    self.analysis_methods[method_name]["weight"] = min(
                        2.0, self.analysis_methods[method_name]["weight"] + 0.1
                    )

            logger.info("Successful research patterns updated")

        except Exception as e:
            logger.error(f"Research pattern update failed: {e}")


    async def _enhanced_research_with_learning(
        self, task: Dict[str, Any], context: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Enhanced research using learning from previous iterations."""
        try:
            # Get learning data from context
            learning_data = context.get("learning_data", [])
            iteration = context.get("iteration", 1)

            # Start with base research
            result = await self.conduct_research(task, context, objective_id)

            # Apply learning enhancements
            if learning_data and iteration > 1:
                result = await self._apply_research_learning_enhancements(
                    result, learning_data, iteration
                )

            return result

        except Exception as e:
            logger.error(f"Enhanced research failed: {e}")
            return await self.conduct_research(task, context, objective_id)


    async def _apply_research_learning_enhancements(
        self,
        result: Dict[str, Any],
        learning_data: List[Dict[str, Any]],
        iteration: int,
    ) -> Dict[str, Any]:
        """Apply learning enhancements to research results."""
        try:
            enhanced_result = result.copy()

            # Apply learning-based enhancements
            if iteration > 1:
                latest_feedback = learning_data[-1].get("feedback", [])

                # Enhance data collection if insufficient
                if "insufficient_data" in str(latest_feedback):
                    enhanced_result["data_enhanced"] = True
                    enhanced_result["search_depth"] = self.adaptive_strategies["search_depth"]
                    enhanced_result["content_length"] = self.adaptive_strategies["content_length"]

                # Enhance source diversity if needed
                if "insufficient_diversity" in str(latest_feedback):
                    enhanced_result["diversity_enhanced"] = True
                    enhanced_result["source_diversity"] = self.adaptive_strategies[
                        "source_diversity"
                    ]

                # Enhance quality if needed
                if "quality_issues" in str(latest_feedback):
                    enhanced_result["quality_enhanced"] = True
                    enhanced_result["quality_threshold"] = 0.8

            # Add learning metadata
            enhanced_result["learning_applied"] = True
            enhanced_result["iteration"] = iteration
            enhanced_result["learning_data_count"] = len(learning_data)

            return enhanced_result

        except Exception as e:
            logger.error(f"Research learning enhancement failed: {e}")
            return result

    # Browser automation methods are now handled by BrowserManager


