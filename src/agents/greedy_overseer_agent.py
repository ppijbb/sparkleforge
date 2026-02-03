"""
Greedy Overseer Agent - Ensures research completeness and quality

This agent sits between Planning and Synthesis, evaluating execution results
and demanding more work when information is insufficient. It enforces high
standards and pushes for excellence in research quality.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

from src.core.llm_manager import execute_llm_task, TaskType
from src.core.researcher_config import get_llm_config
from src.core.shared_memory import get_shared_memory, MemoryScope

logger = logging.getLogger(__name__)


@dataclass
class OverseerEvaluation:
    """Evaluation result from the Overseer"""
    iteration: int
    completeness_score: float
    quality_score: float
    academic_sources_count: int
    verified_sources_count: int
    decision: str  # 'proceed', 'retry', 'ask_user'
    reasoning: str
    additional_requirements: List[Dict[str, Any]]
    timestamp: str


class GreedyOverseerAgent:
    """
    Greedy Overseer Agent - Evaluates research completeness and quality.
    
    Responsibilities:
    - Review planning output and define requirements
    - Evaluate execution results for completeness
    - Assess quality of sources (academic rigor, credibility, verifiability)
    - Demand additional work when insufficient
    - Trigger human-in-loop for clarifications
    - Decide when to proceed to synthesis
    """
    
    def __init__(
        self,
        max_iterations: int = 5,
        completeness_threshold: float = 0.9,
        quality_threshold: float = 0.85,
        min_academic_sources: int = 3,
        min_verified_sources: int = 5,
        require_cross_validation: bool = True,
        enable_human_loop: bool = True
    ):
        """
        Initialize Greedy Overseer Agent.
        
        Args:
            max_iterations: Maximum number of retry iterations
            completeness_threshold: Minimum completeness score (0-1)
            quality_threshold: Minimum quality score (0-1)
            min_academic_sources: Minimum number of academic sources required
            min_verified_sources: Minimum number of verified sources required
            require_cross_validation: Whether cross-validation is required
            enable_human_loop: Whether to trigger human-in-loop for clarifications
        """
        self.max_iterations = max_iterations
        self.completeness_threshold = completeness_threshold
        self.quality_threshold = quality_threshold
        self.min_academic_sources = min_academic_sources
        self.min_verified_sources = min_verified_sources
        self.require_cross_validation = require_cross_validation
        self.enable_human_loop = enable_human_loop
        
        self.llm_config = get_llm_config()
        self.shared_memory = get_shared_memory()
        
        logger.info(f"GreedyOverseerAgent initialized with thresholds: "
                   f"completeness={completeness_threshold}, quality={quality_threshold}, "
                   f"academic_sources={min_academic_sources}")
    
    async def review_planning_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review planning output and define requirements.
        
        Args:
            state: Research state containing planning results
            
        Returns:
            Updated state with overseer requirements
        """
        logger.info("=" * 80)
        logger.info("[OVERSEER] Reviewing planning output")
        logger.info("=" * 80)
        
        try:
            user_request = state.get('user_request', '')
            planned_tasks = state.get('planned_tasks', [])
            analyzed_objectives = state.get('analyzed_objectives', [])
            
            # Generate requirements using LLM
            review_prompt = f"""
            Review the research plan and identify what information is needed to ensure high-quality, complete research.
            
            Research Query: {user_request}
            
            Analyzed Objectives: {analyzed_objectives}
            
            Planned Tasks: {planned_tasks}
            
            Identify and specify:
            1. Key information gaps that must be filled
            2. Required source types (academic papers, official documents, verified data, etc.)
            3. Minimum quality standards (credibility, verifiability, academic rigor)
            4. Additional requirements to ensure completeness
            5. Cross-validation needs
            
            Be thorough and demanding. Set high standards for research quality.
            
            Return JSON format:
            {{
                "key_information_gaps": ["gap1", "gap2", ...],
                "required_source_types": {{
                    "academic_papers": 3,
                    "official_documents": 2,
                    "verified_data": 5
                }},
                "quality_standards": {{
                    "min_credibility": 0.8,
                    "min_verifiability": 0.85,
                    "require_peer_review": true
                }},
                "additional_requirements": [
                    {{"requirement": "...", "priority": "high/medium/low"}}
                ],
                "cross_validation_needs": ["topic1", "topic2", ...]
            }}
            """
            
            result = await execute_llm_task(
                prompt=review_prompt,
                task_type=TaskType.PLANNING,
                system_message="You are a demanding overseer ensuring research excellence."
            )
            
            # Parse requirements
            import json
            import re
            content = result.content or "{}"
            
            # Clean markdown code blocks
            if '```json' in content:
                match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            elif '```' in content:
                match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            
            try:
                requirements = json.loads(content)
            except json.JSONDecodeError:
                logger.warning("Failed to parse requirements JSON, using defaults")
                requirements = {
                    "key_information_gaps": [],
                    "required_source_types": {
                        "academic_papers": self.min_academic_sources,
                        "verified_data": self.min_verified_sources
                    },
                    "quality_standards": {
                        "min_credibility": self.quality_threshold,
                        "min_verifiability": self.quality_threshold
                    },
                    "additional_requirements": [],
                    "cross_validation_needs": []
                }
            
            # Initialize overseer state
            state['overseer_iterations'] = state.get('overseer_iterations', 0)
            state['overseer_requirements'] = [requirements]
            state['overseer_evaluations'] = []
            state['completeness_scores'] = {}
            state['quality_assessments'] = {}
            state['overseer_decision'] = 'continue'
            
            logger.info(f"[OVERSEER] Requirements defined: {len(requirements.get('additional_requirements', []))} additional requirements")
            logger.info(f"[OVERSEER] Required sources: {requirements.get('required_source_types', {})}")
            
            return state
            
        except Exception as e:
            logger.error(f"[OVERSEER] Error reviewing planning output: {e}")
            # Initialize with minimal requirements
            state['overseer_iterations'] = 0
            state['overseer_requirements'] = []
            state['overseer_evaluations'] = []
            state['completeness_scores'] = {}
            state['quality_assessments'] = {}
            state['overseer_decision'] = 'continue'
            return state
    
    async def evaluate_execution_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate execution and validation results.
        
        Args:
            state: Research state containing execution and validation results
            
        Returns:
            Updated state with evaluation and decision
        """
        logger.info("=" * 80)
        logger.info("[OVERSEER] Evaluating execution results")
        logger.info("=" * 80)
        
        try:
            current_iteration = state.get('overseer_iterations', 0)
            
            # Get data from state
            planned_tasks = state.get('planned_tasks', [])
            execution_results = state.get('execution_results', [])
            verified_results = state.get('verified_results', [])
            quality_assessments = state.get('quality_assessments', {})
            requirements = state.get('overseer_requirements', [{}])[-1] if state.get('overseer_requirements') else {}
            
            # Assess completeness
            completeness_score = await self._assess_completeness(
                planned_tasks, execution_results, verified_results
            )
            
            # Assess quality
            quality_score = await self._assess_quality(
                verified_results, quality_assessments
            )
            
            # Count sources by type
            academic_sources = self._count_academic_sources(verified_results)
            verified_sources = len([r for r in verified_results if r.get('status') == 'verified'])
            
            # Make decision
            decision, reasoning, additional_requirements = await self._decide_next_action(
                completeness_score=completeness_score,
                quality_score=quality_score,
                academic_sources=academic_sources,
                verified_sources=verified_sources,
                current_iteration=current_iteration,
                requirements=requirements,
                state=state
            )
            
            # Create evaluation record
            evaluation = OverseerEvaluation(
                iteration=current_iteration,
                completeness_score=completeness_score,
                quality_score=quality_score,
                academic_sources_count=academic_sources,
                verified_sources_count=verified_sources,
                decision=decision,
                reasoning=reasoning,
                additional_requirements=additional_requirements,
                timestamp=datetime.now().isoformat()
            )
            
            # Update state
            state['overseer_iterations'] = current_iteration + 1
            state['overseer_evaluations'] = state.get('overseer_evaluations', []) + [evaluation.__dict__]
            state['completeness_scores'][f"iteration_{current_iteration}"] = completeness_score
            state['overseer_decision'] = decision
            
            # Add requirements if retrying
            if decision == 'retry' and additional_requirements:
                state['overseer_requirements'] = state.get('overseer_requirements', []) + [
                    {
                        "iteration": current_iteration + 1,
                        "requirements": additional_requirements
                    }
                ]
            
            logger.info(f"[OVERSEER] Evaluation complete:")
            logger.info(f"  - Completeness: {completeness_score:.2f}")
            logger.info(f"  - Quality: {quality_score:.2f}")
            logger.info(f"  - Academic sources: {academic_sources}/{self.min_academic_sources}")
            logger.info(f"  - Verified sources: {verified_sources}/{self.min_verified_sources}")
            logger.info(f"  - Decision: {decision}")
            logger.info(f"  - Reasoning: {reasoning}")
            
            if decision == 'retry':
                logger.info(f"  - Additional requirements: {len(additional_requirements)}")
            
            logger.info("=" * 80)
            
            return state
            
        except Exception as e:
            logger.error(f"[OVERSEER] Error evaluating execution results: {e}")
            # Default to proceed on error
            state['overseer_decision'] = 'proceed'
            return state
    
    async def _assess_completeness(
        self,
        planned_tasks: List[Dict[str, Any]],
        execution_results: List[Dict[str, Any]],
        verified_results: List[Dict[str, Any]]
    ) -> float:
        """
        Assess completeness of research results against planned tasks.
        
        Returns:
            Completeness score (0-1)
        """
        if not planned_tasks:
            return 1.0  # No tasks to complete
        
        try:
            # Use LLM to assess completeness
            assessment_prompt = f"""
            Assess how completely the execution results address the planned research tasks.
            
            Planned Tasks ({len(planned_tasks)}):
            {self._format_tasks(planned_tasks)}
            
            Execution Results ({len(execution_results)}):
            {self._format_results(execution_results[:10])}  # Limit for token efficiency
            
            Verified Results ({len(verified_results)}):
            {self._format_results(verified_results[:10])}
            
            For each planned task, assess if it has been adequately addressed (0-1 score).
            Return overall completeness score (0-1) and reasoning.
            
            Return JSON:
            {{
                "task_completeness": {{"task1": 0.X, "task2": 0.X, ...}},
                "overall_completeness": 0.X,
                "reasoning": "..."
            }}
            """
            
            result = await execute_llm_task(
                prompt=assessment_prompt,
                task_type=TaskType.VERIFICATION,
                system_message="You are evaluating research completeness."
            )
            
            # Parse result
            import json
            import re
            content = result.content or "{}"
            
            if '```json' in content:
                match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            
            parsed = json.loads(content)
            completeness_score = parsed.get('overall_completeness', 0.5)
            
            return max(0.0, min(1.0, completeness_score))
            
        except Exception as e:
            logger.warning(f"Failed to assess completeness with LLM: {e}")
            # Fallback: simple ratio
            if not planned_tasks:
                return 1.0
            return min(1.0, len(verified_results) / len(planned_tasks))
    
    async def _assess_quality(
        self,
        verified_results: List[Dict[str, Any]],
        quality_assessments: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Assess overall quality of research results.
        
        Returns:
            Quality score (0-1)
        """
        if not verified_results:
            return 0.0
        
        try:
            # Aggregate quality scores from assessments
            quality_scores = []
            
            for result in verified_results:
                result_id = result.get('id') or result.get('url', '')
                assessment = quality_assessments.get(result_id, {})
                
                if assessment and 'overall_quality' in assessment:
                    quality_scores.append(assessment['overall_quality'])
                else:
                    # Use confidence as proxy
                    confidence = result.get('confidence', 0.5)
                    quality_scores.append(confidence)
            
            if quality_scores:
                return sum(quality_scores) / len(quality_scores)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Failed to assess quality: {e}")
            return 0.5
    
    def _count_academic_sources(self, verified_results: List[Dict[str, Any]]) -> int:
        """Count academic sources in verified results."""
        academic_count = 0
        
        for result in verified_results:
            url = result.get('url', '').lower()
            title = result.get('title', '').lower()
            
            # Check for academic indicators
            academic_indicators = [
                'arxiv.org', 'doi.org', 'scholar.google', 'pubmed',
                '.edu', 'journal', 'conference', 'paper',
                'research', 'peer-reviewed', 'proceedings'
            ]
            
            if any(indicator in url or indicator in title for indicator in academic_indicators):
                academic_count += 1
        
        return academic_count
    
    async def _decide_next_action(
        self,
        completeness_score: float,
        quality_score: float,
        academic_sources: int,
        verified_sources: int,
        current_iteration: int,
        requirements: Dict[str, Any],
        state: Dict[str, Any]
    ) -> tuple[str, str, List[Dict[str, Any]]]:
        """
        Decide next action based on evaluation.
        
        Returns:
            (decision, reasoning, additional_requirements)
        """
        reasons = []
        additional_requirements = []
        
        # Check iteration limit
        if current_iteration >= self.max_iterations:
            return (
                'proceed',
                f"Maximum iterations ({self.max_iterations}) reached. Proceeding with best available results.",
                []
            )
        
        # Check completeness
        if completeness_score < self.completeness_threshold:
            reasons.append(f"Completeness {completeness_score:.2f} below threshold {self.completeness_threshold}")
            additional_requirements.append({
                "type": "completeness",
                "description": "Address incomplete research objectives",
                "priority": "high"
            })
        
        # Check quality
        if quality_score < self.quality_threshold:
            reasons.append(f"Quality {quality_score:.2f} below threshold {self.quality_threshold}")
            additional_requirements.append({
                "type": "quality",
                "description": "Improve source quality and credibility",
                "priority": "high"
            })
        
        # Check academic sources
        if academic_sources < self.min_academic_sources:
            reasons.append(f"Academic sources {academic_sources} below minimum {self.min_academic_sources}")
            additional_requirements.append({
                "type": "academic_sources",
                "description": f"Find at least {self.min_academic_sources - academic_sources} more academic sources",
                "priority": "high"
            })
        
        # Check verified sources
        if verified_sources < self.min_verified_sources:
            reasons.append(f"Verified sources {verified_sources} below minimum {self.min_verified_sources}")
            additional_requirements.append({
                "type": "verified_sources",
                "description": f"Find at least {self.min_verified_sources - verified_sources} more verified sources",
                "priority": "medium"
            })
        
        # Make decision
        if not reasons:
            return (
                'proceed',
                "All quality and completeness thresholds met. Proceeding to synthesis.",
                []
            )
        elif self.enable_human_loop and len(reasons) > 2 and current_iteration > 2:
            # Ask user if we're stuck after multiple iterations
            return (
                'ask_user',
                f"Multiple issues after {current_iteration} iterations: {'; '.join(reasons)}. Seeking user guidance.",
                additional_requirements
            )
        else:
            return (
                'retry',
                f"Insufficient quality/completeness: {'; '.join(reasons)}. Requesting additional research.",
                additional_requirements
            )
    
    async def trigger_human_clarification(
        self,
        state: Dict[str, Any],
        ambiguity_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate questions when Overseer needs user input.
        
        Args:
            state: Research state
            ambiguity_type: Type of ambiguity
            context: Context for the question
            
        Returns:
            Updated state with questions or auto-selected responses
        """
        from src.core.human_clarification_handler import get_clarification_handler
        
        clarification_handler = get_clarification_handler()
        
        # Create ambiguity
        ambiguity = {
            "type": ambiguity_type,
            "field": "research_scope",
            "description": context.get('description', ''),
            "suggested_question": context.get('question', ''),
            "suggested_options": context.get('options', [])
        }
        
        # Generate question
        question = await clarification_handler.generate_question(
            ambiguity,
            context
        )
        
        # Check autopilot mode
        if state.get('autopilot_mode', False):
            logger.info("[OVERSEER] Autopilot mode - auto-selecting response")
            
            # Auto-select response
            response = await clarification_handler.auto_select_response(
                question,
                context,
                self.shared_memory
            )
            
            # Process response
            processed = await clarification_handler.process_user_response(
                question['id'],
                response,
                {'question': question}
            )
            
            if processed.get('validated', False):
                # Apply clarification
                state['user_responses'] = state.get('user_responses', {})
                state['user_responses'][question['id']] = processed
                
                state['clarification_context'] = state.get('clarification_context', {})
                state['clarification_context'][question['id']] = processed.get('clarification', {})
                
                logger.info(f"[OVERSEER] Auto-selected: {response}")
        else:
            # Ask user
            state['pending_questions'] = state.get('pending_questions', []) + [question]
            state['waiting_for_user'] = True
            state['overseer_decision'] = 'ask_user'
            
            logger.info(f"[OVERSEER] Generated question for user: {question['id']}")
        
        return state
    
    def _format_tasks(self, tasks: List[Dict[str, Any]]) -> str:
        """Format tasks for LLM prompt."""
        formatted = []
        for i, task in enumerate(tasks[:10], 1):  # Limit for tokens
            formatted.append(f"{i}. {task.get('description', task.get('task', 'N/A'))}")
        return "\n".join(formatted)
    
    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format results for LLM prompt."""
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'N/A')
            status = result.get('status', 'N/A')
            confidence = result.get('confidence', 0.0)
            formatted.append(f"{i}. {title} (status: {status}, confidence: {confidence:.2f})")
        return "\n".join(formatted)


def get_greedy_overseer_agent(**kwargs) -> GreedyOverseerAgent:
    """Get or create Greedy Overseer Agent instance."""
    return GreedyOverseerAgent(**kwargs)

