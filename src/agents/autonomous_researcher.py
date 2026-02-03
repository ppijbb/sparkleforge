"""
Autonomous Researcher Agent (v2.0 - 8대 혁신)

자율적으로 계획을 수립하고 작업을 수행하는 리서처 에이전트.
Multi-Model Orchestration, Universal MCP Hub, Production-Grade Reliability를 활용.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Lazy import to avoid circular dependencies and configuration issues
def get_config_functions():
    """Lazy import of configuration functions."""
    from src.core.researcher_config import get_llm_config, get_agent_config, get_research_config, get_mcp_config
    return get_llm_config, get_agent_config, get_research_config, get_mcp_config

def get_core_functions():
    """Lazy import of core functions."""
    from src.core.llm_manager import execute_llm_task, TaskType, get_best_model_for_task
    from src.core.mcp_integration import execute_tool, get_best_tool_for_task, ToolCategory
    from src.core.reliability import execute_with_reliability
    from src.core.compression import compress_data
    return execute_llm_task, TaskType, get_best_model_for_task, execute_tool, get_best_tool_for_task, ToolCategory, execute_with_reliability, compress_data


class AutonomousResearcherAgent:
    """
    자율적으로 계획을 수립하고 작업을 수행하는 리서처 에이전트.
    MCP agent 라이브러리를 사용하여 실제 작업을 수행.
    """
    
    def __init__(self):
        """Initialize the autonomous researcher agent."""
        # Load configurations (lazy import)
        get_llm_config, get_agent_config, get_research_config, get_mcp_config = get_config_functions()
        self.llm_config = get_llm_config()
        self.agent_config = get_agent_config()
        self.research_config = get_research_config()
        self.mcp_config = get_mcp_config()
        
        self.name = "autonomous_researcher"
        self.instruction = "Autonomous researcher agent that self-plans and executes research tasks"
        
        # Initialize specialized agents
        self._initialize_agents()
        
        # Store core functions for later use
        self._execute_llm_task = None
        self._TaskType = None
        self._execute_with_reliability = None
        self._compress_data = None
        
    def _initialize_agents(self):
        """Initialize specialized research agents with Multi-Model Orchestration."""
        # Agent instructions with enhanced capabilities
        self.agent_instructions = {
            "task_analyzer": """You are an advanced task analysis agent with Multi-Model Orchestration capabilities. Your role is to:
            1. Analyze user requests and break them down into clear, actionable objectives
            2. Identify required research areas, methodologies, and data sources
            3. Determine success criteria, validation methods, and quality metrics
            4. Create detailed research plans with timelines, resource allocation, and risk assessment
            5. Select optimal tools and models for each research phase
            6. Implement adaptive planning based on complexity and requirements
            Always provide specific, actionable analysis with production-level quality.""",
            
            "research_executor": """You are an advanced research execution agent with Universal MCP Hub integration. Your role is to:
            1. Execute research tasks using 100+ MCP tools and APIs
            2. Gather information from multiple sources with real-time validation
            3. Analyze and synthesize findings with continuous verification
            4. Maintain research quality and accuracy with production-grade reliability
            5. Use streaming pipeline for real-time result delivery
            6. Implement hierarchical compression for efficient data processing
            Always use real data sources and provide evidence-based results with 95%+ reliability.""",
            
            "evaluator": """You are an advanced evaluation agent with Continuous Verification capabilities. Your role is to:
            1. Critically evaluate research findings with 3-stage verification
            2. Assess quality and reliability of sources with confidence scoring
            3. Identify gaps and limitations with early warning systems
            4. Provide improvement recommendations with actionable insights
            5. Implement fact-checking and uncertainty marking
            6. Use multi-model ensemble for validation accuracy
            Always provide objective, evidence-based evaluations with 95%+ confidence.""",
            
            "synthesizer": """You are an advanced synthesis agent with Adaptive Context Window capabilities. Your role is to:
            1. Integrate findings from multiple sources with hierarchical compression
            2. Create comprehensive reports with multi-format support
            3. Generate actionable insights with confidence scoring
            4. Present results in clear, professional format with streaming delivery
            5. Implement adaptive context management for long-form content
            6. Use production-grade reliability for consistent output quality
            Always provide complete, well-structured synthesis with 99.9% availability."""
        }
    
    async def self_plan_research(self, user_request: str) -> Dict[str, Any]:
        """자율적으로 연구 계획을 수립합니다 (Multi-Model Orchestration)."""
        # Lazy import
        execute_llm_task, TaskType, _, _, _, _, _, _ = get_core_functions()
        
        planning_prompt = f"""
        {self.agent_instructions['task_analyzer']}
        
        Analyze the following research request and create a comprehensive research plan:
        
        Request: {user_request}
        
        Create a detailed plan including:
        1. Research objectives and questions
        2. Required data sources and methodologies (specify MCP tools to use)
        3. Timeline and milestones with adaptive scheduling
        4. Success criteria and quality metrics
        5. Risk assessment and mitigation strategies
        6. Resource allocation and cost optimization
        7. Model selection strategy for each phase
        
        Provide specific, actionable steps with production-level quality.
        """
        
        # Multi-Model Orchestration으로 계획 수립
        result = await execute_llm_task(
            prompt=planning_prompt,
            task_type=TaskType.PLANNING,
            system_message="You are an expert research planner with access to 100+ MCP tools and multi-model capabilities."
        )
        
        return {
            "research_plan": result.content,
            "model_used": result.model_used,
            "confidence": result.confidence,
            "execution_time": result.execution_time,
            "created_at": datetime.now().isoformat(),
            "status": "planned"
        }
    
    async def execute_research(self, research_plan: Dict[str, Any]) -> Dict[str, Any]:
        """연구 계획을 실행합니다 (Universal MCP Hub + Streaming Pipeline)."""
        # Lazy import
        execute_llm_task, TaskType, _, execute_tool, get_best_tool_for_task, ToolCategory, _, _ = get_core_functions()
        import logging
        logger = logging.getLogger(__name__)
        
        execution_prompt = f"""
        {self.agent_instructions['research_executor']}
        
        Execute the following research plan:
        
        Plan: {research_plan.get('research_plan', '')}
        
        Perform the research using Universal MCP Hub with 100+ tools.
        Use streaming pipeline for real-time result delivery.
        Implement hierarchical compression for efficient data processing.
        Provide evidence-based results with production-grade reliability.
        """
        
        # Universal MCP Hub를 사용한 연구 실행
        research_results = []
        
        # 1. 웹 검색
        try:
            search_tool = await get_best_tool_for_task("search", ToolCategory.SEARCH)
            if search_tool:
                search_result = await execute_tool(
                    search_tool,
                    {"query": research_plan.get('research_plan', '')[:100]}
                )
                if search_result.success:
                    research_results.append({
                        "source": "web_search",
                        "data": search_result.data,
                        "tool_used": search_tool
                    })
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
        
        # 2. 학술 검색
        try:
            academic_tool = await get_best_tool_for_task("academic", ToolCategory.ACADEMIC)
            if academic_tool:
                academic_result = await execute_tool(
                    academic_tool,
                    {"query": research_plan.get('research_plan', '')[:100]}
                )
                if academic_result.success:
                    research_results.append({
                        "source": "academic_search",
                        "data": academic_result.data,
                        "tool_used": academic_tool
                    })
        except Exception as e:
            logger.warning(f"Academic search failed: {e}")
        
        # 3. 데이터 분석 및 종합
        analysis_prompt = f"""
        Analyze and synthesize the following research data:
        
        Research Plan: {research_plan.get('research_plan', '')}
        Search Results: {json.dumps(research_results, ensure_ascii=False, indent=2)}
        
        Provide comprehensive analysis with evidence-based conclusions.
        """
        
        # Multi-Model Orchestration으로 분석
        analysis_result = await execute_llm_task(
            prompt=analysis_prompt,
            task_type=TaskType.DEEP_REASONING,
            system_message="You are an expert research analyst with access to comprehensive data sources."
        )
        
        return {
            "research_results": analysis_result.content,
            "raw_data": research_results,
            "model_used": analysis_result.model_used,
            "confidence": analysis_result.confidence,
            "execution_time": analysis_result.execution_time,
            "executed_at": datetime.now().isoformat(),
            "status": "executed"
        }
    
    async def evaluate_research(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """연구 결과를 평가합니다 (Continuous Verification)."""
        # Lazy import
        execute_llm_task, TaskType, _, _, _, _, _, _ = get_core_functions()
        
        evaluation_prompt = f"""
        {self.agent_instructions['evaluator']}
        
        Evaluate the following research results with 3-stage verification:
        
        Results: {research_results.get('research_results', '')}
        Raw Data: {json.dumps(research_results.get('raw_data', []), ensure_ascii=False, indent=2)}
        
        Provide critical evaluation including:
        1. Quality assessment with confidence scoring
        2. Source reliability analysis with fact-checking
        3. Gap identification with early warning systems
        4. Improvement recommendations with actionable insights
        5. Uncertainty marking for low-confidence claims
        6. Cross-verification with external sources
        
        Provide objective, evidence-based evaluation with 95%+ confidence.
        """
        
        # Continuous Verification으로 평가
        evaluation_result = await execute_llm_task(
            prompt=evaluation_prompt,
            task_type=TaskType.VERIFICATION,
            system_message="You are an expert research evaluator with continuous verification capabilities.",
            use_ensemble=True  # Weighted Ensemble 사용
        )
        
        return {
            "evaluation_results": evaluation_result.content,
            "model_used": evaluation_result.model_used,
            "confidence": evaluation_result.confidence,
            "verification_score": evaluation_result.confidence,
            "execution_time": evaluation_result.execution_time,
            "evaluated_at": datetime.now().isoformat(),
            "status": "evaluated"
        }
    
    async def synthesize_findings(self, research_results: Dict[str, Any], evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """연구 결과를 종합합니다 (Adaptive Context Window + Hierarchical Compression)."""
        # Lazy import
        execute_llm_task, TaskType, _, _, _, _, _, compress_data = get_core_functions()
        import logging
        logger = logging.getLogger(__name__)
        
        synthesis_prompt = f"""
        {self.agent_instructions['synthesizer']}
        
        Synthesize the following research findings with adaptive context management:
        
        Research Results: {research_results.get('research_results', '')}
        Evaluation: {evaluation_results.get('evaluation_results', '')}
        Raw Data: {json.dumps(research_results.get('raw_data', []), ensure_ascii=False, indent=2)}
        
        Create a comprehensive synthesis including:
        1. Executive summary with key insights
        2. Key findings with confidence scores
        3. Evidence and sources with reliability assessment
        4. Conclusions and recommendations with actionability
        5. Limitations and future work with improvement paths
        6. Multi-format output support (PDF, Markdown, JSON)
        
        Provide complete, well-structured synthesis with production-grade quality.
        """
        
        # Multi-Model Orchestration으로 종합
        synthesis_result = await execute_llm_task(
            prompt=synthesis_prompt,
            task_type=TaskType.SYNTHESIS,
            system_message="You are an expert research synthesizer with adaptive context window capabilities."
        )
        
        # Hierarchical Compression 적용
        try:
            compressed_result = await compress_data(synthesis_result.content)
            synthesis_content = compressed_result.data
            compression_info = {
                "compression_ratio": compressed_result.compression_ratio,
                "validation_score": compressed_result.validation_score,
                "important_info_preserved": compressed_result.important_info_preserved
            }
        except Exception as e:
            logger.warning(f"Compression failed: {e}, using original content")
            synthesis_content = synthesis_result.content
            compression_info = {"compression_ratio": 1.0, "validation_score": 1.0}
        
        return {
            "synthesis_results": synthesis_content,
            "model_used": synthesis_result.model_used,
            "confidence": synthesis_result.confidence,
            "execution_time": synthesis_result.execution_time,
            "compression_info": compression_info,
            "synthesized_at": datetime.now().isoformat(),
            "status": "synthesized"
        }
    
    async def run_autonomous_research(self, user_request: str) -> Dict[str, Any]:
        """자율적으로 전체 연구 프로세스를 실행합니다 (8대 혁신 통합)."""
        # Lazy import
        _, _, _, _, _, _, execute_with_reliability, _ = get_core_functions()
        
        print(f"🚀 Starting autonomous research with 8 core innovations for: {user_request}")
        
        # Production-Grade Reliability로 전체 프로세스 실행
        return await execute_with_reliability(
            self._execute_research_workflow,
            user_request,
            component_name="autonomous_research",
            save_state=True
        )
    
    async def _execute_research_workflow(self, user_request: str) -> Dict[str, Any]:
        """연구 워크플로우 실행 (내부 메서드)."""
        # 1. 자율 계획 수립 (Multi-Model Orchestration)
        if self.agent_config.enable_self_planning:
            print("1. 📋 Self-planning research with Multi-Model Orchestration...")
            research_plan = await self.self_plan_research(user_request)
            print(f"✅ Research plan created: {research_plan['status']} (Model: {research_plan.get('model_used', 'N/A')})")
        else:
            raise ValueError("Self-planning is disabled but required for autonomous operation")
        
        # 2. 연구 실행 (Universal MCP Hub + Streaming Pipeline)
        print("2. 🔍 Executing research with Universal MCP Hub...")
        research_results = await self.execute_research(research_plan)
        print(f"✅ Research executed: {research_results['status']} (Model: {research_results.get('model_used', 'N/A')})")
        
        # 3. 연구 평가 (Continuous Verification)
        print("3. 🔬 Evaluating research with Continuous Verification...")
        evaluation_results = await self.evaluate_research(research_results)
        print(f"✅ Research evaluated: {evaluation_results['status']} (Confidence: {evaluation_results.get('confidence', 0):.2%})")
        
        # 4. 결과 종합 (Adaptive Context Window + Hierarchical Compression)
        print("4. 📊 Synthesizing findings with Hierarchical Compression...")
        synthesis_results = await self.synthesize_findings(research_results, evaluation_results)
        print(f"✅ Findings synthesized: {synthesis_results['status']} (Compression: {synthesis_results.get('compression_info', {}).get('compression_ratio', 1.0):.2%})")
        
        # 5. 최종 결과 반환
        final_result = {
            "user_request": user_request,
            "research_plan": research_plan,
            "research_results": research_results,
            "evaluation_results": evaluation_results,
            "synthesis_results": synthesis_results,
            "innovation_stats": {
                "models_used": [
                    research_plan.get('model_used'),
                    research_results.get('model_used'),
                    evaluation_results.get('model_used'),
                    synthesis_results.get('model_used')
                ],
                "overall_confidence": min(
                    research_plan.get('confidence', 0.8),
                    research_results.get('confidence', 0.8),
                    evaluation_results.get('confidence', 0.8),
                    synthesis_results.get('confidence', 0.8)
                ),
                "compression_applied": synthesis_results.get('compression_info', {}).get('compression_ratio', 1.0),
                "mcp_tools_used": len(research_results.get('raw_data', [])),
                "verification_score": evaluation_results.get('verification_score', 0.8)
            },
            "completed_at": datetime.now().isoformat(),
            "status": "completed"
        }
        
        print("🎉 Autonomous research completed successfully with 8 core innovations!")
        print(f"📈 Innovation Stats:")
        print(f"   - Models Used: {len(set(final_result['innovation_stats']['models_used']))} different models")
        print(f"   - Overall Confidence: {final_result['innovation_stats']['overall_confidence']:.2%}")
        print(f"   - Compression Ratio: {final_result['innovation_stats']['compression_applied']:.2%}")
        print(f"   - MCP Tools Used: {final_result['innovation_stats']['mcp_tools_used']}")
        print(f"   - Verification Score: {final_result['innovation_stats']['verification_score']:.2%}")
        
        return final_result
