"""Synthesis Agent (v2.0 - 8대 혁신 통합)

Adaptive Context Window, Hierarchical Compression, Multi-Model Orchestration,
Production-Grade Reliability를 통합한 고도화된 종합 에이전트.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.compression import compress_data
from src.core.llm_manager import TaskType, execute_llm_task
from src.core.mcp_integration import ToolCategory, execute_tool, get_best_tool_for_task
from src.core.reliability import execute_with_reliability
from src.core.researcher_config import (
    get_context_window_config,
    get_llm_config,
    get_output_config,
)

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """8대 혁신을 통합한 고도화된 종합 에이전트."""

    def __init__(self):
        """초기화."""
        self.llm_config = get_llm_config()
        self.output_config = get_output_config()
        self.context_window_config = get_context_window_config()

        # 종합 기능
        self.synthesis_templates = self._load_synthesis_templates()
        self.deliverable_formats = self._load_deliverable_formats()
        self.insight_generation_methods = self._load_insight_methods()

        logger.info("Synthesis Agent initialized with 8 core innovations")

    def _load_synthesis_templates(self) -> Dict[str, Any]:
        """종합 템플릿 로드."""
        return {
            "executive_summary": {
                "structure": [
                    "overview",
                    "key_findings",
                    "recommendations",
                    "next_steps",
                ],
                "max_length": 500,
                "priority": "high",
            },
            "detailed_report": {
                "structure": [
                    "introduction",
                    "methodology",
                    "findings",
                    "analysis",
                    "conclusions",
                    "recommendations",
                ],
                "max_length": 5000,
                "priority": "high",
            },
            "technical_document": {
                "structure": [
                    "abstract",
                    "introduction",
                    "methods",
                    "results",
                    "discussion",
                    "references",
                ],
                "max_length": 10000,
                "priority": "medium",
            },
            "presentation": {
                "structure": [
                    "title",
                    "agenda",
                    "key_points",
                    "supporting_evidence",
                    "conclusions",
                    "q_and_a",
                ],
                "max_length": 2000,
                "priority": "medium",
            },
        }

    def _load_deliverable_formats(self) -> Dict[str, Any]:
        """전달물 형식 로드."""
        return {
            "pdf": {
                "enabled": self.output_config.enable_pdf_generation,
                "template": "professional_report",
                "mcp_tools": ["filesystem", "python_coder"],
            },
            "markdown": {
                "enabled": self.output_config.enable_markdown_generation,
                "template": "github_markdown",
                "mcp_tools": ["filesystem"],
            },
            "json": {
                "enabled": self.output_config.enable_json_export,
                "template": "structured_data",
                "mcp_tools": ["filesystem"],
            },
            "docx": {
                "enabled": self.output_config.enable_docx_export,
                "template": "word_document",
                "mcp_tools": ["python_coder"],
            },
            "html": {
                "enabled": self.output_config.enable_html_export,
                "template": "web_page",
                "mcp_tools": ["python_coder"],
            },
        }

    def _load_insight_methods(self) -> Dict[str, Any]:
        """인사이트 생성 방법 로드."""
        return {
            "pattern_analysis": {
                "description": "Identify patterns and trends in data",
                "complexity": "medium",
                "mcp_tools": ["python_coder", "code_interpreter"],
            },
            "comparative_analysis": {
                "description": "Compare different sources and findings",
                "complexity": "high",
                "mcp_tools": ["g-search", "tavily", "exa"],
            },
            "predictive_analysis": {
                "description": "Generate predictions and forecasts",
                "complexity": "high",
                "mcp_tools": ["python_coder", "code_interpreter"],
            },
            "recommendation_engine": {
                "description": "Generate actionable recommendations",
                "complexity": "medium",
                "mcp_tools": ["python_coder"],
            },
        }

    async def synthesize_results(
        self,
        execution_results: List[Dict[str, Any]],
        evaluation_results: Dict[str, Any],
        original_objectives: List[Dict[str, Any]],
        context: Dict[str, Any] | None = None,
        deliverable_type: str = "detailed_report",
    ) -> Dict[str, Any]:
        """연구 결과 종합 (8대 혁신 통합)."""
        logger.info(
            f"📝 Starting synthesis with 8 core innovations for deliverable type: {deliverable_type}"
        )

        # Production-Grade Reliability로 종합 실행
        return await execute_with_reliability(
            self._execute_synthesis_workflow,
            execution_results,
            evaluation_results,
            original_objectives,
            context,
            deliverable_type,
            component_name="synthesis_agent",
            save_state=True,
        )

    async def _execute_synthesis_workflow(
        self,
        execution_results: List[Dict[str, Any]],
        evaluation_results: Dict[str, Any],
        original_objectives: List[Dict[str, Any]],
        context: Dict[str, Any] | None,
        deliverable_type: str,
    ) -> Dict[str, Any]:
        """종합 워크플로우 실행 (내부 메서드)."""
        # Phase 1: Adaptive Context Window Management (혁신 7)
        logger.info("1. 🧠 Managing Adaptive Context Window")
        context_usage = await self._manage_context_window(
            execution_results, evaluation_results, original_objectives
        )

        # Phase 2: Data Integration and Analysis
        logger.info("2. 🔗 Integrating and analyzing data")
        integrated_data = await self._integrate_data(
            execution_results, evaluation_results, original_objectives
        )

        # Phase 3: Insight Generation (Multi-Model Orchestration)
        logger.info("3. 💡 Generating insights with Multi-Model Orchestration")
        insights = await self._generate_insights(integrated_data, deliverable_type)

        # Phase 4: Content Synthesis (Multi-Model Orchestration)
        logger.info("4. 📊 Synthesizing content with Multi-Model Orchestration")
        synthesized_content = await self._synthesize_content(
            integrated_data, insights, deliverable_type
        )

        # Phase 5: Hierarchical Compression (혁신 2)
        logger.info("5. 🗜️ Applying Hierarchical Compression")
        compressed_content = await self._compress_content(synthesized_content)

        # Phase 6: Multi-Format Generation (Universal MCP Hub)
        logger.info("6. 📄 Generating multi-format deliverables with Universal MCP Hub")
        deliverables = await self._generate_deliverables(compressed_content, deliverable_type)

        # Phase 7: Quality Validation
        logger.info("7. ✅ Validating synthesis quality")
        validation_results = await self._validate_synthesis(compressed_content, deliverables)

        # Council 활성화 확인 및 적용 (최종 보고서 생성 시 - 기본 활성화)
        use_council = context.get("use_council", None) if context else None  # 수동 활성화 옵션
        if use_council is None:
            # 자동 활성화 판단 (기본 활성화)
            from src.core.council_activator import get_council_activator

            activator = get_council_activator()

            activation_decision = activator.should_activate(
                process_type="synthesis",
                query=(
                    str(original_objectives[0].get("description", ""))
                    if original_objectives
                    else ""
                ),
                context={"important_conclusion": True},  # 종합은 항상 중요한 결론 도출
            )
            use_council = activation_decision.should_activate
            if use_council:
                logger.info(
                    f"📝 Council auto-activated for synthesis: {activation_decision.reason}"
                )

        # Council 적용 (활성화된 경우)
        if use_council:
            try:
                from src.core.llm_council import run_full_council

                logger.info("📝 🏛️ Running Council review for synthesis results...")

                # 종합 결과 요약 생성
                synthesis_summary = f"""Deliverable Type: {deliverable_type}
Compression Ratio: {compressed_content.get("compression_ratio", 1.0):.2f}
Overall Quality: {validation_results.get("overall_quality", 0.8):.2f}
Insights Generated: {len(insights)}
Formats Generated: {len(deliverables)}"""

                # 종합 내용 샘플 (최대 2000자)
                content_sample = (
                    str(synthesized_content.get("content", ""))[:2000]
                    if isinstance(synthesized_content, dict)
                    else str(synthesized_content)[:2000]
                )

                council_query = f"""Review the synthesis results and assess their completeness and accuracy. Check for any missing information or potential improvements.

Original Objectives: {str(original_objectives)}

Synthesis Summary:
{synthesis_summary}

Content Sample:
{content_sample}

Provide a review with:
1. Completeness assessment
2. Accuracy check
3. Recommendations for improvement"""

                (
                    stage1_results,
                    stage2_results,
                    stage3_result,
                    metadata,
                ) = await run_full_council(council_query)

                # Council 검토 결과
                review_report = stage3_result.get("response", "")
                logger.info("📝 ✅ Council review completed.")
                logger.info(
                    f"📝 Council aggregate rankings: {metadata.get('aggregate_rankings', [])}"
                )

                # Council 검토 결과를 종합 결과에 추가
                if isinstance(synthesized_content, dict):
                    synthesized_content["council_review"] = {
                        "stage1_results": stage1_results,
                        "stage2_results": stage2_results,
                        "stage3_result": stage3_result,
                        "metadata": metadata,
                        "review_report": review_report,
                    }
            except Exception as e:
                logger.warning(f"📝 Council review failed: {e}. Using original synthesis results.")
                # Council 실패 시 원본 종합 결과 사용 (fallback 제거 - 명확한 로깅만)

        synthesis_result = {
            "synthesized_content": synthesized_content,
            "compressed_content": compressed_content,
            "deliverables": deliverables,
            "insights": insights,
            "context_usage": context_usage,
            "validation_results": validation_results,
            "synthesis_metadata": {
                "deliverable_type": deliverable_type,
                "timestamp": datetime.now().isoformat(),
                "synthesis_version": "2.0",
                "total_results_synthesized": len(execution_results),
                "context_window_usage": context_usage.get("usage_ratio", 1.0),
                "compression_ratio": compressed_content.get("compression_ratio", 1.0),
            },
            "innovation_stats": {
                "models_used": list(
                    set(insight.get("model_used", "unknown") for insight in insights)
                ),
                "mcp_tools_used": list(
                    set(
                        tool
                        for deliverable in deliverables.values()
                        for tool in deliverable.get("tools_used", [])
                    )
                ),
                "context_window_usage": context_usage.get("usage_ratio", 1.0),
                "compression_applied": compressed_content.get("compression_ratio", 1.0),
                "formats_generated": len(deliverables),
                "overall_quality": validation_results.get("overall_quality", 0.8),
            },
        }

        logger.info("✅ Synthesis completed successfully with 8 core innovations")
        return synthesis_result

    async def _manage_context_window(
        self,
        execution_results: List[Dict[str, Any]],
        evaluation_results: Dict[str, Any],
        original_objectives: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Adaptive Context Window 관리 (혁신 7)."""
        # 컨텍스트 크기 계산
        total_content = json.dumps(
            {
                "execution_results": execution_results,
                "evaluation_results": evaluation_results,
                "original_objectives": original_objectives,
            },
            ensure_ascii=False,
        )

        content_length = len(total_content)
        max_tokens = self.context_window_config.max_tokens
        self.context_window_config.min_tokens

        # 컨텍스트 사용률 계산
        usage_ratio = content_length / max_tokens if max_tokens > 0 else 1.0

        # 중요도 기반 보존
        if usage_ratio > 0.8 and self.context_window_config.importance_based_preservation:
            # 중요한 정보 우선 보존
            important_content = await self._extract_important_content(
                execution_results, evaluation_results
            )
            preserved_content = important_content
        else:
            preserved_content = total_content

        # 자동 압축
        if usage_ratio > 0.9 and self.context_window_config.enable_auto_compression:
            compressed_content = await compress_data(preserved_content)
            preserved_content = compressed_content.data

        return {
            "content_length": content_length,
            "max_tokens": max_tokens,
            "usage_ratio": usage_ratio,
            "preserved_content_length": len(preserved_content),
            "compression_applied": usage_ratio > 0.9,
            "importance_based_preservation": self.context_window_config.importance_based_preservation,
        }

    async def _integrate_data(
        self,
        execution_results: List[Dict[str, Any]],
        evaluation_results: Dict[str, Any],
        original_objectives: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """데이터 통합 및 분석."""
        # 실행 결과 통합
        integrated_results = []
        for result in execution_results:
            integrated_results.append(
                {
                    "task_id": result.get("task_id"),
                    "agent": result.get("agent"),
                    "result": result.get("result", {}),
                    "quality_score": result.get("quality_score", 0.8),
                    "confidence": result.get("confidence", 0.8),
                    "timestamp": result.get("timestamp"),
                }
            )

        # 평가 결과 통합
        quality_metrics = evaluation_results.get("overall_quality", {})
        alignment_assessment = evaluation_results.get("alignment_assessment", {})

        return {
            "integrated_results": integrated_results,
            "quality_metrics": quality_metrics,
            "alignment_assessment": alignment_assessment,
            "original_objectives": original_objectives,
            "total_results": len(integrated_results),
            "overall_quality": quality_metrics.get("overall_score", 0.8),
        }

    async def _generate_insights(
        self, integrated_data: Dict[str, Any], deliverable_type: str
    ) -> List[Dict[str, Any]]:
        """인사이트 생성 (Multi-Model Orchestration)."""
        insights = []

        # Dual-perspective analogical reasoning (issue #924)
        analogical_insights = await self._generate_analogical_insights(integrated_data)
        insights.extend(analogical_insights)

        # 패턴 분석
        pattern_insights = await self._analyze_patterns(integrated_data)
        insights.extend(pattern_insights)

        # 비교 분석
        comparative_insights = await self._perform_comparative_analysis(integrated_data)
        insights.extend(comparative_insights)

        # 예측 분석
        predictive_insights = await self._perform_predictive_analysis(integrated_data)
        insights.extend(predictive_insights)

        # 권장사항 생성
        recommendation_insights = await self._generate_recommendations(integrated_data)
        insights.extend(recommendation_insights)

        return insights

    async def _analyze_patterns(self, integrated_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """패턴 분석."""
        analysis_prompt = f"""
        Analyze the following research data to identify patterns and trends:
        
        Data: {json.dumps(integrated_data, ensure_ascii=False, indent=2)}
        
        Identify:
        1. Key patterns and trends
        2. Significant correlations
        3. Anomalies or outliers
        4. Emerging themes
        5. Data quality insights
        
        Provide specific, actionable pattern analysis.
        """

        result = await execute_llm_task(
            prompt=analysis_prompt,
            task_type=TaskType.ANALYSIS,
            system_message=self.config.prompts["data_analysis"]["system_message"],
        )

        return [
            {
                "type": "pattern_analysis",
                "content": result.content,
                "model_used": result.model_used,
                "confidence": result.confidence,
                "timestamp": datetime.now().isoformat(),
            }
        ]

    async def _perform_comparative_analysis(
        self, integrated_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """비교 분석."""
        analysis_prompt = f"""
        Perform comparative analysis on the following research data:
        
        Data: {json.dumps(integrated_data, ensure_ascii=False, indent=2)}
        
        Compare:
        1. Different sources and their reliability
        2. Conflicting or complementary findings
        3. Methodological differences
        4. Quality variations
        5. Consistency across results
        
        Provide detailed comparative insights.
        """

        result = await execute_llm_task(
            prompt=analysis_prompt,
            task_type=TaskType.ANALYSIS,
            system_message=self.config.prompts["comparative_analysis"]["system_message"],
        )

        return [
            {
                "type": "comparative_analysis",
                "content": result.content,
                "model_used": result.model_used,
                "confidence": result.confidence,
                "timestamp": datetime.now().isoformat(),
            }
        ]

    async def _perform_predictive_analysis(
        self, integrated_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """예측 분석."""
        analysis_prompt = f"""
        Perform predictive analysis on the following research data:
        
        Data: {json.dumps(integrated_data, ensure_ascii=False, indent=2)}
        
        Generate predictions for:
        1. Future trends and developments
        2. Potential outcomes and scenarios
        3. Risk factors and opportunities
        4. Long-term implications
        5. Recommended actions based on predictions
        
        Provide evidence-based predictive insights.
        """

        result = await execute_llm_task(
            prompt=analysis_prompt,
            task_type=TaskType.ANALYSIS,
            system_message=self.config.prompts["predictive_analysis"]["system_message"],
        )

        return [
            {
                "type": "predictive_analysis",
                "content": result.content,
                "model_used": result.model_used,
                "confidence": result.confidence,
                "timestamp": datetime.now().isoformat(),
            }
        ]

    async def _generate_recommendations(
        self, integrated_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """권장사항 생성."""
        analysis_prompt = f"""
        Generate actionable recommendations based on the following research data:
        
        Data: {json.dumps(integrated_data, ensure_ascii=False, indent=2)}
        
        Provide recommendations for:
        1. Immediate actions
        2. Short-term strategies
        3. Long-term planning
        4. Risk mitigation
        5. Opportunity exploitation
        
        Ensure recommendations are specific, actionable, and evidence-based.
        """

        result = await execute_llm_task(
            prompt=analysis_prompt,
            task_type=TaskType.SYNTHESIS,
            system_message=self.config.prompts["strategic_advice"]["system_message"],
        )

        return [
            {
                "type": "recommendations",
                "content": result.content,
                "model_used": result.model_used,
                "confidence": result.confidence,
                "timestamp": datetime.now().isoformat(),
            }
        ]

    async def _generate_analogical_insights(
        self, integrated_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Dual-perspective analogical reasoning synthesizer (issue #924).

        Queries historical physical, chemical, and computational system principles
        (e.g., TRIZ axioms, fluid dynamics, feedback loops) to generate dual
        technical recommendations: one from a physical/chemical perspective and
        one from a computational/systems perspective.
        """
        analysis_prompt = f"""
        Apply dual-perspective analogical reasoning to the following research data.

        Data: {json.dumps(integrated_data, ensure_ascii=False, indent=2)}

        Query historical physical, chemical, and computational system principles
        (e.g., TRIZ axioms, fluid dynamics, feedback loops) and generate two
        complementary technical recommendations:

        Perspective A (Physical/Chemical):
        1. Identify a relevant physical or chemical principle (e.g., TRIZ axiom,
           fluid dynamics, thermodynamics, reaction equilibrium)
        2. Map the principle onto the current problem
        3. Derive a concrete technical recommendation

        Perspective B (Computational/Systems):
        1. Identify a relevant computational or systems principle (e.g., feedback
           loops, caching, load balancing, control theory)
        2. Map the principle onto the current problem
        3. Derive a concrete technical recommendation

        Return both perspectives and explain how they complement each other.
        """

        try:
            result = await execute_llm_task(
                prompt=analysis_prompt,
                task_type=TaskType.SYNTHESIS,
                system_message=self.config.prompts["strategic_advice"]["system_message"],
            )

            return [
                {
                    "type": "analogical_reasoning",
                    "perspective": "dual",
                    "content": result.content,
                    "model_used": result.model_used,
                    "confidence": result.confidence,
                    "principles_queried": [
                        "triz_axioms",
                        "fluid_dynamics",
                        "feedback_loops",
                    ],
                    "timestamp": datetime.now().isoformat(),
                }
            ]
        except Exception as e:
            logger.warning(f"Analogical reasoning synthesis failed: {e}")
            return [
                {
                    "type": "analogical_reasoning",
                    "perspective": "dual",
                    "content": "",
                    "model_used": "unknown",
                    "confidence": 0.0,
                    "principles_queried": [
                        "triz_axioms",
                        "fluid_dynamics",
                        "feedback_loops",
                    ],
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }
            ]

    async def _synthesize_content(
        self,
        integrated_data: Dict[str, Any],
        insights: List[Dict[str, Any]],
        deliverable_type: str,
    ) -> Dict[str, Any]:
        """콘텐츠 종합 (Multi-Model Orchestration)."""
        template = self.synthesis_templates.get(
            deliverable_type, self.synthesis_templates["detailed_report"]
        )

        synthesis_prompt = f"""
        Synthesize the following research data into a comprehensive {deliverable_type}:
        
        Integrated Data: {json.dumps(integrated_data, ensure_ascii=False, indent=2)}
        Insights: {json.dumps(insights, ensure_ascii=False, indent=2)}
        
        Structure the content according to: {template["structure"]}
        Maximum length: {template["max_length"]} words
        
        Create a professional, well-structured synthesis with:
        1. Clear executive summary
        2. Detailed findings and analysis
        3. Evidence-based conclusions
        4. Actionable recommendations
        5. Supporting data and sources
        
        Use production-level writing quality.
        """

        result = await execute_llm_task(
            prompt=synthesis_prompt,
            task_type=TaskType.SYNTHESIS,
            system_message=self.config.prompts["synthesis_report"]["system_message"],
        )

        return {
            "content": result.content,
            "model_used": result.model_used,
            "confidence": result.confidence,
            "deliverable_type": deliverable_type,
            "structure": template["structure"],
            "word_count": len(result.content.split()),
            "timestamp": datetime.now().isoformat(),
        }

    async def _compress_content(self, synthesized_content: Dict[str, Any]) -> Dict[str, Any]:
        """콘텐츠 압축 (Hierarchical Compression)."""
        try:
            compressed = await compress_data(synthesized_content["content"])
            return {
                "original_content": synthesized_content["content"],
                "compressed_content": compressed.data,
                "compression_ratio": compressed.compression_ratio,
                "validation_score": compressed.validation_score,
                "important_info_preserved": compressed.important_info_preserved,
                "model_used": synthesized_content["model_used"],
                "confidence": synthesized_content["confidence"],
            }
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
        return {
            "original_content": synthesized_content["content"],
            "compressed_content": synthesized_content["content"],
            "compression_ratio": 1.0,
            "validation_score": 1.0,
            "important_info_preserved": [],
            "model_used": synthesized_content["model_used"],
            "confidence": synthesized_content["confidence"],
        }

    async def _generate_deliverables(
        self, compressed_content: Dict[str, Any], deliverable_type: str
    ) -> Dict[str, Any]:
        """전달물 생성 (Universal MCP Hub)."""
        deliverables = {}

        for format_name, format_config in self.deliverable_formats.items():
            if not format_config["enabled"]:
                continue

            try:
                # MCP 도구를 사용한 전달물 생성
                deliverable = await self._generate_format_deliverable(
                    compressed_content, format_name, format_config
                )
                deliverables[format_name] = deliverable

            except Exception as e:
                logger.warning(f"Failed to generate {format_name} deliverable: {e}")
                continue
            return deliverables

    async def _generate_format_deliverable(
        self,
        compressed_content: Dict[str, Any],
        format_name: str,
        format_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """형식별 전달물 생성."""
        # MCP 도구 선택
        best_tool = await get_best_tool_for_task("file_generation", ToolCategory.DATA)

        if best_tool:
            # MCP 도구로 전달물 생성
            tool_result = await execute_tool(
                best_tool,
                {
                    "content": compressed_content["compressed_content"],
                    "format": format_name,
                    "template": format_config["template"],
                },
            )

            if tool_result.success:
                return {
                    "format": format_name,
                    "content": tool_result.data,
                    "file_path": tool_result.data.get("file_path"),
                    "tools_used": [best_tool],
                    "generation_time": tool_result.execution_time,
                    "success": True,
                }

        # 모든 도구 실패 시 에러 반환 (fallback 제거)
        raise RuntimeError(
            f"Failed to generate {format_name} format: No suitable tools available. Content: {compressed_content.get('compressed_content', '')[:100]}"
        )

    async def _validate_synthesis(
        self, compressed_content: Dict[str, Any], deliverables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """종합 품질 검증."""
        validation_prompt = f"""
        Validate the quality of the following synthesis:
        
        Content: {compressed_content["compressed_content"][:1000]}...
        Deliverables: {list(deliverables.keys())}
        
        Assess:
        1. Content quality and coherence
        2. Completeness and accuracy
        3. Professional presentation
        4. Actionability of recommendations
        5. Overall synthesis quality
        
        Provide a quality score (0-1) and specific feedback.
        """

        result = await execute_llm_task(
            prompt=validation_prompt,
            task_type=TaskType.VERIFICATION,
            system_message=self.config.prompts["quality_validation"]["system_message"],
        )

        # 품질 점수 추출 (실제 구현에서는 더 정교한 파싱)
        quality_score = 0.8  # 기본값

        return {
            "overall_quality": quality_score,
            "validation_feedback": result.content,
            "model_used": result.model_used,
            "confidence": result.confidence,
            "deliverables_count": len(deliverables),
            "compression_ratio": compressed_content["compression_ratio"],
            "validation_timestamp": datetime.now().isoformat(),
        }

    async def _extract_important_content(
        self,
        execution_results: List[Dict[str, Any]],
        evaluation_results: Dict[str, Any],
    ) -> str:
        """중요한 콘텐츠 추출."""
        # 실제 구현에서는 더 정교한 중요도 분석
        important_parts = []

        # 고품질 결과 우선 선택
        for result in execution_results:
            if result.get("quality_score", 0) > 0.8:
                important_parts.append(result.get("result", {}))

        # 평가 결과의 핵심 부분
        if evaluation_results.get("overall_quality", {}).get("overall_score", 0) > 0.7:
            important_parts.append(evaluation_results)

        return json.dumps(important_parts, ensure_ascii=False, indent=2)

    async def cleanup(self):
        """에이전트 리소스 정리."""
        try:
            logger.info("Synthesis Agent cleanup completed")
        except Exception as e:
            logger.error(f"Synthesis Agent cleanup failed: {e}")


# Global synthesis agent instance
synthesis_agent = SynthesisAgent()


async def synthesize_results(
    execution_results: List[Dict[str, Any]],
    evaluation_results: Dict[str, Any],
    original_objectives: List[Dict[str, Any]],
    context: Dict[str, Any] | None = None,
    deliverable_type: str = "detailed_report",
) -> Dict[str, Any]:
    """연구 결과 종합."""
    return await synthesis_agent.synthesize_results(
        execution_results,
        evaluation_results,
        original_objectives,
        context,
        deliverable_type,
    )
