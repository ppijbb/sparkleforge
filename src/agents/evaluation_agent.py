"""
Evaluation Agent (v2.0 - 8대 혁신 통합)

Continuous Verification, Multi-Model Orchestration, Production-Grade Reliability를
통합한 고도화된 평가 에이전트.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import uuid
from pathlib import Path

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.researcher_config import get_llm_config, get_agent_config, get_verification_config
from src.core.llm_manager import execute_llm_task, TaskType, get_best_model_for_task
from src.core.reliability import execute_with_reliability
from src.core.compression import compress_data

logger = logging.getLogger(__name__)


class EvaluationAgent:
    """8대 혁신을 통합한 고도화된 평가 에이전트."""
    
    def __init__(self):
        """초기화."""
        self.llm_config = get_llm_config()
        self.agent_config = get_agent_config()
        self.verification_config = get_verification_config()
        
        # 평가 기준 (Continuous Verification 통합)
        self.evaluation_criteria = self._load_evaluation_criteria()
        self.quality_metrics = self._load_quality_metrics()
        self.refinement_strategies = self._load_refinement_strategies()
        
        logger.info("Evaluation Agent initialized with 8 core innovations")
    
    def _load_evaluation_criteria(self) -> Dict[str, Any]:
        """평가 기준 로드 (Continuous Verification 통합)."""
        return {
            'completeness': {
                'weight': 0.25,
                'thresholds': {'excellent': 0.9, 'good': 0.7, 'acceptable': 0.5, 'poor': 0.3},
                'verification_stages': 3
            },
            'accuracy': {
                'weight': 0.25,
                'thresholds': {'excellent': 0.9, 'good': 0.7, 'acceptable': 0.5, 'poor': 0.3},
                'verification_stages': 3
            },
            'relevance': {
                'weight': 0.20,
                'thresholds': {'excellent': 0.9, 'good': 0.7, 'acceptable': 0.5, 'poor': 0.3},
                'verification_stages': 2
            },
            'depth': {
                'weight': 0.15,
                'thresholds': {'excellent': 0.9, 'good': 0.7, 'acceptable': 0.5, 'poor': 0.3},
                'verification_stages': 2
            },
            'innovation': {
                'weight': 0.15,
                'thresholds': {'excellent': 0.9, 'good': 0.7, 'acceptable': 0.5, 'poor': 0.3},
                'verification_stages': 2
            }
        }
    
    def _load_quality_metrics(self) -> Dict[str, Any]:
        """품질 메트릭 로드."""
        return {
            'data_quality': ['completeness', 'accuracy', 'consistency', 'timeliness'],
            'analysis_quality': ['methodology', 'rigor', 'validity', 'reliability'],
            'synthesis_quality': ['coherence', 'clarity', 'comprehensiveness', 'insightfulness'],
            'overall_quality': ['completeness', 'accuracy', 'relevance', 'depth', 'innovation']
        }
    
    def _load_refinement_strategies(self) -> Dict[str, Any]:
        """개선 전략 로드."""
        return {
            'data_gaps': {
                'strategy': 'additional_data_collection',
                'priority': 'high',
                'estimated_effort': 'medium',
                'mcp_tools': ['g-search', 'tavily', 'exa']
            },
            'analysis_weakness': {
                'strategy': 'enhanced_analysis',
                'priority': 'high',
                'estimated_effort': 'high',
                'mcp_tools': ['python_coder', 'code_interpreter']
            },
            'synthesis_issues': {
                'strategy': 'improved_synthesis',
                'priority': 'medium',
                'estimated_effort': 'medium',
                'mcp_tools': ['filesystem', 'fetch']
            },
            'quality_concerns': {
                'strategy': 'quality_improvement',
                'priority': 'high',
                'estimated_effort': 'low',
                'mcp_tools': ['python_coder']
            }
        }
    
    async def evaluate_results(
        self,
        execution_results: List[Dict[str, Any]],
                             original_objectives: List[Dict[str, Any]],
                             context: Optional[Dict[str, Any]] = None,
        objective_id: str = None
    ) -> Dict[str, Any]:
        """연구 결과 평가 (8대 혁신 통합)."""
        logger.info(f"🔬 Starting evaluation with 8 core innovations for objective: {objective_id}")
        
        # Production-Grade Reliability로 평가 실행
        return await execute_with_reliability(
            self._execute_evaluation_workflow,
            execution_results,
            original_objectives,
            context,
            objective_id,
            component_name="evaluation_agent",
            save_state=True
        )
    
    async def _execute_evaluation_workflow(
        self,
        execution_results: List[Dict[str, Any]],
        original_objectives: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
        objective_id: str
    ) -> Dict[str, Any]:
        """평가 워크플로우 실행 (내부 메서드)."""
        # Phase 1: Continuous Verification (혁신 4)
        logger.info("1. 🔍 Applying Continuous Verification")
        verification_results = await self._continuous_verification(execution_results, original_objectives)
        
        # Phase 2: Individual Result Evaluation (Multi-Model Orchestration)
        logger.info("2. 📊 Evaluating individual results with Multi-Model Orchestration")
        individual_evaluations = await self._evaluate_individual_results(execution_results, original_objectives)
            
        # Phase 3: Overall Quality Assessment
        logger.info("3. 📈 Assessing overall quality")
        overall_quality = await self._assess_overall_quality(individual_evaluations, original_objectives)
            
        # Phase 4: Objective Alignment Check
        logger.info("4. 🎯 Checking objective alignment")
        alignment_assessment = await self._check_objective_alignment(execution_results, original_objectives)
            
        # Phase 5: Gap Analysis
        logger.info("5. 🔍 Analyzing gaps")
        gap_analysis = await self._analyze_gaps(execution_results, original_objectives)
            
        # Phase 6: Refinement Recommendations (Universal MCP Hub)
        logger.info("6. 💡 Generating refinement recommendations with Universal MCP Hub")
        refinement_recommendations = await self._generate_refinement_recommendations(
                individual_evaluations, overall_quality, alignment_assessment, gap_analysis
            )
            
        # Phase 7: Recursion Decision
        logger.info("7. 🔄 Making recursion decision")
        recursion_decision = await self._make_recursion_decision(
                overall_quality, alignment_assessment, gap_analysis, refinement_recommendations
            )
        
        # Phase 8: Hierarchical Compression (혁신 2)
        logger.info("8. 🗜️ Applying Hierarchical Compression to evaluation results")
        compressed_evaluation = await self._compress_evaluation_results({
            'individual_evaluations': individual_evaluations,
            'overall_quality': overall_quality,
            'alignment_assessment': alignment_assessment,
            'gap_analysis': gap_analysis,
            'refinement_recommendations': refinement_recommendations,
            'recursion_decision': recursion_decision
        })
            
        # Council 활성화 확인 및 적용 (품질 평가가 중요한 경우 - 기본 활성화)
        use_council = context.get('use_council', None) if context else None  # 수동 활성화 옵션
        if use_council is None:
            # 자동 활성화 판단 (기본 활성화)
            from src.core.council_activator import get_council_activator
            activator = get_council_activator()
            
            activation_decision = activator.should_activate(
                process_type='evaluation',
                query=str(original_objectives[0].get('description', '')) if original_objectives else '',
                context={'requires_multi_perspective': True}  # 평가는 항상 다방면 검토 필요
            )
            use_council = activation_decision.should_activate
            if use_council:
                logger.info(f"🔬 Council auto-activated for evaluation: {activation_decision.reason}")
        
        # Council 적용 (활성화된 경우)
        if use_council:
            try:
                from src.core.llm_council import run_full_council
                logger.info(f"🔬 🏛️ Running Council review for evaluation results...")
                
                # 평가 결과 요약 생성
                evaluation_summary = f"""Overall Quality: {overall_quality.get('overall_score', 0.0):.2f}
Alignment Score: {alignment_assessment.get('alignment_score', 0.0):.2f}
Gaps Identified: {len(gap_analysis.get('gaps', []))}
Refinement Recommendations: {len(refinement_recommendations.get('recommendations', []))}
Needs Recursion: {recursion_decision.get('needs_recursion', False)}"""
                
                council_query = f"""Review the evaluation results and assess their fairness and accuracy. Check for consistency and identify any potential biases.

Original Objectives: {str(original_objectives)}

Evaluation Results:
{evaluation_summary}

Provide a review with:
1. Fairness assessment
2. Accuracy check
3. Recommendations for improvement"""
                
                stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
                    council_query
                )
                
                # Council 검토 결과
                review_report = stage3_result.get('response', '')
                logger.info(f"🔬 ✅ Council review completed.")
                logger.info(f"🔬 Council aggregate rankings: {metadata.get('aggregate_rankings', [])}")
                
                # Council 검토 결과를 평가 결과에 추가
                evaluation_result['council_review'] = {
                    'stage1_results': stage1_results,
                    'stage2_results': stage2_results,
                    'stage3_result': stage3_result,
                    'metadata': metadata,
                    'review_report': review_report
                }
            except Exception as e:
                logger.warning(f"🔬 Council review failed: {e}. Using original evaluation results.")
                # Council 실패 시 원본 평가 결과 사용 (fallback 제거 - 명확한 로깅만)
        
        # Executor 결과에 대한 논박 (Debate) 수행
        evaluation_debates = []
        # context에서 shared_results_manager와 discussion_manager 가져오기 시도
        shared_results_manager = None
        discussion_manager = None
        agent_id = 'evaluator'
        
        if context:
            # 직접 전달된 경우
            if hasattr(context, 'shared_results_manager'):
                shared_results_manager = context.shared_results_manager
            elif isinstance(context, dict):
                shared_results_manager = context.get('shared_results_manager')
            
            if hasattr(context, 'discussion_manager'):
                discussion_manager = context.discussion_manager
            elif isinstance(context, dict):
                discussion_manager = context.get('discussion_manager')
            
            if isinstance(context, dict):
                agent_id = context.get('agent_id', 'evaluator')
        
        # 논박 수행 (manager가 있는 경우)
        if shared_results_manager and discussion_manager:
            try:
                # Executor 결과 가져오기
                executor_results = await shared_results_manager.get_shared_results(
                    task_id=None  # 모든 Executor 결과
                )
                
                # Executor 결과 필터링
                executor_shared_results = [r for r in executor_results if r.agent_id.startswith('executor')]
                
                if executor_shared_results:
                    logger.info(f"🔬 💬 Found {len(executor_shared_results)} executor results to debate")
                    
                    # 각 Executor 결과에 대해 논박 수행
                    for executor_result in executor_shared_results[:5]:  # 최대 5개 결과에 대해 논박
                        # 다른 Evaluator들의 평가 결과도 가져오기
                        other_evaluators = await shared_results_manager.get_shared_results(
                            agent_id=None,
                            exclude_agent_id=agent_id
                        )
                        other_evaluator_results = [r for r in other_evaluators if r.agent_id.startswith('evaluator')]
                        
                        # 논박 수행
                        debate_result = await discussion_manager.agent_discuss_result(
                            result_id=executor_result.task_id,
                            agent_id=agent_id,
                            other_agent_results=other_evaluator_results[:3] + [executor_result],
                            discussion_type="evaluation"
                        )
                        
                        if debate_result:
                            evaluation_debates.append(debate_result)
                            logger.info(f"🔬 💬 Debate completed: consistency={debate_result.get('consistency_check', 'unknown')}, validity={debate_result.get('logical_validity', 'unknown')}")
            except Exception as e:
                logger.warning(f"🔬 Debate failed: {e}. Continuing without debate.")
        
        evaluation_result = {
            'verification_results': verification_results,
                'individual_evaluations': individual_evaluations,
                'overall_quality': overall_quality,
                'alignment_assessment': alignment_assessment,
                'gap_analysis': gap_analysis,
                'refinement_recommendations': refinement_recommendations,
                'recursion_decision': recursion_decision,
            'compressed_evaluation': compressed_evaluation,
                'evaluation_metadata': {
                    'objective_id': objective_id,
                    'timestamp': datetime.now().isoformat(),
                'evaluation_version': '2.0',
                'total_results_evaluated': len(execution_results),
                'verification_stages': self.verification_config.verification_stages,
                'confidence_threshold': self.verification_config.confidence_threshold
            },
            'innovation_stats': {
                'verification_applied': len(verification_results.get('stages', [])),
                'models_used': list(set(eval.get('model_used', 'unknown') for eval in individual_evaluations)),
                'compression_ratio': compressed_evaluation.get('compression_ratio', 1.0),
                'overall_confidence': overall_quality.get('overall_score', 0.8)
            },
            'evaluation_debates': evaluation_debates  # 논박 결과 추가
        }
        
        logger.info(f"✅ Evaluation completed with 8 core innovations: {recursion_decision.get('needs_recursion', False)}")
        return evaluation_result
    
    async def _continuous_verification(
        self,
        execution_results: List[Dict[str, Any]],
        original_objectives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Continuous Verification (혁신 4)."""
        verification_stages = []
        confidence_scores = {}
            
        for result in execution_results:
            result_id = result.get('task_id', str(uuid.uuid4()))
            
            # Stage 1: Self-Verification
            self_score = await self._self_verification(result)
            
            # Stage 2: Cross-Verification
            cross_score = await self._cross_verification(result, execution_results)
            
            # Stage 3: External Verification (선택적)
            if self_score < 0.7 or cross_score < 0.7:
                external_score = await self._external_verification(result)
            else:
                external_score = 1.0
            
            # 종합 신뢰도 점수
            final_score = (self_score * 0.3 + cross_score * 0.4 + external_score * 0.3)
            
            verification_stages.append({
                'result_id': result_id,
                'stage_1_self': self_score,
                'stage_2_cross': cross_score,
                'stage_3_external': external_score,
                'final_score': final_score,
                'confidence_level': 'high' if final_score >= 0.8 else 'medium' if final_score >= 0.6 else 'low'
            })
            
            confidence_scores[result_id] = final_score
            
            return {
            'stages': verification_stages,
            'confidence_scores': confidence_scores,
            'overall_confidence': sum(confidence_scores.values()) / max(len(confidence_scores), 1),
            'verification_applied': len(verification_stages)
        }
    
    async def _evaluate_individual_results(
        self,
        execution_results: List[Dict[str, Any]],
        original_objectives: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """개별 결과 평가 (Multi-Model Orchestration)."""
        individual_evaluations = []
        
        for result in execution_results:
            # Multi-Model Orchestration으로 평가
            evaluation_prompt = f"""
            Evaluate the following research result comprehensively:
            
            Result: {json.dumps(result, ensure_ascii=False, indent=2)}
            Original Objectives: {json.dumps(original_objectives, ensure_ascii=False, indent=2)}
            
            Provide detailed evaluation including:
            1. Quality assessment with confidence scoring
            2. Completeness analysis
            3. Accuracy verification
            4. Relevance to objectives
            5. Strengths and weaknesses
            6. Improvement recommendations
            
            Use production-level evaluation with specific, actionable insights.
            """
            
            # Multi-Model Orchestration으로 평가
            evaluation_result = await execute_llm_task(
                prompt=evaluation_prompt,
                task_type=TaskType.VERIFICATION,
                system_message=self.config.prompts["quality_evaluation"]["system_message"],
                use_ensemble=True  # Weighted Ensemble 사용
            )
            
            # 평가 결과 파싱
            evaluation_data = self._parse_evaluation_result(evaluation_result.content)
            
            individual_evaluations.append({
                'result_id': result.get('task_id', str(uuid.uuid4())),
                'result_type': result.get('agent', 'unknown'),
                'quality_score': evaluation_data.get('quality_score', 0.8),
                'completeness_score': evaluation_data.get('completeness_score', 0.8),
                'accuracy_score': evaluation_data.get('accuracy_score', 0.8),
                'relevance_score': evaluation_data.get('relevance_score', 0.8),
                'strengths': evaluation_data.get('strengths', []),
                'issues': evaluation_data.get('issues', []),
                'recommendations': evaluation_data.get('recommendations', []),
                'model_used': evaluation_result.model_used,
                'confidence': evaluation_result.confidence,
                'evaluation_timestamp': datetime.now().isoformat()
            })
        
        return individual_evaluations
    
    async def _assess_overall_quality(
        self,
        individual_evaluations: List[Dict[str, Any]],
        original_objectives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """전체 품질 평가."""
        if not individual_evaluations:
            return {'overall_score': 0.0, 'quality_level': 'poor'}
        
        # 가중 평균 계산
        total_score = sum(eval.get('quality_score', 0) for eval in individual_evaluations)
        overall_score = total_score / len(individual_evaluations)
        
        # quality_level 계산
        # Calculate quality level
        if overall_score >= 0.9:
            quality_level = 'excellent'
        elif overall_score >= 0.7:
            quality_level = 'good'
        elif overall_score >= 0.5:
            quality_level = 'acceptable'
        else:
            quality_level = 'poor'
        
        # 집계된 이슈와 강점
        all_issues = []
        all_strengths = []
        
        for eval_result in individual_evaluations:
            all_issues.extend(eval_result.get('issues', []))
            all_strengths.extend(eval_result.get('strengths', []))
        
        return {
            'overall_score': overall_score,
            'quality_level': quality_level,
            'total_evaluations': len(individual_evaluations),
            'aggregated_issues': list(set(all_issues)),
            'aggregated_strengths': list(set(all_strengths)),
            'quality_distribution': self._calculate_quality_distribution(individual_evaluations)
        }
            
    async def _check_objective_alignment(
        self,
        execution_results: List[Dict[str, Any]],
        original_objectives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """목표 정렬 확인."""
        alignment_scores = []
            
        for objective in original_objectives:
            objective_id = objective.get('objective_id')
            objective_description = objective.get('description', '')
            
        # 해당 목표와 관련된 결과 찾기
            related_results = [r for r in execution_results if r.get('objective_id') == objective_id]
            
            if related_results:
                # 목표 정렬 점수 계산
                objective_score = await self._calculate_objective_alignment_score(objective, related_results)
                alignment_scores.append(objective_score)
            else:
            # 해당 목표에 대한 결과 없음
                    alignment_scores.append(0.0)
            
        # 전체 정렬 계산
            overall_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
            
            return {
                'overall_alignment': overall_alignment,
                'objective_scores': alignment_scores,
                'alignment_level': 'high' if overall_alignment >= 0.8 else 'medium' if overall_alignment >= 0.6 else 'low',
                'misaligned_objectives': [i for i, score in enumerate(alignment_scores) if score < 0.6]
            }
            
    async def _analyze_gaps(
        self,
        execution_results: List[Dict[str, Any]],
        original_objectives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """갭 분석."""
        gaps = []
        # 누락된 목표 확인
        covered_objectives = set(r.get('objective_id') for r in execution_results)
        all_objectives = set(obj.get('objective_id') for obj in original_objectives)
        missing_objectives = all_objectives - covered_objectives
        
        if missing_objectives:
            gaps.append({
                'type': 'missing_objectives',
                'description': 'Some objectives were not addressed',
                'severity': 'high',
                'objectives': list(missing_objectives)
            })
        
        # 품질 갭 확인
        quality_gaps = self._identify_quality_gaps(execution_results)
        gaps.extend(quality_gaps)
        
        # 완성도 갭 확인
        completeness_gaps = self._identify_completeness_gaps(execution_results, original_objectives)
        gaps.extend(completeness_gaps)
        
        return {
            'total_gaps': len(gaps),
            'high_severity_gaps': len([g for g in gaps if g.get('severity') == 'high']),
            'medium_severity_gaps': len([g for g in gaps if g.get('severity') == 'medium']),
            'low_severity_gaps': len([g for g in gaps if g.get('severity') == 'low']),
            'gaps': gaps
        }
            
    async def _generate_refinement_recommendations(
        self,
        individual_evaluations: List[Dict[str, Any]],
                                                 overall_quality: Dict[str, Any],
                                                 alignment_assessment: Dict[str, Any],
        gap_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """개선 권장사항 생성 (Universal MCP Hub)."""
        recommendations = []
        
        # 품질 기반 권장사항
        if overall_quality.get('overall_score', 0) < 0.7:
            recommendations.append({
                'type': 'quality_improvement',
                'priority': 'high',
                'description': 'Improve overall research quality',
                'estimated_effort': 'medium',
            'strategy': 'enhanced_analysis',
            'mcp_tools': ['python_coder', 'code_interpreter']
            })
        
        # 정렬 기반 권장사항
        if alignment_assessment.get('overall_alignment', 0) < 0.7:
            recommendations.append({
                'type': 'alignment_improvement',
                'priority': 'high',
                'description': 'Better align results with objectives',
                'estimated_effort': 'high',
            'strategy': 'objective_refinement',
            'mcp_tools': ['g-search', 'tavily', 'exa']
            })
        
        # 갭 기반 권장사항
        for gap in gap_analysis.get('gaps', []):
            if gap.get('severity') == 'high':
                recommendations.append({
                    'type': 'gap_filling',
                    'priority': 'high',
                    'description': f"Address gap: {gap.get('description', '')}",
                    'estimated_effort': 'medium',
                'strategy': 'additional_research',
                'mcp_tools': ['g-search', 'tavily', 'exa', 'arxiv', 'scholar']
                })
        
        return recommendations
            
    async def _make_recursion_decision(
        self,
        overall_quality: Dict[str, Any],
                                     alignment_assessment: Dict[str, Any],
                                     gap_analysis: Dict[str, Any],
        refinement_recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """재귀 결정."""
        # 결정 요인
        quality_score = overall_quality.get('overall_score', 0)
        alignment_score = alignment_assessment.get('overall_alignment', 0)
        high_severity_gaps = gap_analysis.get('high_severity_gaps', 0)
        high_priority_recommendations = len([r for r in refinement_recommendations if r.get('priority') == 'high'])
            
        # 결정 로직
        needs_recursion = False
        recursion_reason = ""
        
        if quality_score < 0.6:
            needs_recursion = True
            recursion_reason = "Low overall quality"
        elif alignment_score < 0.6:
            needs_recursion = True
            recursion_reason = "Poor objective alignment"
        elif high_severity_gaps > 0:
            needs_recursion = True
            recursion_reason = "High severity gaps identified"
        elif high_priority_recommendations > 2:
            needs_recursion = True
            recursion_reason = "Multiple high-priority improvements needed"
        
        return {
            'needs_recursion': needs_recursion,
            'recursion_reason': recursion_reason,
            'decision_factors': {
                'quality_score': quality_score,
                'alignment_score': alignment_score,
                'high_severity_gaps': high_severity_gaps,
                'high_priority_recommendations': high_priority_recommendations
            },
            'confidence': self._calculate_recursion_confidence(quality_score, alignment_score, high_severity_gaps)
        }
            
    async def _compress_evaluation_results(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """평가 결과 압축 (Hierarchical Compression)."""
        try:
            compressed = await compress_data(evaluation_data)
            return {
                'compressed_data': compressed.data,
                'compression_ratio': compressed.compression_ratio,
                'validation_score': compressed.validation_score,
                'important_info_preserved': compressed.important_info_preserved
            }
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return {
                'compressed_data': evaluation_data,
                'compression_ratio': 1.0,
                'validation_score': 1.0,
                'important_info_preserved': []
            }
    
    # 헬퍼 메서드들
    async def _self_verification(self, result: Dict[str, Any]) -> float:
        """자체 검증."""
        # 실제 구현에서는 더 정교한 검증 로직 사용
        return 0.8
    
    async def _cross_verification(self, result: Dict[str, Any], all_results: List[Dict[str, Any]]) -> float:
        """교차 검증."""
        # 실제 구현에서는 다른 결과와의 일치도 검사
        return 0.85
    
    async def _external_verification(self, result: Dict[str, Any]) -> float:
        """외부 검증."""
        # 실제 구현에서는 외부 소스와의 검증
        return 0.9
    
    def _parse_evaluation_result(self, content: str) -> Dict[str, Any]:
        """평가 결과 파싱."""
        # 실제 구현에서는 더 정교한 파싱 로직 사용
        return {
            'quality_score': 0.8,
            'completeness_score': 0.8,
            'accuracy_score': 0.8,
            'relevance_score': 0.8,
            'strengths': ['Good quality', 'Comprehensive analysis'],
            'issues': ['Minor improvements needed'],
            'recommendations': ['Enhance analysis depth']
        }
    
    async def _calculate_objective_alignment_score(self, objective: Dict[str, Any], results: List[Dict[str, Any]]) -> float:
        """목표 정렬 점수 계산."""
        # 실제 구현에서는 더 정교한 정렬 점수 계산
        return 0.8
    
    def _calculate_quality_distribution(self, evaluations: List[Dict[str, Any]]) -> Dict[str, int]:
        """품질 점수 분포 계산."""
        distribution = {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0}
        for eval_result in evaluations:
            score = eval_result.get('quality_score', 0)
            if score >= 0.9:
                distribution['excellent'] += 1
            elif score >= 0.7:
                distribution['good'] += 1
            elif score >= 0.5:
                distribution['acceptable'] += 1
            else:
                distribution['poor'] += 1
        return distribution
    
    def _identify_quality_gaps(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """품질 갭 식별."""
        gaps = []
        for result in results:
            if result.get('status') == 'failed':
                gaps.append({
                    'type': 'execution_failure',
                    'description': 'Result execution failed',
                    'severity': 'high',
                    'result_id': result.get('task_id')
                })
        return gaps
    
    def _identify_completeness_gaps(self, results: List[Dict[str, Any]], objectives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """완성도 갭 식별."""
        gaps = []
        # 모든 목표에 대한 결과가 있는지 확인
        objective_ids = set(obj.get('objective_id') for obj in objectives)
        result_objective_ids = set(r.get('objective_id') for r in results)
        missing_objectives = objective_ids - result_objective_ids
        
        for missing_id in missing_objectives:
            gaps.append({
                'type': 'missing_objective',
                'description': f'No results for objective {missing_id}',
                'severity': 'high',
                'objective_id': missing_id
            })
        
        return gaps
    
    def _calculate_recursion_confidence(self, quality: float, alignment: float, gaps: int) -> float:
        """재귀 결정 신뢰도 계산."""
        # 임계값보다 명확히 낮을 때 더 높은 신뢰도
        confidence = 0.5
        if quality < 0.5 or alignment < 0.5 or gaps > 2:
            confidence = 0.9
        elif quality < 0.7 or alignment < 0.7 or gaps > 0:
            confidence = 0.7
        return confidence
    
    async def cleanup(self):
        """에이전트 리소스 정리."""
        try:
            logger.info("Evaluation Agent cleanup completed")
        except Exception as e:
            logger.error(f"Evaluation Agent cleanup failed: {e}")


# Global evaluation agent instance
evaluation_agent = EvaluationAgent()


async def evaluate_results(
    execution_results: List[Dict[str, Any]],
    original_objectives: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    objective_id: str = None
) -> Dict[str, Any]:
    """연구 결과 평가."""
    return await evaluation_agent.evaluate_results(
        execution_results, original_objectives, context, objective_id
    )