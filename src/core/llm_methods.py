#!/usr/bin/env python3
"""
LLM-based methods for research tasks
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)


class LLMMethods:
    """LLM-based methods for autonomous research."""
    
    def __init__(self, llm):
        self.llm = llm
        self.decision_memory = []
    
    async def llm_decompose_tasks(self, objective: Any) -> Dict[str, Any]:
        """LLM-based task decomposition."""
        try:
            prompt = f"""
            연구 목표들을 구체적인 작업들로 분해하세요.
            
            목표들: {objective.analyzed_objectives}
            
            다음을 수행하세요:
            1. 각 목표를 실행 가능한 작업들로 분해
            2. 작업 간 의존성 관계 설정
            3. 각 작업에 적합한 에이전트 할당
            4. 작업 우선순위 설정
            5. 예상 소요 시간과 복잡도 평가
            
            JSON 형태로 응답하세요:
            {{
                "tasks": [
                    {{
                        "task_id": "unique_id",
                        "objective_id": "목표_id",
                        "description": "작업 설명",
                        "task_type": "data_collection|analysis|synthesis|validation",
                        "assigned_to": "agent_name",
                        "priority": 0.0-1.0,
                        "estimated_duration": "short|medium|long",
                        "complexity": "low|medium|high",
                        "dependencies": ["다른_작업_id"],
                        "success_criteria": ["기준1", "기준2"]
                    }}
                ],
                "assignments": [
                    {{
                        "task_id": "작업_id",
                        "agent": "에이전트명",
                        "assignment_reason": "할당 이유"
                    }}
                ],
                "strategy": "sequential|parallel|iterative|hierarchical"
            }}
            """
            
            response = await asyncio.to_thread(self.llm.generate_content, prompt)
            result = json.loads(response.text)
            
            # Store decision for learning
            self.decision_memory.append({
                'type': 'task_decomposition',
                'input': objective.analyzed_objectives,
                'output': result,
                'timestamp': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"LLM task decomposition failed: {e}")
            return {'tasks': [], 'assignments': []}
    
    async def llm_coordinate_execution(self, objective: Any, agents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """LLM-coordinated task execution."""
        try:
            execution_results = []
            
            # Execute tasks with LLM coordination
            for task in objective.decomposed_tasks:
                agent_name = task.get('assigned_to')
                if agent_name and agent_name in agents:
                    agent = agents[agent_name]
                    
                    # Get LLM guidance for task execution
                    guidance = await self.get_llm_execution_guidance(task, objective)
                    
                    # Execute task with guidance
                    if hasattr(agent, 'execute_task'):
                        result = await agent.execute_task(task, guidance, objective.objective_id)
                    else:
                        result = await agent.conduct_research(task, guidance, objective.objective_id)
                    
                    execution_results.append(result)
            
            return execution_results
            
        except Exception as e:
            logger.error(f"LLM-coordinated execution failed: {e}")
            return []
    
    async def llm_evaluate_results(self, objective: Any) -> Dict[str, Any]:
        """LLM-based result evaluation."""
        try:
            prompt = f"""
            실행 결과를 평가하고 개선 방안을 제시하세요.
            
            원래 목표들: {objective.analyzed_objectives}
            실행 결과: {objective.execution_results}
            
            다음을 수행하세요:
            1. 각 목표 달성도 평가 (0.0-1.0)
            2. 결과의 품질과 완성도 분석
            3. 부족한 부분이나 개선이 필요한 영역 식별
            4. 추가 작업이 필요한지 판단
            5. 전체적인 성공도 평가
            
            JSON 형태로 응답하세요:
            {{
                "overall_score": 0.0-1.0,
                "objective_scores": {{
                    "objective_id": 0.0-1.0
                }},
                "quality_metrics": {{
                    "completeness": 0.0-1.0,
                    "accuracy": 0.0-1.0,
                    "relevance": 0.0-1.0,
                    "depth": 0.0-1.0
                }},
                "improvement_areas": ["개선영역1", "개선영역2"],
                "needs_additional_work": true/false,
                "recommendations": ["추천사항1", "추천사항2"]
            }}
            """
            
            response = await asyncio.to_thread(self.llm.generate_content, prompt)
            result = json.loads(response.text)
            
            # Store decision for learning
            self.decision_memory.append({
                'type': 'result_evaluation',
                'input': {'objectives': objective.analyzed_objectives, 'results': objective.execution_results},
                'output': result,
                'timestamp': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"LLM result evaluation failed: {e}")
            return {'overall_score': 0.0, 'needs_recursion': False}
    
    async def llm_validate_results(self, objective: Any) -> Dict[str, Any]:
        """LLM-based result validation."""
        try:
            prompt = f"""
            결과가 원래 요청을 충족하는지 검증하세요.
            
            원래 요청: {objective.user_request}
            목표들: {objective.analyzed_objectives}
            실행 결과: {objective.execution_results}
            
            다음을 수행하세요:
            1. 원래 요청과 결과의 일치도 평가
            2. 누락된 중요한 요소들 식별
            3. 결과의 신뢰성과 정확성 검증
            4. 최종 검증 점수 산출
            
            JSON 형태로 응답하세요:
            {{
                "validation_score": 0.0-1.0,
                "alignment_score": 0.0-1.0,
                "completeness_score": 0.0-1.0,
                "accuracy_score": 0.0-1.0,
                "missing_elements": ["누락요소1", "누락요소2"],
                "is_valid": true/false,
                "validation_feedback": "검증 피드백"
            }}
            """
            
            response = await asyncio.to_thread(self.llm.generate_content, prompt)
            result = json.loads(response.text)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM result validation failed: {e}")
            return {'validation_score': 0.0, 'is_valid': False}
    
    async def llm_synthesize_deliverable(self, objective: Any) -> Dict[str, Any]:
        """LLM-based final synthesis."""
        try:
            prompt = f"""
            모든 결과를 종합하여 최종 보고서를 생성하세요.
            
            원래 요청: {objective.user_request}
            실행 결과: {objective.execution_results}
            평가 결과: {objective.evaluation_results}
            검증 결과: {objective.validation_results}
            
            다음을 포함한 종합 보고서를 작성하세요:
            1. 요약 및 핵심 발견사항
            2. 상세 분석 결과
            3. 결론 및 권고사항
            4. 참고자료 및 출처
            
            JSON 형태로 응답하세요:
            {{
                "deliverable_path": "파일경로",
                "deliverable_format": "markdown|pdf|html",
                "summary": "요약",
                "key_findings": ["발견사항1", "발견사항2"],
                "detailed_analysis": "상세분석",
                "conclusions": "결론",
                "recommendations": ["권고사항1", "권고사항2"],
                "references": ["참고자료1", "참고자료2"]
            }}
            """
            
            response = await asyncio.to_thread(self.llm.generate_content, prompt)
            result = json.loads(response.text)
            
            # Generate actual deliverable file
            deliverable_path = await self.generate_deliverable_file(result, objective)
            result['deliverable_path'] = deliverable_path
            
            return result
            
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return {'deliverable_path': None, 'error': str(e)}
    
    async def get_llm_execution_guidance(self, task: Dict[str, Any], objective: Any) -> Dict[str, Any]:
        """Get LLM guidance for task execution."""
        try:
            prompt = f"""
            작업 실행을 위한 구체적인 가이드라인을 제공하세요.
            
            작업: {task}
            목표: {objective.analyzed_objectives}
            
            다음을 포함한 실행 가이드를 제공하세요:
            1. 구체적인 실행 방법
            2. 주의사항과 제약조건
            3. 품질 기준
            4. 예상 결과물
            
            JSON 형태로 응답하세요.
            """
            
            response = await asyncio.to_thread(self.llm.generate_content, prompt)
            return json.loads(response.text)
            
        except Exception as e:
            logger.error(f"LLM execution guidance failed: {e}")
            return {}
    
    async def generate_deliverable_file(self, synthesis_result: Dict[str, Any], objective: Any) -> str:
        """Generate the actual deliverable file."""
        try:
            # Create output directory
            output_dir = Path("./outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename
            filename = f"research_report_{objective.objective_id}.md"
            file_path = output_dir / filename
            
            # Create markdown content
            content = f"""# 연구 보고서: {objective.user_request}

## 요약
{synthesis_result.get('summary', '')}

## 핵심 발견사항
{chr(10).join(f"- {finding}" for finding in synthesis_result.get('key_findings', []))}

## 상세 분석
{synthesis_result.get('detailed_analysis', '')}

## 결론
{synthesis_result.get('conclusions', '')}

## 권고사항
{chr(10).join(f"- {rec}" for rec in synthesis_result.get('recommendations', []))}

## 참고자료
{chr(10).join(f"- {ref}" for ref in synthesis_result.get('references', []))}

---
*생성일시: {datetime.now().isoformat()}*
*목표 ID: {objective.objective_id}*
"""
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Deliverable generated: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Deliverable generation failed: {e}")
            return None
