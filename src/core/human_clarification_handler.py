"""
Human Clarification Handler - Human-in-the-Loop 질문 생성 및 응답 처리

Planning 단계와 Task 실행 과정에서 불명확한 부분을 감지하고,
사용자에게 질문을 제시하여 선택지 또는 자연어 응답을 받아
작업을 명확하게 진행할 수 있도록 지원합니다.
"""

import logging
import uuid
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.llm_manager import execute_llm_task, TaskType
from src.core.researcher_config import get_llm_config

logger = logging.getLogger(__name__)


class HumanClarificationHandler:
    """
    Human-in-the-Loop 질문 생성 및 응답 처리 핸들러
    
    불명확한 부분을 감지하고, 적절한 질문을 생성하며,
    사용자 응답을 처리하여 계획/작업에 반영합니다.
    """
    
    def __init__(self):
        """HumanClarificationHandler 초기화"""
        self.llm_config = get_llm_config()
        logger.info("HumanClarificationHandler initialized")
    
    async def detect_ambiguities(
        self,
        user_query: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Planning 단계에서 불명확한 부분 감지
        
        Args:
            user_query: 사용자 쿼리
            context: 컨텍스트 정보 (objectives, domain 등)
            
        Returns:
            불명확한 부분 리스트
            [{"type": "time_range", "field": "...", "description": "..."}, ...]
        """
        try:
            # 간단한 쿼리는 건너뛰기 (무한 대기 방지)
            if len(user_query.split()) < 3:
                logger.debug("Simple query detected, skipping ambiguity detection")
                return []
            detection_prompt = f"""
            Analyze the following user query and identify any ambiguous or unclear aspects
            that would benefit from user clarification before creating a research plan.
            
            User Query: {user_query}
            
            Context:
            - Objectives: {context.get('objectives', [])}
            - Domain: {context.get('domain', {})}
            - Scope: {context.get('scope', {})}
            
            Identify ambiguities in the following areas:
            1. **Time Range**: "최신", "최근", "올해" 등의 시간 범위가 불명확한 경우
            2. **Scope/Depth**: "상세한", "깊이 있는", "포괄적인" 등의 범위/깊이가 불명확한 경우
            3. **Preference**: 사용자 선호도가 필요한 경우 (예: 형식, 스타일, 관점)
            4. **Priority**: 여러 주제 중 우선순위가 불명확한 경우
            5. **Output Format**: 원하는 출력 형식이 불명확한 경우
            6. **Resource Constraints**: 리소스 제약으로 선택이 필요한 경우
            
            For each ambiguity found, provide:
            - type: One of "time_range", "scope_depth", "preference", "priority", "output_format", "resource_constraint"
            - field: The specific field or aspect that is ambiguous
            - description: Why this is ambiguous and what clarification is needed
            - suggested_question: A suggested question to ask the user
            - suggested_options: If applicable, suggested choice options (empty list if natural language is better)
            
            Return your analysis in JSON format:
            {{
                "ambiguities": [
                    {{
                        "type": "time_range",
                        "field": "time_range",
                        "description": "The query mentions '최신 정보' but doesn't specify the time range",
                        "suggested_question": "어느 기간의 정보를 원하시나요?",
                        "suggested_options": [
                            {{"label": "최근 1개월", "value": "1_month"}},
                            {{"label": "최근 3개월", "value": "3_months"}},
                            {{"label": "최근 6개월", "value": "6_months"}},
                            {{"label": "최근 1년", "value": "1_year"}},
                            {{"label": "전체 기간", "value": "all"}}
                        ]
                    }}
                ]
            }}
            
            If no ambiguities are found, return an empty list.
            """
            
            # 타임아웃 설정 (15초)
            try:
                result = await asyncio.wait_for(
                    execute_llm_task(
                        prompt=detection_prompt,
                        task_type=TaskType.ANALYSIS,
                        system_message="You are an expert at identifying ambiguous requirements and determining when user clarification is needed."
                    ),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning("LLM timeout in detect_ambiguities, returning empty list")
                return []
            
            # 결과 파싱
            import json
            try:
                # JSON 추출 시도
                content = result.content or "{}"
                # JSON 코드 블록 제거
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(content)
                ambiguities = parsed.get("ambiguities", [])
                
                logger.info(f"Detected {len(ambiguities)} ambiguities")
                return ambiguities
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse ambiguity detection result: {e}")
                # 기본 파싱 시도
                if "ambiguities" in content.lower():
                    # 간단한 추출 시도
                    return []
                return []
                
        except Exception as e:
            logger.error(f"Error detecting ambiguities: {e}")
            return []
    
    async def generate_question(
        self,
        ambiguity: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        질문 생성
        
        Args:
            ambiguity: 불명확한 부분 정보
            context: 컨텍스트 정보
            
        Returns:
            질문 정보
            {
                "id": "question_id",
                "text": "질문 텍스트",
                "format": "choice" or "natural_language",
                "options": [...],  # 선택지 형식인 경우
                "type": "time_range",
                "field": "..."
            }
        """
        try:
            question_id = str(uuid.uuid4())
            ambiguity_type = ambiguity.get("type", "general")
            suggested_question = ambiguity.get("suggested_question", "")
            suggested_options = ambiguity.get("suggested_options", [])
            
            # 질문 형식 결정
            question_format = self.determine_question_format(ambiguity_type, suggested_options)
            
            # 질문 텍스트 개선 (LLM 사용)
            if not suggested_question:
                question_prompt = f"""
                Generate a clear, user-friendly question to clarify the following ambiguity:
                
                Type: {ambiguity_type}
                Field: {ambiguity.get('field', '')}
                Description: {ambiguity.get('description', '')}
                
                User Query Context: {context.get('user_request', '')}
                
                Generate a question that:
                1. Is clear and easy to understand
                2. Is polite and professional
                3. Helps the user understand what information is needed
                4. Is in Korean if the user query is in Korean, otherwise in English
                
                Return only the question text, no additional explanation.
                """
                
                # 타임아웃 설정 (10초)
                try:
                    result = await asyncio.wait_for(
                        execute_llm_task(
                            prompt=question_prompt,
                            task_type=TaskType.GENERATION,
                            system_message="You are an expert at creating clear, user-friendly questions."
                        ),
                        timeout=10.0
                    )
                    question_text = result.content.strip() if result.content else suggested_question
                except asyncio.TimeoutError:
                    logger.warning("LLM timeout in generate_question, using suggested question")
                    question_text = suggested_question
            else:
                question_text = suggested_question
            
            # 선택지 개선 (필요한 경우)
            if question_format == "choice" and suggested_options:
                # 선택지가 이미 있으면 그대로 사용
                options = suggested_options
            elif question_format == "choice" and not suggested_options:
                # 선택지 생성
                options = await self._generate_choice_options(ambiguity, context)
            else:
                options = []
            
            question = {
                "id": question_id,
                "text": question_text,
                "format": question_format,
                "options": options,
                "type": ambiguity_type,
                "field": ambiguity.get("field", ""),
                "description": ambiguity.get("description", ""),
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"Generated question: {question_id} ({question_format})")
            return question
            
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            # 기본 질문 반환
            return {
                "id": str(uuid.uuid4()),
                "text": "추가 정보가 필요합니다. 원하시는 내용을 알려주세요.",
                "format": "natural_language",
                "options": [],
                "type": "general",
                "field": "",
                "description": str(e),
                "created_at": datetime.now().isoformat()
            }
    
    def determine_question_format(
        self,
        ambiguity_type: str,
        suggested_options: List[Dict[str, Any]]
    ) -> str:
        """
        질문 형식 결정: 'choice' or 'natural_language'
        
        Args:
            ambiguity_type: 불명확한 부분 타입
            suggested_options: 제안된 선택지
            
        Returns:
            'choice' or 'natural_language'
        """
        # 선택지가 이미 있으면 choice 형식
        if suggested_options and len(suggested_options) > 0:
            return "choice"
        
        # 타입에 따라 기본 형식 결정
        choice_types = [
            "time_range",
            "priority",
            "output_format",
            "resource_constraint"
        ]
        
        if ambiguity_type in choice_types:
            return "choice"
        else:
            return "natural_language"
    
    async def _generate_choice_options(
        self,
        ambiguity: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        선택지 생성
        
        Args:
            ambiguity: 불명확한 부분 정보
            context: 컨텍스트 정보
            
        Returns:
            선택지 리스트
        """
        try:
            options_prompt = f"""
            Generate appropriate choice options for the following ambiguity:
            
            Type: {ambiguity.get('type', '')}
            Field: {ambiguity.get('field', '')}
            Description: {ambiguity.get('description', '')}
            
            User Query: {context.get('user_request', '')}
            
            Generate 3-5 clear, mutually exclusive options that cover common scenarios.
            Return in JSON format:
            {{
                "options": [
                    {{"label": "Option 1", "value": "value1"}},
                    {{"label": "Option 2", "value": "value2"}}
                ]
            }}
            """
            
            # 타임아웃 설정 (10초)
            try:
                result = await asyncio.wait_for(
                    execute_llm_task(
                        prompt=options_prompt,
                        task_type=TaskType.GENERATION,
                        system_message="You are an expert at creating clear, mutually exclusive choice options."
                    ),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("LLM timeout in _generate_choice_options, returning empty list")
                return []
            
            import json
            content = result.content or "{}"
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(content)
            return parsed.get("options", [])
            
        except Exception as e:
            logger.warning(f"Failed to generate choice options: {e}")
            return []
    
    async def process_user_response(
        self,
        question_id: str,
        response: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        사용자 응답 처리 및 검증
        
        Args:
            question_id: 질문 ID
            response: 사용자 응답 (선택지 값 또는 자연어 텍스트)
            context: 컨텍스트 정보
            
        Returns:
            처리된 응답 정보
            {
                "question_id": "...",
                "response": "...",
                "validated": True,
                "clarification": {...}
            }
        """
        try:
            # 응답 검증
            validated = True
            clarification = {
                "question_id": question_id,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            
            # 응답이 비어있는지 확인
            if not response or (isinstance(response, str) and not response.strip()):
                validated = False
                clarification["error"] = "응답이 비어있습니다."
                return {
                    "question_id": question_id,
                    "response": response,
                    "validated": False,
                    "clarification": clarification
                }
            
            # 응답 정규화
            if isinstance(response, str):
                response = response.strip()
            
            clarification["normalized_response"] = response
            clarification["validated"] = True
            
            logger.info(f"Processed user response for question {question_id}: {response}")
            
            return {
                "question_id": question_id,
                "response": response,
                "validated": validated,
                "clarification": clarification
            }
            
        except Exception as e:
            logger.error(f"Error processing user response: {e}")
            return {
                "question_id": question_id,
                "response": response,
                "validated": False,
                "clarification": {
                    "error": str(e)
                }
            }
    
    async def auto_select_response(
        self,
        question: Dict[str, Any],
        context: Dict[str, Any],
        shared_memory: Optional[Any] = None
    ) -> Any:
        """
        자동으로 응답 선택 (사용자 입력 없이 LLM이 결정)
        
        Args:
            question: 질문 정보
            context: 컨텍스트 정보
            shared_memory: 공유 메모리 (선택적)
            
        Returns:
            선택된 응답 (선택지 값 또는 자연어 텍스트)
        """
        try:
            question_text = question.get("text", "")
            question_format = question.get("format", "natural_language")
            options = question.get("options", [])
            question_type = question.get("type", "general")
            
            # 컨텍스트에서 사용자 요청 추출
            user_request = context.get("user_request", "")
            
            # LLM을 사용하여 자동 응답 생성
            if question_format == "choice" and options:
                # 선택지 형식인 경우 가장 적절한 선택지 선택
                options_text = "\n".join([
                    f"{i+1}. {opt.get('label', opt.get('value', ''))} (value: {opt.get('value', '')})"
                    for i, opt in enumerate(options)
                ])
                
                selection_prompt = f"""
                Based on the user's request and context, automatically select the most appropriate option.
                
                User Request: {user_request}
                
                Question: {question_text}
                
                Available Options:
                {options_text}
                
                Question Type: {question_type}
                
                Select the most appropriate option value based on:
                1. The user's original request
                2. Common sense and best practices
                3. The context of the research task
                
                Return only the option value (e.g., "1_month", "3_months", etc.), not the label.
                If none of the options seem appropriate, return the first option's value as a default.
                """
            else:
                # 자연어 형식인 경우 적절한 응답 생성
                selection_prompt = f"""
                Based on the user's request and context, automatically provide an appropriate response.
                
                User Request: {user_request}
                
                Question: {question_text}
                
                Question Type: {question_type}
                
                Provide a clear, concise response that:
                1. Addresses the question appropriately
                2. Is consistent with the user's original request
                3. Uses common sense and best practices
                
                Return only the response text, no additional explanation.
                """
            
            # 타임아웃 설정 (10초)
            try:
                result = await asyncio.wait_for(
                    execute_llm_task(
                        prompt=selection_prompt,
                        task_type=TaskType.ANALYSIS,
                        system_message="You are an expert at making reasonable decisions based on context and user intent."
                    ),
                    timeout=10.0
                )
                auto_response = result.content.strip() if result.content else ""
            except asyncio.TimeoutError:
                logger.warning("LLM timeout in auto_select_response, using default")
                auto_response = ""
            
            # 선택지 형식인 경우 값 검증
            if question_format == "choice" and options:
                # 응답이 선택지 값 중 하나인지 확인
                option_values = [opt.get("value", "") for opt in options]
                if auto_response in option_values:
                    return auto_response
                else:
                    # 응답이 라벨인 경우 값으로 변환
                    for opt in options:
                        if opt.get("label", "").lower() == auto_response.lower():
                            return opt.get("value", "")
                    # 기본값: 첫 번째 옵션
                    logger.warning(f"Auto-selected response '{auto_response}' not in options, using first option")
                    return options[0].get("value", "") if options else ""
            
            return auto_response
            
        except Exception as e:
            logger.error(f"Error in auto_select_response: {e}")
            # 기본값 반환
            if question.get("format") == "choice" and question.get("options"):
                return question["options"][0].get("value", "") if question["options"] else ""
            return "기본값"
    
    def apply_clarification(
        self,
        clarification: Dict[str, Any],
        plan_or_task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        명확화된 정보를 계획/작업에 적용
        
        Args:
            clarification: 명확화된 정보
            plan_or_task: 계획 또는 작업 정보
            
        Returns:
            업데이트된 계획/작업
        """
        try:
            updated = plan_or_task.copy()
            
            # 질문 타입에 따라 정보 적용
            question_type = clarification.get("type", "")
            response = clarification.get("response", "")
            
            if question_type == "time_range":
                updated["time_range"] = response
                updated["search_filters"] = updated.get("search_filters", {})
                updated["search_filters"]["time_range"] = response
                
            elif question_type == "scope_depth":
                updated["depth"] = response
                updated["analysis_level"] = response
                
            elif question_type == "preference":
                updated["user_preferences"] = updated.get("user_preferences", {})
                updated["user_preferences"][clarification.get("field", "")] = response
                
            elif question_type == "priority":
                updated["priorities"] = updated.get("priorities", [])
                if response not in updated["priorities"]:
                    updated["priorities"].append(response)
                    
            elif question_type == "output_format":
                updated["output_format"] = response
                
            elif question_type == "resource_constraint":
                updated["resource_constraints"] = updated.get("resource_constraints", {})
                updated["resource_constraints"][clarification.get("field", "")] = response
            
            # 명확화 정보 메타데이터 추가
            updated["clarifications"] = updated.get("clarifications", [])
            updated["clarifications"].append({
                "question_id": clarification.get("question_id"),
                "type": question_type,
                "response": response,
                "applied_at": datetime.now().isoformat()
            })
            
            logger.info(f"Applied clarification to plan/task: {question_type} = {response}")
            return updated
            
        except Exception as e:
            logger.error(f"Error applying clarification: {e}")
            return plan_or_task


# 싱글톤 인스턴스
_handler_instance: Optional[HumanClarificationHandler] = None


def get_clarification_handler() -> HumanClarificationHandler:
    """HumanClarificationHandler 싱글톤 인스턴스 반환"""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = HumanClarificationHandler()
    return _handler_instance

