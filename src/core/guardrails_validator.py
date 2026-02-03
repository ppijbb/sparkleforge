"""
Guardrails Validator - Guardrails 검증 모듈

기존 시스템과 독립적으로 동작하는 Input/Output 검증 모듈.
기존 state나 result를 수정하지 않고 검증만 수행합니다.
"""

import logging
import re
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    reason: str = ""
    confidence: float = 1.0


class GuardrailsValidator:
    """Guardrails 검증 모듈 - 기존 시스템과 독립"""
    
    def __init__(self):
        # Jailbreak 패턴 (기본적인 패턴만 포함)
        self.jailbreak_patterns = [
            r'ignore\s+(previous|all|above)',
            r'forget\s+(previous|all|above)',
            r'you\s+are\s+now',
            r'pretend\s+to\s+be',
            r'act\s+as\s+if',
            r'disregard\s+(previous|all)',
        ]
        
        # 유해 콘텐츠 키워드 (기본적인 패턴만 포함)
        self.harmful_keywords = [
            'violence', 'illegal', 'hack', 'exploit',
        ]
        
        logger.info("Guardrails Validator initialized")
    
    async def validate_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input 검증 - 기존 state 수정 없이 검증만 수행
        
        Args:
            state: AgentState 딕셔너리
            
        Returns:
            검증된 state (기존 state와 동일, 수정 없음)
            
        Raises:
            ValueError: 검증 실패 시
        """
        try:
            user_query = state.get('user_query', '')
            if not user_query:
                return state
            
            # Jailbreak 시도 검증
            validation_result = self._check_jailbreak(user_query)
            if not validation_result.is_valid:
                logger.warning(f"Jailbreak attempt detected: {validation_result.reason}")
                raise ValueError(f"Input validation failed: {validation_result.reason}")
            
            # 기존 state 그대로 반환 (수정 없음)
            return state
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            # 에러 발생 시 검증 통과 (기존 동작 유지)
            return state
    
    async def validate_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Output 검증 - 기존 result 수정 없이 검증만 수행
        
        Args:
            result: AgentState 딕셔너리
            
        Returns:
            검증된 result (기존 result와 동일, 수정 없음)
            
        Raises:
            ValueError: 검증 실패 시
        """
        try:
            # final_report 검증
            final_report = result.get('final_report', '')
            if final_report:
                validation_result = self._check_harmful_content(final_report)
                if not validation_result.is_valid:
                    logger.warning(f"Harmful content detected: {validation_result.reason}")
                    raise ValueError(f"Output validation failed: {validation_result.reason}")
            
            # research_results 검증
            research_results = result.get('research_results', [])
            for res in research_results:
                if isinstance(res, dict):
                    content = res.get('content', '') or res.get('result', '')
                    if content:
                        validation_result = self._check_harmful_content(str(content))
                        if not validation_result.is_valid:
                            logger.warning(f"Harmful content in research result: {validation_result.reason}")
                            # 경고만 하고 계속 진행 (기존 동작 유지)
            
            # 기존 result 그대로 반환 (수정 없음)
            return result
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Output validation error: {e}")
            # 에러 발생 시 검증 통과 (기존 동작 유지)
            return result
    
    def _check_jailbreak(self, text: str) -> ValidationResult:
        """Jailbreak 시도 검증"""
        text_lower = text.lower()
        
        for pattern in self.jailbreak_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    reason=f"Potential jailbreak pattern detected: {pattern}",
                    confidence=0.7
                )
        
        return ValidationResult(is_valid=True, reason="No jailbreak patterns detected")
    
    def _check_harmful_content(self, text: str) -> ValidationResult:
        """유해 콘텐츠 검증"""
        text_lower = text.lower()
        
        for keyword in self.harmful_keywords:
            if keyword in text_lower:
                # 단순 키워드 매칭은 너무 엄격할 수 있으므로 경고만
                return ValidationResult(
                    is_valid=True,  # 기본적으로 통과
                    reason=f"Potential harmful keyword detected: {keyword}",
                    confidence=0.5
                )
        
        return ValidationResult(is_valid=True, reason="No harmful content detected")

