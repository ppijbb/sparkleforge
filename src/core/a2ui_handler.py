"""
A2UI Handler - Agent 출력에서 A2UI JSON 감지 및 처리

Agent가 생성한 A2UI (Agent-to-User Interface) JSON을 감지하고 파싱하여
Streamlit에서 렌더링할 수 있도록 준비합니다.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class A2UIHandler:
    """
    A2UI 출력 감지 및 처리 핸들러
    
    Agent 응답에서 A2UI JSON을 감지하고 검증합니다.
    """
    
    def __init__(self):
        """A2UIHandler 초기화"""
        self.detected_surfaces: Dict[str, Dict[str, Any]] = {}
    
    def detect_a2ui(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Agent 응답에서 A2UI JSON 감지
        
        Args:
            response: Agent 응답 텍스트
            
        Returns:
            A2UI JSON 객체 또는 None
        """
        try:
            # A2UI 메시지 패턴 감지
            patterns = [
                r'\{[^{}]*"createSurface"[^{}]*\}',  # createSurface 메시지
                r'\{[^{}]*"updateComponents"[^{}]*\}',  # updateComponents 메시지
                r'\{[^{}]*"updateDataModel"[^{}]*\}',  # updateDataModel 메시지
                r'\{[^{}]*"deleteSurface"[^{}]*\}',  # deleteSurface 메시지
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    try:
                        a2ui_json = json.loads(match.group())
                        if self._validate_a2ui(a2ui_json):
                            logger.info(f"Detected A2UI message: {list(a2ui_json.keys())[0]}")
                            return a2ui_json
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse A2UI JSON: {e}")
                        continue
            
            # JSON 코드 블록에서 감지 시도
            code_block_pattern = r'```(?:json)?\s*(\{.*?"createSurface".*?\})\s*```'
            match = re.search(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    a2ui_json = json.loads(match.group(1))
                    if self._validate_a2ui(a2ui_json):
                        logger.info("Detected A2UI in code block")
                        return a2ui_json
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse A2UI from code block: {e}")
            
            # 배열 형식 감지 (여러 메시지)
            array_pattern = r'\[\s*(\{.*?"createSurface".*?\})\s*\]'
            match = re.search(array_pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    a2ui_array = json.loads(match.group(0))
                    if isinstance(a2ui_array, list) and len(a2ui_array) > 0:
                        # 첫 번째 메시지 반환
                        if self._validate_a2ui(a2ui_array[0]):
                            logger.info("Detected A2UI array, using first message")
                            return a2ui_array[0]
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse A2UI array: {e}")
            
        except Exception as e:
            logger.debug(f"Error detecting A2UI: {e}")
        
        return None
    
    def _validate_a2ui(self, a2ui_json: Dict[str, Any]) -> bool:
        """
        A2UI JSON 검증
        
        Args:
            a2ui_json: A2UI JSON 객체
            
        Returns:
            유효성 여부
        """
        if not isinstance(a2ui_json, dict):
            return False
        
        # A2UI 메시지 타입 확인
        valid_keys = ['createSurface', 'updateComponents', 'updateDataModel', 'deleteSurface']
        has_valid_key = any(key in a2ui_json for key in valid_keys)
        
        if not has_valid_key:
            return False
        
        # createSurface 검증
        if 'createSurface' in a2ui_json:
            cs = a2ui_json['createSurface']
            if not isinstance(cs, dict):
                return False
            if 'surfaceId' not in cs or 'catalogId' not in cs:
                return False
        
        # updateComponents 검증
        if 'updateComponents' in a2ui_json:
            uc = a2ui_json['updateComponents']
            if not isinstance(uc, dict):
                return False
            if 'surfaceId' not in uc or 'components' not in uc:
                return False
            if not isinstance(uc['components'], list):
                return False
            # root 컴포넌트 확인
            has_root = any(comp.get('id') == 'root' for comp in uc['components'])
            if not has_root:
                logger.warning("updateComponents missing root component")
                return False
        
        # updateDataModel 검증
        if 'updateDataModel' in a2ui_json:
            udm = a2ui_json['updateDataModel']
            if not isinstance(udm, dict):
                return False
            if 'surfaceId' not in udm:
                return False
        
        # deleteSurface 검증
        if 'deleteSurface' in a2ui_json:
            ds = a2ui_json['deleteSurface']
            if not isinstance(ds, dict):
                return False
            if 'surfaceId' not in ds:
                return False
        
        return True
    
    def extract_a2ui_messages(self, response: str) -> List[Dict[str, Any]]:
        """
        Agent 응답에서 모든 A2UI 메시지 추출
        
        Args:
            response: Agent 응답 텍스트
            
        Returns:
            A2UI 메시지 리스트
        """
        messages = []
        
        # 단일 메시지 감지
        single_message = self.detect_a2ui(response)
        if single_message:
            messages.append(single_message)
            return messages
        
        # 배열 형식에서 여러 메시지 추출
        try:
            # JSON 배열 패턴
            array_pattern = r'\[\s*(\{.*?\})\s*(?:,\s*(\{.*?\}))*\s*\]'
            matches = re.finditer(array_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    array_json = json.loads(match.group(0))
                    if isinstance(array_json, list):
                        for msg in array_json:
                            if self._validate_a2ui(msg):
                                messages.append(msg)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.debug(f"Error extracting A2UI messages: {e}")
        
        return messages
    
    def normalize_a2ui(self, a2ui_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        A2UI JSON 정규화 (필요한 경우)
        
        Args:
            a2ui_json: A2UI JSON 객체
            
        Returns:
            정규화된 A2UI JSON
        """
        # 현재는 그대로 반환, 나중에 정규화 로직 추가 가능
        return a2ui_json


# 싱글톤 인스턴스
_handler_instance: Optional[A2UIHandler] = None


def get_a2ui_handler() -> A2UIHandler:
    """A2UIHandler 싱글톤 인스턴스 반환"""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = A2UIHandler()
    return _handler_instance

