"""
A2UI Generator - 연구 결과를 A2UI 형식으로 변환

연구 결과나 보고서를 A2UI (Agent-to-User Interface) JSON 형식으로 변환하여
풍부한 인터랙티브 UI로 표현합니다.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class A2UIGenerator:
    """
    A2UI JSON 생성기
    
    연구 결과를 A2UI 형식으로 변환합니다.
    """
    
    def __init__(self):
        """A2UIGenerator 초기화"""
        self.catalog_id = "sparkleforge.com:standard"
    
    def generate_research_report_a2ui(
        self,
        query: str,
        verified_results: List[Dict[str, Any]],
        report_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        연구 보고서를 A2UI 형식으로 생성
        
        Args:
            query: 연구 쿼리
            verified_results: 검증된 연구 결과 리스트
            report_text: 텍스트 보고서 (선택적)
            
        Returns:
            A2UI JSON 메시지
        """
        surface_id = f"research-report-{int(datetime.now().timestamp())}"
        
        # createSurface 메시지
        create_surface = {
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": self.catalog_id
            }
        }
        
        # updateComponents 메시지
        components = self._build_report_components(query, verified_results, report_text)
        update_components = {
            "updateComponents": {
                "surfaceId": surface_id,
                "components": components
            }
        }
        
        # updateDataModel 메시지
        data_model = self._build_data_model(query, verified_results, report_text)
        update_data_model = {
            "updateDataModel": {
                "surfaceId": surface_id,
                "path": "/",
                "op": "replace",
                "value": data_model
            }
        }
        
        # 첫 번째 메시지로 createSurface 반환 (클라이언트는 순차적으로 처리)
        # 실제로는 여러 메시지를 배열로 반환하거나, 단일 메시지로 통합
        return create_surface
    
    def _build_report_components(
        self,
        query: str,
        verified_results: List[Dict[str, Any]],
        report_text: Optional[str]
    ) -> List[Dict[str, Any]]:
        """보고서 컴포넌트 트리 구축"""
        components = []
        
        # Root Column
        root_column = {
            "id": "root",
            "component": "Column",
            "children": ["title", "summary", "results-list", "report-section"]
        }
        components.append(root_column)
        
        # Title
        title_text = {
            "id": "title",
            "component": "Text",
            "text": {"path": "query"},
            "usageHint": "h1"
        }
        components.append(title_text)
        
        # Summary Card
        summary_card = {
            "id": "summary",
            "component": "Card",
            "child": "summary-content"
        }
        components.append(summary_card)
        
        summary_content = {
            "id": "summary-content",
            "component": "Column",
            "children": ["summary-text", "results-count"]
        }
        components.append(summary_content)
        
        summary_text = {
            "id": "summary-text",
            "component": "Text",
            "text": {"path": "summary"},
            "usageHint": "body"
        }
        components.append(summary_text)
        
        results_count = {
            "id": "results-count",
            "component": "Text",
            "text": {"path": "results_count"},
            "usageHint": "caption"
        }
        components.append(results_count)
        
        # Results List
        results_list = {
            "id": "results-list",
            "component": "List",
            "direction": "vertical",
            "children": {
                "componentId": "result-item-template",
                "path": "results"
            }
        }
        components.append(results_list)
        
        # Result Item Template
        result_item_template = {
            "id": "result-item-template",
            "component": "Card",
            "child": "result-content"
        }
        components.append(result_item_template)
        
        result_content = {
            "id": "result-content",
            "component": "Column",
            "children": ["result-title", "result-url", "result-snippet"]
        }
        components.append(result_content)
        
        result_title = {
            "id": "result-title",
            "component": "Text",
            "text": {"path": "title"},
            "usageHint": "h3"
        }
        components.append(result_title)
        
        result_url = {
            "id": "result-url",
            "component": "Text",
            "text": {"path": "url"},
            "usageHint": "caption"
        }
        components.append(result_url)
        
        result_snippet = {
            "id": "result-snippet",
            "component": "Text",
            "text": {"path": "snippet"},
            "usageHint": "body"
        }
        components.append(result_snippet)
        
        # Report Section
        report_section = {
            "id": "report-section",
            "component": "Card",
            "child": "report-content"
        }
        components.append(report_section)
        
        report_content = {
            "id": "report-content",
            "component": "Column",
            "children": ["report-title", "report-text"]
        }
        components.append(report_content)
        
        report_title = {
            "id": "report-title",
            "component": "Text",
            "text": "연구 보고서",
            "usageHint": "h2"
        }
        components.append(report_title)
        
        report_text_component = {
            "id": "report-text",
            "component": "Text",
            "text": {"path": "report_text"},
            "usageHint": "body"
        }
        components.append(report_text_component)
        
        return components
    
    def _build_data_model(
        self,
        query: str,
        verified_results: List[Dict[str, Any]],
        report_text: Optional[str]
    ) -> Dict[str, Any]:
        """데이터 모델 구축"""
        # 결과 데이터 변환
        results = []
        for i, result in enumerate(verified_results):
            if isinstance(result, dict):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", "")[:500]  # 최대 500자
                })
        
        data_model = {
            "query": query,
            "summary": f"'{query}'에 대한 연구 결과입니다. 총 {len(results)}개의 검증된 결과를 수집했습니다.",
            "results_count": f"검증된 결과: {len(results)}개",
            "results": results,
            "report_text": report_text or "보고서가 아직 생성되지 않았습니다."
        }
        
        return data_model
    
    def generate_simple_card_a2ui(
        self,
        title: str,
        content: str,
        items: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        간단한 카드 형식 A2UI 생성
        
        Args:
            title: 카드 제목
            content: 카드 내용
            items: 항목 리스트 (선택적)
            
        Returns:
            A2UI JSON 메시지
        """
        surface_id = f"simple-card-{int(datetime.now().timestamp())}"
        
        components = []
        
        # Root Column
        root = {
            "id": "root",
            "component": "Column",
            "children": ["card"]
        }
        components.append(root)
        
        # Card
        card = {
            "id": "card",
            "component": "Card",
            "child": "card-content"
        }
        components.append(card)
        
        # Card Content
        card_children = ["title", "content"]
        if items:
            card_children.append("items-list")
        
        card_content = {
            "id": "card-content",
            "component": "Column",
            "children": card_children
        }
        components.append(card_content)
        
        # Title
        title_component = {
            "id": "title",
            "component": "Text",
            "text": {"path": "title"},
            "usageHint": "h2"
        }
        components.append(title_component)
        
        # Content
        content_component = {
            "id": "content",
            "component": "Text",
            "text": {"path": "content"},
            "usageHint": "body"
        }
        components.append(content_component)
        
        # Items List (if provided)
        if items:
            items_list = {
                "id": "items-list",
                "component": "List",
                "direction": "vertical",
                "children": {
                    "componentId": "item-template",
                    "path": "items"
                }
            }
            components.append(items_list)
            
            item_template = {
                "id": "item-template",
                "component": "Text",
                "text": {"path": "text"},
                "usageHint": "body"
            }
            components.append(item_template)
        
        # Data Model
        data_model = {
            "title": title,
            "content": content,
            "items": items or []
        }
        
        return {
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": self.catalog_id
            },
            "updateComponents": {
                "surfaceId": surface_id,
                "components": components
            },
            "updateDataModel": {
                "surfaceId": surface_id,
                "path": "/",
                "op": "replace",
                "value": data_model
            }
        }
    
    def generate_question_a2ui(
        self,
        question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        질문을 A2UI 형식으로 생성
        
        Args:
            question: 질문 정보
                {
                    "id": "question_id",
                    "text": "질문 텍스트",
                    "format": "choice" or "natural_language",
                    "options": [...],  # 선택지 형식인 경우
                    "type": "...",
                    "field": "..."
                }
            
        Returns:
            A2UI JSON 메시지
        """
        surface_id = f"question-{question['id']}"
        
        if question['format'] == 'choice':
            # ChoicePicker 사용
            components = [
                {
                    "id": "root",
                    "component": "Column",
                    "children": ["question-text", "choice-picker", "submit-button"]
                },
                {
                    "id": "question-text",
                    "component": "Text",
                    "text": question['text'],
                    "usageHint": "h3"
                },
                {
                    "id": "choice-picker",
                    "component": "ChoicePicker",
                    "label": "선택하세요",
                    "usageHint": "mutuallyExclusive",
                    "options": [
                        {"label": opt['label'], "value": opt['value']}
                        for opt in question.get('options', [])
                    ],
                    "value": {"path": "selected_value"}
                },
                {
                    "id": "submit-button",
                    "component": "Button",
                    "child": "submit-text",
                    "primary": True,
                    "action": {
                        "name": "submit_answer",
                        "context": {
                            "question_id": question['id'],
                            "response_path": {"path": "selected_value"}
                        }
                    }
                },
                {
                    "id": "submit-text",
                    "component": "Text",
                    "text": "제출"
                }
            ]
            
            data_model = {
                "selected_value": []
            }
        else:
            # TextField 사용
            components = [
                {
                    "id": "root",
                    "component": "Column",
                    "children": ["question-text", "text-field", "submit-button"]
                },
                {
                    "id": "question-text",
                    "component": "Text",
                    "text": question['text'],
                    "usageHint": "h3"
                },
                {
                    "id": "text-field",
                    "component": "TextField",
                    "label": "답변",
                    "text": {"path": "user_response"},
                    "usageHint": "longText"
                },
                {
                    "id": "submit-button",
                    "component": "Button",
                    "child": "submit-text",
                    "primary": True,
                    "action": {
                        "name": "submit_answer",
                        "context": {
                            "question_id": question['id'],
                            "response_path": {"path": "user_response"}
                        }
                    }
                },
                {
                    "id": "submit-text",
                    "component": "Text",
                    "text": "제출"
                }
            ]
            
            data_model = {
                "user_response": ""
            }
        
        return {
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": self.catalog_id
            },
            "updateComponents": {
                "surfaceId": surface_id,
                "components": components
            },
            "updateDataModel": {
                "surfaceId": surface_id,
                "path": "/",
                "op": "replace",
                "value": data_model
            }
        }


def get_a2ui_generator() -> A2UIGenerator:
    """A2UIGenerator 싱글톤 인스턴스 반환"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = A2UIGenerator()
    return _generator_instance

