"""
프롬프트 관리자

하드코딩된 프롬프트들을 구조화된 방식으로 관리하고,
템플릿 변수 치환을 지원하는 중앙 집중 관리 시스템입니다.
"""

import os
from string import Template
from typing import Dict, Any, Optional
from pathlib import Path


class PromptManager:
    """프롬프트 템플릿 관리자"""

    def __init__(self):
        self._prompts: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._loaded_modules = set()

    def _load_module_if_needed(self, module_name: str):
        """필요시 모듈 로드"""
        if module_name in self._loaded_modules:
            return

        try:
            if module_name == 'orchestrator':
                from . import orchestrator as mod
            elif module_name == 'agents':
                from . import agents as mod
            elif module_name == 'core':
                from . import core as mod
            elif module_name == 'shared':
                from . import shared as mod
            else:
                raise ValueError(f"Unknown module: {module_name}")

            # 모듈의 모든 프롬프트 로드
            for attr_name in dir(mod):
                if not attr_name.startswith('_'):
                    attr_value = getattr(mod, attr_name)
                    if isinstance(attr_value, dict) and 'template' in attr_value:
                        category = module_name
                        prompt_name = attr_name
                        if category not in self._prompts:
                            self._prompts[category] = {}
                        self._prompts[category][prompt_name] = attr_value

            self._loaded_modules.add(module_name)

        except ImportError as e:
            raise ImportError(f"Failed to load prompt module '{module_name}': {e}")

    def get_prompt(self, category: str, prompt_name: str, **kwargs) -> str:
        """프롬프트 템플릿 렌더링"""
        self._load_module_if_needed(category)

        if category not in self._prompts or prompt_name not in self._prompts[category]:
            raise ValueError(f"Prompt '{prompt_name}' not found in category '{category}'")

        prompt_config = self._prompts[category][prompt_name]
        template_str = prompt_config['template']

        # 템플릿 변수 치환
        template = Template(template_str)
        try:
            return template.safe_substitute(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {e}")

    def get_system_message(self, category: str, prompt_name: str) -> str:
        """시스템 메시지 반환"""
        self._load_module_if_needed(category)

        if category not in self._prompts or prompt_name not in self._prompts[category]:
            raise ValueError(f"Prompt '{prompt_name}' not found in category '{category}'")

        return self._prompts[category][prompt_name].get('system_message', '')

    def get_prompt_config(self, category: str, prompt_name: str) -> Dict[str, Any]:
        """프롬프트 설정 전체 반환"""
        self._load_module_if_needed(category)

        if category not in self._prompts or prompt_name not in self._prompts[category]:
            raise ValueError(f"Prompt '{prompt_name}' not found in category '{category}'")

        return self._prompts[category][prompt_name]

    def list_available_prompts(self, category: Optional[str] = None) -> Dict[str, list]:
        """사용 가능한 프롬프트 목록 반환"""
        if category:
            self._load_module_if_needed(category)
            return {category: list(self._prompts.get(category, {}).keys())}
        else:
            # 모든 카테고리 로드
            for cat in ['orchestrator', 'agents', 'core', 'shared']:
                self._load_module_if_needed(cat)

            return {cat: list(prompts.keys()) for cat, prompts in self._prompts.items()}


# 전역 인스턴스
_manager = None

def get_prompt_manager() -> PromptManager:
    """전역 프롬프트 매니저 인스턴스 반환"""
    global _manager
    if _manager is None:
        _manager = PromptManager()
    return _manager

def get_prompt(category: str, prompt_name: str, **kwargs) -> str:
    """편의 함수: 프롬프트 템플릿 렌더링"""
    return get_prompt_manager().get_prompt(category, prompt_name, **kwargs)

def get_system_message(category: str, prompt_name: str) -> str:
    """편의 함수: 시스템 메시지 반환"""
    return get_prompt_manager().get_system_message(category, prompt_name)

