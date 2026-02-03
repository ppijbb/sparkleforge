"""
Core 모듈 프롬프트 모듈

Core 모듈에서 사용되는 프롬프트들을 포함합니다.
"""

from .autonomous_orchestrator import autonomous_orchestrator_prompts
from .result_sharing import result_sharing_prompts
from .context_engineering import context_engineering_prompts

__all__ = [
    'autonomous_orchestrator_prompts',
    'result_sharing_prompts',
    'context_engineering_prompts'
]

