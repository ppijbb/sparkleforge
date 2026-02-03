"""
Prompt Refiner Wrapper for SparkleForge

prompt-refiner를 사용하여 모든 LLM 호출과 agent 요청 시 prompt를 최적화합니다.
기존 코드 변경 최소화를 위해 wrapper 패턴을 사용합니다.
"""

import asyncio
import functools
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal, Callable, TypeVar, ParamSpec
from dataclasses import dataclass, field
from enum import Enum

# prompt-refiner import 경로 설정
project_root = Path(__file__).parent.parent.parent.parent
prompt_refiner_path = project_root / "open_researcher" / "prompt-refiner" / "src"
if str(prompt_refiner_path) not in sys.path:
    sys.path.insert(0, str(prompt_refiner_path))

try:
    from prompt_refiner import (
        Pipeline,
        StripHTML,
        NormalizeWhitespace,
        Deduplicate,
        TruncateTokens,
    )
    # CountTokens는 analyzer 모듈에 있음
    try:
        from prompt_refiner.analyzer import TokenTracker
        CountTokens = TokenTracker  # CountTokens는 TokenTracker의 별칭
    except ImportError:
        # TokenTracker도 없으면 간단한 클래스로 대체
        class CountTokens:
            def __init__(self, *args, **kwargs):
                pass
    PROMPT_REFINER_AVAILABLE = True
except ImportError:
    # 로그 출력 없이 조용히 처리
    PROMPT_REFINER_AVAILABLE = False
    # Fallback classes
    class Pipeline:
        def __init__(self, *args, **kwargs):
            pass
        def pipe(self, *args, **kwargs):
            return self
        def run(self, text: str) -> str:
            return text
    class StripHTML:
        pass
    class NormalizeWhitespace:
        pass
    class Deduplicate:
        pass
    class TruncateTokens:
        def __init__(self, *args, **kwargs):
            pass
    class CountTokens:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)


class RefinerStrategy(str, Enum):
    """Refiner 전략 타입."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class RefinerStats:
    """Refiner 통계 데이터."""
    original_tokens: int = 0
    refined_tokens: int = 0
    reduction_percentage: float = 0.0
    total_refinements: int = 0
    total_token_savings: int = 0
    
    def update(self, original: int, refined: int):
        """통계 업데이트."""
        self.original_tokens += original
        self.refined_tokens += refined
        self.total_refinements += 1
        savings = original - refined
        self.total_token_savings += savings
        if original > 0:
            self.reduction_percentage = (self.total_token_savings / self.original_tokens) * 100


class PromptRefinerWrapper:
    """
    Prompt Refiner Wrapper 클래스.
    
    모든 LLM 호출과 agent 요청 시 prompt를 최적화합니다.
    """
    
    def __init__(
        self,
        strategy: Literal["minimal", "standard", "aggressive", "custom"] = "aggressive",
        enabled: bool = True,
        max_tokens: Optional[int] = None,
        collect_stats: bool = True,
    ):
        """
        PromptRefinerWrapper 초기화.
        
        Args:
            strategy: 최적화 전략 (minimal/standard/aggressive/custom)
            enabled: Refiner 활성화 여부
            max_tokens: TruncateTokens의 최대 토큰 수 (None이면 자동 계산)
            collect_stats: 통계 수집 여부
        """
        self.strategy = strategy
        self.enabled = enabled and PROMPT_REFINER_AVAILABLE
        self.max_tokens = max_tokens
        self.collect_stats = collect_stats
        self.stats = RefinerStats()
        
        if self.enabled:
            self.refiner = self._build_pipeline(strategy)
            # 로그 출력 없음 (조용히 초기화)
        else:
            self.refiner = None
            # 로그 출력 없음 (조용히 비활성화)
    
    def _build_pipeline(self, strategy: str) -> Pipeline:
        """
        전략에 따라 refiner pipeline 구성.
        
        Args:
            strategy: 최적화 전략
            
        Returns:
            구성된 Pipeline 인스턴스
        """
        if not PROMPT_REFINER_AVAILABLE:
            return Pipeline()
        
        refiner = Pipeline()
        
        if strategy == "minimal":
            # Minimal: HTML 제거 + 공백 정규화 (약 4-5% 절감)
            refiner = refiner.pipe(StripHTML(to_markdown=True))
            refiner = refiner.pipe(NormalizeWhitespace())
            
        elif strategy == "standard":
            # Standard: Minimal + 중복 제거 (약 5-8% 절감)
            refiner = refiner.pipe(StripHTML(to_markdown=True))
            refiner = refiner.pipe(NormalizeWhitespace())
            refiner = refiner.pipe(Deduplicate(similarity_threshold=0.85, method="jaccard", granularity="paragraph"))
            
        elif strategy == "aggressive":
            # Aggressive: Standard + 토큰 자르기 (약 15% 절감)
            refiner = refiner.pipe(StripHTML(to_markdown=True))
            refiner = refiner.pipe(NormalizeWhitespace())
            refiner = refiner.pipe(Deduplicate(similarity_threshold=0.85, method="jaccard", granularity="paragraph"))
            # TruncateTokens는 max_tokens가 설정된 경우에만 추가
            if self.max_tokens:
                refiner = refiner.pipe(TruncateTokens(
                    max_tokens=self.max_tokens,
                    strategy="head",
                    respect_sentence_boundary=True
                ))
            else:
                # max_tokens가 없으면 매우 큰 값으로 설정 (실제로는 자르지 않음)
                # 대신 모델별 기본값 사용 (나중에 동적으로 설정 가능)
                logger.debug("TruncateTokens not added (max_tokens not specified)")
        
        elif strategy == "custom":
            # Custom: 사용자 정의 (현재는 aggressive와 동일)
            refiner = refiner.pipe(StripHTML(to_markdown=True))
            refiner = refiner.pipe(NormalizeWhitespace())
            refiner = refiner.pipe(Deduplicate(similarity_threshold=0.85, method="jaccard", granularity="paragraph"))
            if self.max_tokens:
                refiner = refiner.pipe(TruncateTokens(
                    max_tokens=self.max_tokens,
                    strategy="head",
                    respect_sentence_boundary=True
                ))
        
        return refiner
    
    def _estimate_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수 추정 (간단한 방법).
        
        Args:
            text: 입력 텍스트
            
        Returns:
            추정된 토큰 수
        """
        # 간단한 추정: 1 token ≈ 4 characters (영어 기준)
        # 한국어는 더 적은 토큰 사용 가능
        return len(text) // 4
    
    def refine_prompt(self, prompt: str, model_max_tokens: Optional[int] = None) -> str:
        """
        단일 prompt 최적화.
        
        Args:
            prompt: 원본 prompt
            model_max_tokens: 모델의 최대 토큰 수 (TruncateTokens에 사용)
            
        Returns:
            최적화된 prompt
        """
        if not self.enabled or not prompt:
            return prompt
        
        # 원본 토큰 수 추정
        original_tokens = self._estimate_tokens(prompt)
        
        try:
            # TruncateTokens의 max_tokens를 동적으로 설정
            if model_max_tokens and self.max_tokens is None and self.strategy in ["aggressive", "custom"]:
                # 모델의 max_tokens를 기반으로 설정 (80% 사용)
                effective_max = int(model_max_tokens * 0.8)
                # 동적으로 TruncateTokens를 추가한 새로운 refiner 생성
                dynamic_refiner = Pipeline()
                # 기존 pipeline의 모든 operation 복사
                for operation in self.refiner._refiners:
                    dynamic_refiner = dynamic_refiner.pipe(operation)
                # TruncateTokens 추가 (동적 설정)
                dynamic_refiner = dynamic_refiner.pipe(TruncateTokens(
                    max_tokens=effective_max,
                    strategy="head",
                    respect_sentence_boundary=True
                ))
                refined = dynamic_refiner.run(prompt)
            else:
                refined = self.refiner.run(prompt)
            
            # 최적화 후 토큰 수 추정
            refined_tokens = self._estimate_tokens(refined)
            
            # 통계 업데이트
            if self.collect_stats:
                self.stats.update(original_tokens, refined_tokens)
                if original_tokens > 0:
                    reduction = ((original_tokens - refined_tokens) / original_tokens) * 100
                    if reduction > 0:
                        logger.debug(
                            f"Prompt refined: {original_tokens} -> {refined_tokens} tokens "
                            f"({reduction:.1f}% reduction)"
                        )
            
            return refined
            
        except Exception as e:
            logger.warning(f"Prompt refinement failed: {e}. Returning original prompt.")
            return prompt
    
    def refine_messages(self, messages: List[Dict[str, str]], model_max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Messages 리스트의 content 최적화.
        
        Args:
            messages: 원본 messages 리스트 (각 dict는 'role'과 'content' 키를 가짐)
            model_max_tokens: 모델의 최대 토큰 수
            
        Returns:
            최적화된 messages 리스트
        """
        if not self.enabled or not messages:
            return messages
        
        refined_messages = []
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                refined_content = self.refine_prompt(message["content"], model_max_tokens)
                refined_message = {**message, "content": refined_content}
                refined_messages.append(refined_message)
            else:
                # content가 없거나 형식이 맞지 않으면 그대로 유지
                refined_messages.append(message)
        
        return refined_messages
    
    def refine_system_and_prompt(
        self,
        system_message: Optional[str] = None,
        prompt: Optional[str] = None,
        model_max_tokens: Optional[int] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """
        System message와 prompt를 모두 최적화.
        
        Args:
            system_message: 원본 system message
            prompt: 원본 prompt
            model_max_tokens: 모델의 최대 토큰 수
            
        Returns:
            (최적화된 system_message, 최적화된 prompt) 튜플
        """
        refined_system = self.refine_prompt(system_message, model_max_tokens) if system_message else None
        refined_prompt = self.refine_prompt(prompt, model_max_tokens) if prompt else None
        return refined_system, refined_prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """
        통계 정보 반환.
        
        Returns:
            통계 딕셔너리
        """
        if not self.collect_stats:
            return {}
        
        return {
            "original_tokens": self.stats.original_tokens,
            "refined_tokens": self.stats.refined_tokens,
            "total_token_savings": self.stats.total_token_savings,
            "reduction_percentage": self.stats.reduction_percentage,
            "total_refinements": self.stats.total_refinements,
            "average_reduction": (
                (self.stats.original_tokens - self.stats.refined_tokens) / self.stats.total_refinements
                if self.stats.total_refinements > 0 else 0
            ),
        }
    
    def reset_stats(self):
        """통계 초기화."""
        self.stats = RefinerStats()


# Global refiner instance (lazy initialization)
_refiner_wrapper: Optional[PromptRefinerWrapper] = None


def get_refiner_wrapper() -> PromptRefinerWrapper:
    """
    Global refiner wrapper 인스턴스 반환 (lazy initialization).
    
    Returns:
        PromptRefinerWrapper 인스턴스
    """
    global _refiner_wrapper
    if _refiner_wrapper is None:
        # 설정 파일에서 우선 읽기, 없으면 환경변수에서 읽기
        try:
            from src.core.researcher_config import get_prompt_refiner_config
            refiner_config = get_prompt_refiner_config()
            strategy = refiner_config.strategy
            enabled = refiner_config.enabled
            max_tokens = refiner_config.max_tokens
            collect_stats = refiner_config.collect_stats
        except (RuntimeError, ImportError):
            # 설정이 로드되지 않았거나 import 실패 시 환경변수 사용
            strategy = os.getenv("PROMPT_REFINER_STRATEGY", "aggressive")
            enabled = os.getenv("PROMPT_REFINER_ENABLED", "true").lower() == "true"
            max_tokens_str = os.getenv("PROMPT_REFINER_MAX_TOKENS")
            max_tokens = int(max_tokens_str) if max_tokens_str else None
            collect_stats = os.getenv("PROMPT_REFINER_COLLECT_STATS", "true").lower() == "true"
        
        _refiner_wrapper = PromptRefinerWrapper(
            strategy=strategy,
            enabled=enabled,
            max_tokens=max_tokens,
            collect_stats=collect_stats,
        )
    return _refiner_wrapper


def refine_prompt(prompt: str, model_max_tokens: Optional[int] = None) -> str:
    """
    편의 함수: prompt 최적화.
    
    Args:
        prompt: 원본 prompt
        model_max_tokens: 모델의 최대 토큰 수
        
    Returns:
        최적화된 prompt
    """
    refiner = get_refiner_wrapper()
    return refiner.refine_prompt(prompt, model_max_tokens)


def refine_messages(messages: List[Dict[str, str]], model_max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
    """
    편의 함수: messages 최적화.
    
    Args:
        messages: 원본 messages 리스트
        model_max_tokens: 모델의 최대 토큰 수
        
    Returns:
        최적화된 messages 리스트
    """
    refiner = get_refiner_wrapper()
    return refiner.refine_messages(messages, model_max_tokens)


# ============================================================================
# Decorator 패턴 구현
# ============================================================================

# Type variables for decorator
P = ParamSpec('P')
R = TypeVar('R')


def refine_llm_call(func: Callable[P, R]) -> Callable[P, R]:
    """
    LLM 호출 함수를 자동으로 최적화하는 decorator.
    
    표준 LLM 호출 시그니처를 가정:
    - prompt: str
    - system_message: Optional[str] = None
    - model_name: Optional[str] = None
    - **kwargs
    
    model_name이 있으면 자동으로 max_tokens를 감지하여 적용합니다.
    
    Example:
        @refine_llm_call
        async def execute_with_model(self, prompt: str, task_type: TaskType, 
                                     model_name: str = None, system_message: str = None, **kwargs):
            ...
    """
    @functools.wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        refiner = get_refiner_wrapper()
        
        # model_max_tokens 자동 감지
        model_max_tokens = None
        if "model_name" in kwargs and kwargs["model_name"]:
            try:
                from src.core.llm_manager import get_llm_orchestrator
                orchestrator = get_llm_orchestrator()
                model_name = kwargs["model_name"]
                model_name_clean = model_name.replace("_langchain", "")
                if model_name_clean in orchestrator.models:
                    model_max_tokens = orchestrator.models[model_name_clean].max_tokens
            except Exception:
                pass
        
        # prompt 최적화
        if "prompt" in kwargs and kwargs["prompt"]:
            kwargs["prompt"] = refiner.refine_prompt(kwargs["prompt"], model_max_tokens)
        
        # system_message 최적화
        if "system_message" in kwargs and kwargs["system_message"]:
            kwargs["system_message"] = refiner.refine_prompt(kwargs["system_message"], model_max_tokens)
        
        return await func(*args, **kwargs)
    
    @functools.wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        refiner = get_refiner_wrapper()
        
        model_max_tokens = None
        if "model_name" in kwargs and kwargs["model_name"]:
            try:
                from src.core.llm_manager import get_llm_orchestrator
                orchestrator = get_llm_orchestrator()
                model_name = kwargs["model_name"]
                model_name_clean = model_name.replace("_langchain", "")
                if model_name_clean in orchestrator.models:
                    model_max_tokens = orchestrator.models[model_name_clean].max_tokens
            except Exception:
                pass
        
        if "prompt" in kwargs and kwargs["prompt"]:
            kwargs["prompt"] = refiner.refine_prompt(kwargs["prompt"], model_max_tokens)
        
        if "system_message" in kwargs and kwargs["system_message"]:
            kwargs["system_message"] = refiner.refine_prompt(kwargs["system_message"], model_max_tokens)
        
        return func(*args, **kwargs)
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def refine_prompt_args(
    prompt_param: str = "prompt",
    system_message_param: Optional[str] = "system_message",
    model_max_tokens_param: Optional[str] = None,
    auto_detect_max_tokens: bool = True
):
    """
    함수의 prompt 인자를 자동으로 최적화하는 decorator.
    
    Args:
        prompt_param: prompt 인자 이름 (기본값: "prompt")
        system_message_param: system_message 인자 이름 (기본값: "system_message", None이면 무시)
        model_max_tokens_param: model_max_tokens를 가져올 인자 이름 (예: "model_name")
        auto_detect_max_tokens: model_name에서 자동으로 max_tokens 감지 (기본값: True)
    
    Example:
        @refine_prompt_args()
        async def domain_exploration(self, query: str):
            domain_prompt = get_prompt(...)
            # domain_prompt는 자동으로 최적화됨
        
        @refine_prompt_args(prompt_param="domain_prompt", system_message_param=None)
        async def domain_exploration(self, query: str, domain_prompt: str):
            ...
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            refiner = get_refiner_wrapper()
            
            # model_max_tokens 결정
            model_max_tokens = None
            if model_max_tokens_param and model_max_tokens_param in kwargs:
                if auto_detect_max_tokens:
                    try:
                        from src.core.llm_manager import get_llm_orchestrator
                        orchestrator = get_llm_orchestrator()
                        model_name = kwargs[model_max_tokens_param]
                        model_name_clean = model_name.replace("_langchain", "") if model_name else None
                        if model_name_clean and model_name_clean in orchestrator.models:
                            model_max_tokens = orchestrator.models[model_name_clean].max_tokens
                    except Exception:
                        pass
                else:
                    model_max_tokens = kwargs.get(model_max_tokens_param)
            elif "model_max_tokens" in kwargs:
                model_max_tokens = kwargs["model_max_tokens"]
            
            # prompt 최적화
            if prompt_param in kwargs and kwargs[prompt_param]:
                kwargs[prompt_param] = refiner.refine_prompt(kwargs[prompt_param], model_max_tokens)
            
            # system_message 최적화
            if system_message_param and system_message_param in kwargs and kwargs[system_message_param]:
                kwargs[system_message_param] = refiner.refine_prompt(kwargs[system_message_param], model_max_tokens)
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            refiner = get_refiner_wrapper()
            
            model_max_tokens = None
            if model_max_tokens_param and model_max_tokens_param in kwargs:
                if auto_detect_max_tokens:
                    try:
                        from src.core.llm_manager import get_llm_orchestrator
                        orchestrator = get_llm_orchestrator()
                        model_name = kwargs[model_max_tokens_param]
                        model_name_clean = model_name.replace("_langchain", "") if model_name else None
                        if model_name_clean and model_name_clean in orchestrator.models:
                            model_max_tokens = orchestrator.models[model_name_clean].max_tokens
                    except Exception:
                        pass
                else:
                    model_max_tokens = kwargs.get(model_max_tokens_param)
            elif "model_max_tokens" in kwargs:
                model_max_tokens = kwargs["model_max_tokens"]
            
            if prompt_param in kwargs and kwargs[prompt_param]:
                kwargs[prompt_param] = refiner.refine_prompt(kwargs[prompt_param], model_max_tokens)
            
            if system_message_param and system_message_param in kwargs and kwargs[system_message_param]:
                kwargs[system_message_param] = refiner.refine_prompt(kwargs[system_message_param], model_max_tokens)
            
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
