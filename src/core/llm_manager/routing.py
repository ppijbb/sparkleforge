"""Provider rotation, rate-limit tracking, and model selection.

Split out of the former monolithic llm_manager.py (issue #582).
"""

import logging
import os
import time
from typing import List

from src.core.llm_manager.types import ModelConfig, TaskType

logger = logging.getLogger(__name__)


class RoutingMixin:
    """Provider rotation order, rate-limit bookkeeping, and select_model."""

    # Provider별 rate limit 쿨다운 (초). NIM 429는 순간 동시성 제한이라 짧게 잡는다.
    PROVIDER_RATE_LIMIT_COOLDOWN = {"nvidia": 30}
    DEFAULT_RATE_LIMIT_COOLDOWN = 300

    @staticmethod
    def _build_provider_rotation_order() -> List[str]:
        """API 키가 있는 provider로 로테이션 순서 구성 (설정된 LLM_PROVIDER 우선)."""
        available = []
        if os.getenv("NVIDIA_API_KEY"):
            available.append("nvidia")
        if os.getenv("OPENROUTER_API_KEY"):
            available.append("openrouter")
        if os.getenv("GROQ_API_KEY"):
            available.append("groq")
        if os.getenv("CEREBRAS_API_KEY"):
            available.append("cerebras")
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            available.append("google")
        if os.getenv("OPENAI_API_KEY"):
            available.append("openai")
        preferred = (os.getenv("LLM_PROVIDER") or "").strip().lower()
        if preferred in available:
            available.remove(preferred)
            available.insert(0, preferred)
        return available or ["google"]

    def _resolve_configured_model_key(self, configured: str | None) -> str | None:
        """Resolve a configured model name or 'provider/model_id' string to a registered key."""
        if not configured:
            return None
        if configured in self.models:
            return configured

        provider_hint = ""
        bare_id = configured
        if "/" in configured:
            provider_hint, bare_id = configured.split("/", 1)
            provider_hint = provider_hint.strip().lower()

        for key, cfg in self.models.items():
            if cfg.model_id == configured:
                return key
            if cfg.model_id == bare_id and (not provider_hint or cfg.provider == provider_hint):
                return key

        # If an explicit Google/Gemini model ID was requested (e.g. google/gemini-3.1-flash-lite-preview)
        # and Google credentials are loaded, register an on-demand ModelConfig so CLI --model honors it.
        has_google = any(cfg.provider == "google" for cfg in self.models.values())
        if has_google and (provider_hint == "google" or bare_id.startswith("gemini-")):
            self.models[configured] = ModelConfig(
                name=configured,
                provider="google",
                model_id=bare_id,
                temperature=0.2,
                max_tokens=4000,
                cost_per_token=0.0001,
                speed_rating=9.0,
                quality_rating=8.0,
                capabilities=list(TaskType),
            )
            return configured

        return None


    def _is_provider_rate_limited(self, provider: str) -> bool:
        """Provider가 rate limit에 걸렸는지 확인 (쿨다운 후 자동 해제)."""
        if provider not in self.provider_rate_limited:
            return False

        cooldown = self.PROVIDER_RATE_LIMIT_COOLDOWN.get(
            provider, self.DEFAULT_RATE_LIMIT_COOLDOWN
        )
        rate_limit_time = self.provider_rate_limited[provider]
        if time.time() - rate_limit_time > cooldown:
            del self.provider_rate_limited[provider]
            logger.info(f"Provider {provider} rate limit automatically cleared after {cooldown}s")
            return False

        return True


    def _mark_provider_rate_limited(self, provider: str):
        """Provider를 rate limit 상태로 표시."""
        self.provider_rate_limited[provider] = time.time()
        logger.warning(f"Provider {provider} marked as rate-limited (will retry after 5 minutes)")


    def _get_available_providers(self) -> List[str]:
        """Rate limit되지 않은 사용 가능한 Provider 목록 반환."""
        available = []
        for provider in self.provider_rotation_order:
            # Rate limit 확인
            if not self._is_provider_rate_limited(provider):
                # Provider에 사용 가능한 모델이 있는지 확인
                has_models = any(
                    config.provider == provider for config in self.models.values()
                )
                if has_models:
                    available.append(provider)

        return available


    def select_model(
        self, task_type: TaskType, complexity: float = 5.0, budget: float = None
    ) -> str:
        """작업에 최적 모델 선택 - Provider 로테이션: OpenRouter -> Groq -> Cerebras."""
        if budget is None:
            budget = self.llm_config.budget_limit

        # .env에 지정된 작업별 모델을 최우선 사용 (provider 사용 가능 시)
        configured = {
            TaskType.PLANNING: self.llm_config.planning_model,
            TaskType.DEEP_REASONING: self.llm_config.reasoning_model,
            TaskType.VERIFICATION: self.llm_config.verification_model,
            TaskType.GENERATION: self.llm_config.generation_model,
            TaskType.COMPRESSION: self.llm_config.compression_model,
        }.get(task_type) or self.llm_config.primary_model
        resolved_key = self._resolve_configured_model_key(configured)
        if resolved_key and not self._is_provider_rate_limited(
            self.models[resolved_key].provider
        ):
            explicit_override = (os.getenv("SPARKLEFORGE_EXPLICIT_MODEL") or "").strip()
            # Explicit CLI `--model` overrides apply across all task types;
            # default/env-configured models follow capability filtering (issue #1223).
            if explicit_override and configured == explicit_override:
                logger.info(
                    f"Selected explicit CLI model for {task_type.value}: {resolved_key}"
                )
                return resolved_key
            if task_type in self.models[resolved_key].capabilities:
                logger.info(f"Selected configured model for {task_type.value}: {resolved_key}")
                return resolved_key
            logger.info(
                f"Configured model {resolved_key} lacks {task_type.value} capability, "
                f"falling back to capability-based selection"
            )

        # 작업 유형에 적합한 모델 필터링
        suitable_models = [
            name for name, config in self.models.items() if task_type in config.capabilities
        ]

        if not suitable_models:
            # 기본 모델 사용
            return "gemini-flash-lite"

        # 사용 가능한 Provider 목록 가져오기
        available_providers = self._get_available_providers()

        if not available_providers:
            # 모든 Provider가 rate limit에 걸린 경우, 가장 오래된 것부터 재시도
            logger.warning("All providers are rate-limited, using oldest rate-limited provider")
            if self.provider_rate_limited:
                oldest_provider = min(self.provider_rate_limited.items(), key=lambda x: x[1])[0]
                available_providers = [oldest_provider]
            else:
                # 폴백: Gemini 모델 사용
                gemini_models = [
                    name for name in suitable_models if self.models[name].provider == "google"
                ]
                if gemini_models:
                    return gemini_models[0]
                return "gemini-flash-lite"

        # Provider 로테이션: 사용 횟수가 가장 적은 Provider 선택
        provider_usage = {
            provider: self.provider_usage_count.get(provider, 0) for provider in available_providers
        }

        # 사용 횟수가 가장 적은 Provider 선택 (동일하면 순서대로)
        selected_provider = min(
            available_providers,
            key=lambda p: (provider_usage[p], available_providers.index(p)),
        )

        # 사용 횟수 증가
        self.provider_usage_count[selected_provider] = provider_usage[selected_provider] + 1

        # 해당 Provider의 모델 선택
        provider_models = [
            name for name in suitable_models if self.models[name].provider == selected_provider
        ]

        if provider_models:
            selected_model = provider_models[0]
            logger.info(
                f"Selected {selected_provider.upper()} model (rotation #{self.provider_usage_count[selected_provider]}): {selected_model}"
            )
            return selected_model

        # 해당 Provider에 모델이 없으면 다음 Provider 시도
        remaining_providers = [p for p in available_providers if p != selected_provider]
        if remaining_providers:
            return self.select_model(task_type, complexity, budget)

        # 모든 Provider 실패 시 폴백: Gemini 모델 사용
        gemini_models = [name for name in suitable_models if self.models[name].provider == "google"]
        if gemini_models:
            return gemini_models[0]

        return "gemini-flash-lite"


