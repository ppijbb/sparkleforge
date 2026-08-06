"""MultiModelOrchestrator: the composed orchestrator class.

Split out of the former monolithic llm_manager.py (issue #582, mirroring the
Sigma-1 split of mcp_integration.py -- module by responsibility, facade
re-export kept at src/core/llm_manager/__init__.py). Composes the mixins in
this package (model registry, routing, provider adapters, cascade/fallback)
and adds the main execute_with_model dispatcher and weighted_ensemble.
"""

import logging
import sys
import time
import asyncio
from typing import Any, Dict, List

import requests

from src.core.llm_manager.cascade import CascadeMixin
from src.core.llm_manager.connection_pool import ConnectionPool
from src.core.llm_manager.model_registry import ModelRegistryMixin
from src.core.llm_manager.performance_tracker import ModelPerformanceTracker
from src.core.llm_manager.progress import with_progress
from src.core.llm_manager.providers import ProviderAdaptersMixin
from src.core.llm_manager.routing import RoutingMixin
from src.core.llm_manager.types import ModelConfig, ModelResult, TaskType
from src.core.researcher_config import get_agent_config, get_cascade_config, get_llm_config

logger = logging.getLogger(__name__)


class MultiModelOrchestrator(
    ModelRegistryMixin,
    RoutingMixin,
    ProviderAdaptersMixin,
    CascadeMixin,
):
    """다중 모델 오케스트레이터 (혁신 3)."""

    def __init__(self):
        self.llm_config = get_llm_config()
        self.agent_config = get_agent_config()

        # Provider별 API 키 검증
        self._validate_provider_config()

        self.models: Dict[str, ModelConfig] = {}
        self.performance_tracker = ModelPerformanceTracker()
        self.model_clients: Dict[str, Any] = {}

        # Connection pooling for performance optimization
        self.connection_pool = ConnectionPool(pool_size=5, max_idle_time=300)

        # Provider 로테이션 추적
        self.provider_rotation_index = 0  # 현재 Provider 인덱스
        # 키가 설정된 무료 provider를 모두 로테이션 풀에 포함
        self.provider_rotation_order = self._build_provider_rotation_order()
        self.provider_rate_limited = {}  # Rate limit에 걸린 Provider (timestamp)
        self.provider_usage_count = {
            provider: 0 for provider in self.provider_rotation_order
        }  # 사용 횟수 추적

        self._initialize_models()
        self._initialize_clients()

    async def _run_with_feedback(self, coro, provider: str, model: str):
        """Run a provider call with a live terminal feedback ticker."""
        if not sys.stdout.isatty():
            return await coro

        start_time = time.time()
        stop_event = asyncio.Event()

        async def ticker():
            while not stop_event.is_set():
                elapsed = int(time.time() - start_time)
                print(f"\r⏳ {provider}/{model}... {elapsed}s", end="", flush=True)
                await asyncio.sleep(1)
            print("\r" + " " * 40 + "\r", end="", flush=True)

        task = asyncio.create_task(ticker())
        try:
            return await coro
        finally:
            stop_event.set()
            await task

    async def execute_with_model(
        self,
        prompt: str,
        task_type: TaskType,
        model_name: str = None,
        system_message: str = None,
        use_cascade: bool = True,
        complexity: float = 5.0,
        history_messages: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> ModelResult:
        """모델로 실행 - Cascade 및 Tool support 지원."""
        if history_messages:
            kwargs["history_messages"] = history_messages
        if model_name is None:
            model_name = self.select_model(task_type)

        # 모델 클라이언트 확인
        model_name_clean = model_name.replace("_langchain", "")

        # 모델 provider 확인
        if model_name_clean not in self.models:
            raise ValueError(f"Model {model_name_clean} not found in models")

        model_provider = self.models[model_name_clean].provider
        model_config = self.models[model_name_clean]
        start_time = time.time()
        actual_model_used = model_name_clean  # 실제 사용된 모델 추적

        # prompt와 system_message는 execute_llm_task의 decorator에서 자동으로 최적화됨

        try:
            # Cascade 설정 확인
            try:
                cascade_config = get_cascade_config()
            except RuntimeError:
                cascade_config = None

            cascade_enabled = (
                (use_cascade and cascade_config and cascade_config.enabled)
                if cascade_config
                else use_cascade
            )

            # Provider의 모든 모델 리스트 가져오기
            provider_models = self._get_provider_models(model_provider, task_type)

            # Cascade 실행 조건 체크
            use_cascade_for_provider = cascade_enabled and len(provider_models) >= (
                cascade_config.min_models_for_cascade if cascade_config else 2
            )

            if use_cascade_for_provider:
                # Cascade 실행
                logger.info(
                    f"Using cascade for provider {model_provider} with {len(provider_models)} models"
                )
                try:
                    result, actual_model_used = await self._execute_provider_cascade(
                        provider_models,
                        prompt,
                        system_message,
                        task_type,
                        complexity,
                        **kwargs,
                    )
                except Exception as cascade_error:
                    logger.warning(
                        f"Cascade execution failed: {cascade_error}, falling back to single model..."
                    )
                    # Cascade 실패 시 기존 단일 모델 실행 로직으로 fallback
                    use_cascade_for_provider = False

            if not use_cascade_for_provider and model_provider == "openrouter":
                # 기존 단일 모델 실행 로직
                # 우선순위에 따라 모델 실행 및 폴백
                if model_provider == "openrouter":
                    logger.info(f"Executing with OpenRouter model: {model_name_clean}")
                try:
                    result = await self._run_with_feedback(
                        self._execute_openrouter_model(model_name_clean, prompt, system_message, **kwargs),
                        "openrouter", model_name_clean
                    )
                except Exception as error:
                    error_str = str(error).lower()
                    # Rate limit 에러인 경우 Provider를 rate-limited로 표시
                    if "rate-limited" in error_str or "429" in error_str:
                        self._mark_provider_rate_limited("openrouter")
                    logger.warning(
                        f"OpenRouter model {model_name_clean} failed: {error}, trying fallback..."
                    )
                    result, actual_model_used = await self._try_fallback_models(
                        task_type,
                        prompt,
                        system_message,
                        skip_providers=["openrouter"],
                        **kwargs,
                    )
            elif not use_cascade_for_provider and model_provider == "groq":
                logger.info(f"Executing with Groq model: {model_name_clean}")
                try:
                    result = await self._run_with_feedback(
                        self._execute_groq_model(model_name_clean, prompt, system_message, **kwargs),
                        "groq", model_name_clean
                    )
                except Exception as error:
                    error_str = str(error).lower()
                    # Rate limit 에러인 경우 Provider를 rate-limited로 표시
                    if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                        self._mark_provider_rate_limited("groq")
                    logger.warning(
                        f"Groq model {model_name_clean} failed: {error}, trying fallback..."
                    )
                    result, actual_model_used = await self._try_fallback_models(
                        task_type,
                        prompt,
                        system_message,
                        skip_providers=["openrouter", "groq"],
                        **kwargs,
                    )
            elif not use_cascade_for_provider and model_provider == "cerebras":
                logger.info(f"Executing with Cerebras model: {model_name_clean}")
                try:
                    result = await self._run_with_feedback(
                        self._execute_cerebras_model(model_name_clean, prompt, system_message, **kwargs),
                        "cerebras", model_name_clean
                    )
                except Exception as error:
                    error_str = str(error).lower()
                    if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                        self._mark_provider_rate_limited("cerebras")
                    logger.warning(
                        f"Cerebras model {model_name_clean} failed: {error}, trying fallback..."
                    )
                    result, actual_model_used = await self._try_fallback_models(
                        task_type,
                        prompt,
                        system_message,
                        skip_providers=["openrouter", "groq", "cerebras"],
                        **kwargs,
                    )
            elif not use_cascade_for_provider and model_provider == "google":
                logger.info(f"Executing with Gemini model: {model_name_clean}")
                try:
                    if model_name.endswith("_langchain"):
                        result = await self._run_with_feedback(
                            self._execute_langchain_model(model_name, prompt, system_message, **kwargs),
                            "gemini", model_name
                        )
                    else:
                        result = await self._run_with_feedback(
                            self._execute_gemini_model(model_name, prompt, system_message, **kwargs),
                            "gemini", model_name
                        )
                except Exception as error:
                    logger.warning(
                        f"Gemini model {model_name_clean} failed: {error}, trying fallback..."
                    )
                    result, actual_model_used = await self._try_fallback_models(
                        task_type,
                        prompt,
                        system_message,
                        skip_providers=["openrouter", "groq", "google"],
                        **kwargs,
                    )
            elif not use_cascade_for_provider and model_provider == "openai":
                logger.info(f"Executing with GPT model: {model_name_clean}")
                try:
                    result = await self._run_with_feedback(
                        self._execute_openai_model(model_name_clean, prompt, system_message, **kwargs),
                        "openai", model_name_clean
                    )
                except Exception as error:
                    logger.warning(
                        f"GPT model {model_name_clean} failed: {error}, trying fallback..."
                    )
                    result, actual_model_used = await self._try_fallback_models(
                        task_type,
                        prompt,
                        system_message,
                        skip_providers=["openrouter", "groq", "google", "openai"],
                        **kwargs,
                    )
            elif not use_cascade_for_provider and model_provider == "nvidia":
                logger.info(f"Executing with NVIDIA NIM model: {model_name_clean}")
                try:
                    result = await self._run_with_feedback(
                        self._execute_nvidia_model(model_name_clean, prompt, system_message, **kwargs),
                        "nvidia", model_name_clean
                    )
                except Exception as error:
                    error_str = str(error).lower()
                    if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
                        self._mark_provider_rate_limited("nvidia")
                    logger.warning(
                        f"NVIDIA NIM model {model_name_clean} failed: {error}, trying fallback..."
                    )
                    result, actual_model_used = await self._try_fallback_models(
                        task_type,
                        prompt,
                        system_message,
                        skip_providers=["nvidia"],
                        **kwargs,
                    )
            elif not use_cascade_for_provider:
                raise ValueError(f"Unknown provider: {model_provider}")

            execution_time = time.time() - start_time

            # 비용 계산 (실제 사용된 모델 기준)
            if actual_model_used in self.models:
                model_config = self.models[actual_model_used]
                cost = len(prompt.split()) * model_config.cost_per_token
            else:
                cost = 0.0

            # 성능 기록 (실제 사용된 모델 기준)
            self.performance_tracker.record_execution(
                actual_model_used,
                task_type,
                execution_time,
                True,
                result.get("quality_score", 0.8),
            )

            return ModelResult(
                content=result["content"],
                model_used=actual_model_used,  # 실제 사용된 모델 반환
                execution_time=execution_time,
                confidence=result.get("confidence", 0.8),
                cost=cost,
                metadata=result.get("metadata", {}),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Model execution failed: {e}")

            # 실패 기록 (실제 사용된 모델 기준)
            self.performance_tracker.record_execution(
                actual_model_used, task_type, execution_time, False
            )

            raise


    async def weighted_ensemble(
        self,
        prompt: str,
        task_type: TaskType,
        models: List[str] = None,
        weights: List[float] = None,
    ) -> ModelResult:
        """Weighted Ensemble 실행."""
        if models is None:
            models = [self.select_model(task_type) for _ in range(3)]

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        # weights 타입 검증 및 변환 (엄격한 검증)
        if weights:
            try:
                validated_weights = []
                for w in weights:
                    if isinstance(w, (int, float)):
                        # 숫자 타입은 그대로 사용 (0 이상인지 확인)
                        validated_weights.append(max(0.0, float(w)))
                    elif isinstance(w, str):
                        # 문자열인 경우 숫자로 변환 시도
                        # 먼저 숫자가 아닌 문자 제거 (공백, 문자 등)
                        cleaned_str = "".join(
                            c for c in w if c.isdigit() or c == "." or c == "-" or c == "+"
                        )

                        # '.' 만 있거나 숫자가 없는 경우 처리
                        if (
                            not cleaned_str
                            or cleaned_str == "."
                            or cleaned_str in ["-", "+", "-.", "+."]
                        ):
                            logger.warning(
                                f"Invalid weight value '{w}' (no valid number), using 1.0"
                            )
                            validated_weights.append(1.0)
                        else:
                            try:
                                float_val = float(cleaned_str)
                                validated_weights.append(max(0.0, float_val))
                            except (ValueError, TypeError):
                                # 변환 실패 시 기본값 1.0 사용
                                logger.warning(
                                    f"Invalid weight value '{w}' (cleaned: '{cleaned_str}'), using 1.0"
                                )
                                validated_weights.append(1.0)
                    else:
                        # 기타 타입은 기본값 사용
                        logger.warning(f"Invalid weight type '{type(w)}', using 1.0")
                        validated_weights.append(1.0)

                # 검증된 weights 사용
                if len(validated_weights) == len(weights):
                    weights = validated_weights
                else:
                    raise ValueError("Weight validation failed")
            except Exception as e:
                logger.warning(f"Invalid weights format, using equal weights: {e}")
                weights = [1.0 / len(models)] * len(models)

        # 모든 모델로 실행
        results = []
        for model in models:
            try:
                result = await self.execute_with_model(prompt, task_type, model)
                results.append(result)
            except Exception as e:
                logger.warning(f"Model {model} failed in ensemble: {e}")
                continue

        if not results:
            raise RuntimeError("All models failed in ensemble")

        # weights 개수를 results 개수에 맞춤
        if len(weights) > len(results):
            weights = weights[: len(results)]
        elif len(weights) < len(results):
            # 부족한 weights는 동일하게 분배
            remaining = 1.0 - sum(weights[: len(weights)])
            weights.extend(
                [remaining / (len(results) - len(weights))] * (len(results) - len(weights))
            )

        # 가중 평균으로 결과 통합
        try:
            total_weight = sum(weights)
            if total_weight <= 0:
                # 모든 weight가 0이거나 음수면 동일하게 분배
                logger.warning("Total weight is 0 or negative, using equal weights")
                weights = [1.0 / len(results)] * len(results)
                total_weight = 1.0
        except (TypeError, ValueError) as e:
            logger.error(f"Error calculating total weight: {e}, weights: {weights}")
            # 기본값 사용
            weights = [1.0 / len(results)] * len(results)
            total_weight = 1.0
        weighted_content = ""
        total_confidence = 0.0
        total_cost = 0.0
        total_time = 0.0

        for i, result in enumerate(results):
            # 안전한 weight 계산
            try:
                weight = (
                    float(weights[i]) / total_weight if total_weight > 0 else 1.0 / len(results)
                )
            except (TypeError, ValueError, IndexError) as e:
                logger.warning(f"Error calculating weight for result {i}: {e}, using equal weight")
                weight = 1.0 / len(results)

            weighted_content += f"[{result.model_used}] {result.content}\n\n"
            total_confidence += result.confidence * weight
            total_cost += result.cost * weight
            total_time += result.execution_time * weight

        return ModelResult(
            content=weighted_content.strip(),
            model_used=f"ensemble({','.join([r.model_used for r in results])})",
            execution_time=total_time,
            confidence=total_confidence,
            cost=total_cost,
            metadata={
                "ensemble_models": [r.model_used for r in results],
                "weights": weights[: len(results)],
                "individual_results": [
                    {"model": r.model_used, "confidence": r.confidence, "cost": r.cost}
                    for r in results
                ],
            },
        )


    def get_model_performance_stats(self) -> Dict[str, Any]:
        """모델 성능 통계 반환."""
        stats = {}
        for model_name in self.models:
            stats[model_name] = {
                "overall_score": self.performance_tracker.get_model_score(model_name),
                "task_scores": {
                    task_type.value: self.performance_tracker.get_model_score(model_name, task_type)
                    for task_type in TaskType
                },
            }
        return stats


    def get_best_model_for_task(self, task_type: TaskType) -> str:
        """작업에 최적 모델 반환."""
        best_model = None
        best_score = 0.0

        for model_name in self.models:
            score = self.performance_tracker.get_model_score(model_name, task_type)
            if score > best_score:
                best_score = score
                best_model = model_name

        return best_model or "gemini-flash-lite"


# Global orchestrator instance (lazy initialization)
_llm_orchestrator = None



