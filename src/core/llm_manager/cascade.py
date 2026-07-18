"""Cascade classification, draft-quality validation, and fallback logic.

Split out of the former monolithic llm_manager.py (issue #582).
"""

import logging
from typing import Any, Dict, List, Tuple

from src.core.llm_manager.types import TaskType
from src.core.researcher_config import get_cascade_config

logger = logging.getLogger(__name__)


class CascadeMixin:
    """Cascade model classification and provider-fallback execution."""

    def _get_provider_models(self, provider: str, task_type: TaskType) -> List[str]:
        """Provider의 모든 사용 가능한 모델 리스트 반환.

        Args:
            provider: Provider 이름 (openrouter, groq, google, openai 등)
            task_type: 작업 유형

        Returns:
            해당 provider의 모델 이름 리스트 (비용 오름차순 정렬)
        """
        models = [
            name
            for name, config in self.models.items()
            if config.provider == provider and task_type in config.capabilities
        ]

        # 비용 기준 정렬 (저비용 우선)
        models.sort(key=lambda name: self.models[name].cost_per_token)

        return models


    def _classify_models_for_cascade(
        self, provider_models: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Provider 모델 리스트를 Drafter/Verifier로 분류.

        기준:
        - Drafter: cost_per_token < threshold 또는 speed_rating > threshold
        - Verifier: cost_per_token >= threshold 또는 quality_rating > threshold

        Returns:
            (drafter_models, verifier_models) 튜플
        """
        try:
            cascade_config = get_cascade_config()
        except RuntimeError:
            cascade_config = None

        if cascade_config:
            drafter_cost_threshold = cascade_config.drafter_cost_threshold
            drafter_speed_threshold = cascade_config.drafter_speed_threshold
            verifier_quality_threshold = cascade_config.verifier_quality_threshold
        else:
            # 기본값
            drafter_cost_threshold = 0.0002
            drafter_speed_threshold = 7.0
            verifier_quality_threshold = 8.0

        drafter_models = []
        verifier_models = []

        for model_name in provider_models:
            config = self.models[model_name]

            is_drafter = (
                config.cost_per_token < drafter_cost_threshold
                or config.speed_rating > drafter_speed_threshold
            )

            is_verifier = (
                config.cost_per_token >= drafter_cost_threshold
                or config.quality_rating > verifier_quality_threshold
            )

            if is_drafter:
                drafter_models.append(model_name)
            if is_verifier:
                verifier_models.append(model_name)

        # 기본값: 첫 번째 모델을 drafter, 마지막 모델을 verifier로
        if not drafter_models:
            drafter_models = [provider_models[0]] if provider_models else []
        if not verifier_models:
            verifier_models = [provider_models[-1]] if provider_models else []

        return drafter_models, verifier_models


    def _domain_validate_content(
        self, task_type: TaskType, content: str, min_length: int = 20
    ) -> bool:
        """도메인별 콘텐츠 휴리스틱 검증 (cascadeflow 스타일).

        - PLANNING: 단계/목록 구조 존재
        - RESEARCH: 최소 길이 및 정보성
        - GENERATION: 문단/문장 구조
        - VERIFICATION: 판정/결론 키워드
        """
        if not content or len(content.strip()) < min_length:
            return False
        text = content.strip().lower()
        if task_type == TaskType.PLANNING:
            has_structure = (
                any(c in content for c in ["1.", "2.", "①", "1)", "- ", "* "])
                or "step" in text
                or "단계" in text
                or "phase" in text
            )
            return has_structure or len(content) > 200
        if task_type == TaskType.RESEARCH:
            return len(content) >= 100
        if task_type == TaskType.GENERATION:
            return len(content) >= 80 and ("\n\n" in content or ". " in content or "。" in content)
        if task_type == TaskType.VERIFICATION:
            verdict_like = (
                "verified" in text
                or "valid" in text
                or "confirm" in text
                or "검증" in text
                or "확인" in text
                or "결론" in text
            )
            return verdict_like or len(content) >= 50
        return True


    def _validate_draft_quality(
        self,
        draft_result: Dict[str, Any],
        prompt: str,
        task_type: TaskType,
        complexity: float = 5.0,
    ) -> bool:
        """Draft 품질 검증.

        Cascadeflow 스타일: confidence, 도메인별 콘텐츠 검증, (옵션) semantic agreement.
        """
        try:
            cascade_config = get_cascade_config()
        except RuntimeError:
            cascade_config = None

        base_threshold = cascade_config.confidence_threshold if cascade_config else 0.75
        enable_adaptive = cascade_config.enable_adaptive_threshold if cascade_config else True
        domain_validation = (
            getattr(cascade_config, "domain_validation_enabled", True) if cascade_config else True
        )

        # Complexity 기반 threshold 조정
        if enable_adaptive:
            if complexity > 7.0:
                threshold = 0.85
            elif complexity > 5.0:
                threshold = 0.80
            else:
                threshold = base_threshold
        else:
            threshold = base_threshold

        # Confidence 체크
        confidence = draft_result.get("confidence", 0.0)
        if confidence >= threshold:
            pass
        else:
            quality_score = draft_result.get("quality_score", None)
            if quality_score is not None and quality_score >= threshold:
                pass
            else:
                return False

        # 도메인별 콘텐츠 검증
        if domain_validation:
            content = draft_result.get("content") or ""
            if not self._domain_validate_content(task_type, content):
                logger.debug(f"Draft rejected by domain validation (task_type={task_type})")
                return False

        return True


    async def _execute_single_model_by_provider(
        self,
        model_name: str,
        prompt: str,
        system_message: str = None,
        task_type: TaskType = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Provider별 단일 모델 실행 (기존 로직 재사용).

        기존의 _execute_openrouter_model, _execute_groq_model 등을 호출.
        """
        model_provider = self.models[model_name].provider

        if model_provider == "openrouter":
            return await self._execute_openrouter_model(
                model_name, prompt, system_message, **kwargs
            )
        elif model_provider == "groq":
            return await self._execute_groq_model(model_name, prompt, system_message, **kwargs)
        elif model_provider == "google":
            if model_name.endswith("_langchain"):
                return await self._execute_langchain_model(
                    model_name, prompt, system_message, **kwargs
                )
            else:
                return await self._execute_gemini_model(
                    model_name, prompt, system_message, **kwargs
                )
        elif model_provider == "openai":
            return await self._execute_openai_model(model_name, prompt, system_message, **kwargs)
        elif model_provider == "nvidia":
            return await self._execute_nvidia_model(model_name, prompt, system_message, **kwargs)
        elif model_provider == "cerebras":
            return await self._execute_cerebras_model(model_name, prompt, system_message, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {model_provider}")


    async def _execute_provider_cascade(
        self,
        provider_models: List[str],
        prompt: str,
        system_message: str = None,
        task_type: TaskType = None,
        complexity: float = 5.0,
        **kwargs,
    ) -> Tuple[Dict[str, Any], str]:
        """Provider 내부 모델 리스트로 Cascade 실행.

        흐름:
        1. Drafter/Verifier 분류
        2. Drafter로 실행
        3. Quality validation
        4. Accept → Drafter 결과 반환
        5. Reject → Verifier로 승격

        Returns:
            (result_dict, actual_model_used) 튜플
        """
        # 1. Drafter/Verifier 분류
        drafter_models, verifier_models = self._classify_models_for_cascade(provider_models)

        drafter = drafter_models[0]
        verifier = verifier_models[0] if verifier_models else provider_models[-1]

        # 2. Drafter 실행
        logger.info(f"Executing cascade drafter: {drafter}")
        draft_result = await self._execute_single_model_by_provider(
            drafter, prompt, system_message, task_type, **kwargs
        )

        # 3. Quality validation (confidence + domain heuristics)
        should_accept = self._validate_draft_quality(draft_result, prompt, task_type, complexity)

        # 3b. Optional semantic agreement: second drafter run and similarity check
        try:
            cascade_cfg = get_cascade_config()
            if should_accept and getattr(cascade_cfg, "semantic_agreement_enabled", False):
                draft_2 = await self._execute_single_model_by_provider(
                    drafter, prompt, system_message, task_type, **kwargs
                )
                c1 = (draft_result.get("content") or "").strip()
                c2 = (draft_2.get("content") or "").strip()
                if c1 and c2:
                    w1 = set(c1.lower().split())
                    w2 = set(c2.lower().split())
                    if w1 or w2:
                        jaccard = len(w1 & w2) / len(w1 | w2)
                        th = getattr(
                            cascade_cfg,
                            "semantic_agreement_threshold",
                            0.7,
                        )
                        if jaccard < th:
                            should_accept = False
                            logger.info(f"Semantic agreement failed (jaccard={jaccard:.2f} < {th})")
        except Exception as e:
            logger.debug(f"Semantic agreement check skipped: {e}")

        if should_accept:
            logger.info(f"✓ Draft accepted: {drafter}")
            return draft_result, drafter

        # 3c. 비용 상한(C3PO 스타일): escalation 비용이 cap 초과 시 draft 수용
        try:
            cascade_cfg_cost = get_cascade_config()
            cap = getattr(cascade_cfg_cost, "cost_cap_per_run", None)
        except RuntimeError:
            cap = None
        if cap is not None and cap > 0:
            try:
                verifier_config = self.models.get(verifier)
                if verifier_config:
                    est_tokens = int(len(prompt.split()) * 1.3) + 500
                    est_cost = est_tokens * verifier_config.cost_per_token
                    if est_cost > cap:
                        logger.info(
                            f"Cost cap exceeded (est={est_cost:.4f} > {cap}), accepting draft"
                        )
                        return draft_result, drafter
            except Exception as e:
                logger.debug(f"Cost cap check skipped: {e}")

        # 4. Verifier로 승격
        logger.info(f"✗ Draft rejected, escalating to verifier: {verifier}")
        verifier_result = await self._execute_single_model_by_provider(
            verifier, prompt, system_message, task_type, **kwargs
        )

        return verifier_result, verifier


    async def _try_fallback_models(
        self,
        task_type: TaskType,
        prompt: str,
        system_message: str = None,
        skip_providers: List[str] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], str]:
        """우선순위에 따라 폴백 모델 시도: OpenRouter -> Groq -> Cerebras -> Gemini -> GPT -> Claude."""
        if skip_providers is None:
            skip_providers = []

        # 키가 있는 무료 provider 풀 전체를 순서대로 시도
        fallback_order = list(self.provider_rotation_order)

        for provider in fallback_order:
            if provider in skip_providers:
                continue
            if self._is_provider_rate_limited(provider):
                continue

            # 해당 provider의 사용 가능한 모델 찾기
            available_models = [
                name
                for name, config in self.models.items()
                if config.provider == provider and task_type in config.capabilities
            ]

            if not available_models:
                continue

            # 첫 번째 사용 가능한 모델 시도
            fallback_model = available_models[0]
            logger.info(f"Trying fallback model: {fallback_model} (provider: {provider})")

            try:
                if provider == "openrouter":
                    result = await self._execute_openrouter_model(
                        fallback_model, prompt, system_message, **kwargs
                    )
                elif provider == "groq":
                    result = await self._execute_groq_model(
                        fallback_model, prompt, system_message, **kwargs
                    )
                elif provider == "cerebras":
                    result = await self._execute_cerebras_model(
                        fallback_model, prompt, system_message, **kwargs
                    )
                elif provider == "google":
                    result = await self._execute_gemini_model(
                        fallback_model, prompt, system_message, **kwargs
                    )
                elif provider == "openai":
                    result = await self._execute_openai_model(
                        fallback_model, prompt, system_message, **kwargs
                    )
                elif provider == "nvidia":
                    result = await self._execute_nvidia_model(
                        fallback_model, prompt, system_message, **kwargs
                    )
                elif provider == "claude":
                    # Claude는 OpenAI API 호환 또는 OpenRouter를 통해 접근
                    # OpenRouter를 통해 Claude 모델 찾기
                    claude_models = [
                        name
                        for name, config in self.models.items()
                        if config.provider == "openrouter"
                        and "claude" in config.model_id.lower()
                        and task_type in config.capabilities
                    ]
                    if claude_models:
                        result = await self._execute_openrouter_model(
                            claude_models[0], prompt, system_message, **kwargs
                        )
                    else:
                        continue
                else:
                    continue

                logger.info(f"✅ Fallback successful with {fallback_model}")
                return result, fallback_model
            except Exception as e:
                # 에러 메시지에서 HTML 필터링 및 중첩 방지
                error_str = str(e)

                # 모델 존재하지 않음 (404) 또는 Decommissioned 모델 감지
                if (
                    "does not exist" in error_str.lower()
                    or "model_not_found" in error_str.lower()
                    or "decommissioned" in error_str.lower()
                    or "model_decommissioned" in error_str.lower()
                ):
                    logger.warning(
                        f"Fallback model {fallback_model} is not available (404/decommissioned), trying next..."
                    )
                    # Groq 모델이 존재하지 않는 경우 모델 목록에서 제거
                    if provider == "groq" and fallback_model in self.models:
                        logger.warning(
                            f"Removing unavailable Groq model from available models: {fallback_model}"
                        )
                        del self.models[fallback_model]
                    continue

                # Rate limit 에러 (429)는 재시도 가능하지만 fallback에서는 다음 모델로
                if (
                    "429" in error_str
                    or "rate limit" in error_str.lower()
                    or "rate-limited" in error_str.lower()
                    or "rate limit exceeded" in error_str.lower()
                ):
                    logger.warning(
                        f"Fallback model {fallback_model} rate limited (429), trying next..."
                    )
                    continue

                if "<!DOCTYPE html>" in error_str or "<html" in error_str.lower():
                    import re

                    status_match = re.search(r"(\d{3})", error_str)
                    status_code = status_match.group(1) if status_match else "Unknown"
                    title_match = re.search(r"<title>([^<]+)</title>", error_str, re.IGNORECASE)
                    if title_match:
                        error_msg = f"HTTP {status_code}: {title_match.group(1).strip()}"
                    else:
                        error_msg = f"HTTP {status_code}: Server Error"
                else:
                    # 중첩된 메시지 방지: "OpenRouter model X failed: OpenRouter model X failed: ..." 형식 제거
                    if "failed:" in error_str:
                        # 마지막 "failed:" 이후의 메시지만 사용
                        parts = error_str.split("failed:")
                        if len(parts) > 1:
                            error_msg = parts[-1].strip()
                        else:
                            error_msg = error_str[:200]
                    else:
                        error_msg = error_str[:200]

                logger.warning(
                    f"Fallback model {fallback_model} failed: {error_msg}, trying next..."
                )
                continue

        # 모든 폴백 실패
        raise RuntimeError("All fallback models failed. No available models.")


