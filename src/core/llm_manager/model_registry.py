"""Provider model-catalog loading and client initialization.

Split out of the former monolithic llm_manager.py (issue #582). The single
biggest chunk of the original MultiModelOrchestrator -- building the model
registry (Google/OpenRouter/Groq/OpenAI/NVIDIA/Cerebras) and initializing
provider clients.
"""

import logging
import os
import warnings

from typing import Any, Dict, List, Optional

import requests

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        import google.generativeai as genai
    except ImportError:
        genai = None  # type: ignore[assignment]
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]

from src.core.llm_manager.types import ModelConfig, TaskType

logger = logging.getLogger(__name__)

# Safety settings to block nothing (allow all content)
# This is required for the research agent to function without being blocked by safety filters
# for harmless queries or research topics.
# Note: ChatGoogleGenerativeAI does not support safety_settings parameter directly.
# Safety settings are handled at the genai.GenerativeModel level, not in LangChain wrapper.
# Setting to None to avoid validation errors.
SAFETY_SETTINGS_BLOCK_NONE = None


def _parse_openrouter_json_response(response, context: str) -> dict:
    """Parse an OpenRouter JSON response and raise a clear error on invalid JSON.

    Shared by model_registry.py (fetching the models catalog) and
    providers.py (chat completions) -- lives here rather than in
    providers.py to avoid a model_registry <-> providers import cycle, since
    providers.py already imports SAFETY_SETTINGS_BLOCK_NONE from this module.
    """
    try:
        data = response.json()
    except ValueError as exc:
        body = getattr(response, "text", "") or ""
        status = getattr(response, "status_code", "unknown")
        snippet = body[:200].replace("\n", " ")
        logger.warning(
            "OpenRouter returned non-JSON response during %s: status=%s body=%r",
            context,
            status,
            snippet,
        )
        raise RuntimeError(
            f"OpenRouter returned non-JSON response during {context}: HTTP {status}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            f"OpenRouter returned invalid JSON shape during {context}: "
            f"{type(data).__name__}"
        )
    return data


class ModelRegistryMixin:
    """Model catalog loading (per-provider) and client initialization."""

    def _validate_provider_config(self):
        """Provider별 API 키 검증."""
        # API 키가 없어도 경고만 출력 (폴백 메커니즘 사용)
        if not os.getenv("OPENROUTER_API_KEY"):
            logger.warning("OPENROUTER_API_KEY not found - OpenRouter models will be unavailable")
        if not os.getenv("GROQ_API_KEY"):
            logger.warning("GROQ_API_KEY not found - Groq models will be unavailable")
        if not (os.getenv("GOOGLE_API_KEY") or self.llm_config.api_key):
            logger.warning("GOOGLE_API_KEY not found - Gemini models will be unavailable")
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found - GPT models will be unavailable")
        if not os.getenv("NVIDIA_API_KEY"):
            logger.warning("NVIDIA_API_KEY not found - NVIDIA NIM models will be unavailable")
        if not os.getenv("CEREBRAS_API_KEY"):
            logger.warning("CEREBRAS_API_KEY not found - Cerebras models will be unavailable")


    def _initialize_models(self):
        """모델 초기화."""
        # 모델 로딩: 우선순위에 따라 로드 (다른 provider와 동일하게 API 키 존재 시에만 등록)
        # 1. OpenRouter 모델 (최우선)
        if os.getenv("OPENROUTER_API_KEY"):
            try:
                self._load_openrouter_models()
                logger.info("✅ OpenRouter models loaded")
            except Exception as e:
                logger.warning(f"OpenRouter models not loaded: {e}")
        else:
            logger.info("OpenRouter disabled - OPENROUTER_API_KEY not found")

        # 2. Groq 모델
        if os.getenv("GROQ_API_KEY"):
            try:
                self._load_groq_models()
                logger.info("✅ Groq models loaded")
            except Exception as e:
                logger.warning(f"Groq models not loaded: {e}")
        else:
            logger.info("Groq disabled - GROQ_API_KEY not found")

        # 3. Gemini 모델
        if os.getenv("GOOGLE_API_KEY"):
            try:
                self._load_google_models()
                logger.info("✅ Gemini models loaded")
            except Exception as e:
                logger.warning(f"Gemini models not loaded: {e}")
        else:
            logger.info("Gemini disabled - GOOGLE_API_KEY not found")

        # 4. GPT 모델
        if os.getenv("OPENAI_API_KEY"):
            try:
                self._load_openai_models()
                logger.info("✅ OpenAI/GPT models loaded")
            except Exception as e:
                logger.warning(f"OpenAI/GPT models not loaded: {e}")
        else:
            logger.info("OpenAI/GPT disabled - OPENAI_API_KEY not found")

        # 5. NVIDIA NIM 모델
        if os.getenv("NVIDIA_API_KEY"):
            try:
                self._load_nvidia_models()
                logger.info("✅ NVIDIA NIM models loaded")
            except Exception as e:
                logger.warning(f"NVIDIA NIM models not loaded: {e}")
        else:
            logger.info("NVIDIA NIM disabled - NVIDIA_API_KEY not found")

        # 6. Cerebras 모델
        if os.getenv("CEREBRAS_API_KEY"):
            try:
                self._load_cerebras_models()
                logger.info("✅ Cerebras models loaded")
            except Exception as e:
                logger.warning(f"Cerebras models not loaded: {e}")
        else:
            logger.info("Cerebras disabled - CEREBRAS_API_KEY not found")


    def _load_google_models(self):
        """Gemini 모델 로딩 (GOOGLE_API_KEY가 있을 때만 호출됨)."""
        # Gemini Flash Lite (빠른 계획, 압축)
        self.models["gemini-flash-lite"] = ModelConfig(
            name="gemini-flash-lite",
            provider="google",
            model_id="gemini-3.5-flash-lite",
            temperature=0.1,
            max_tokens=2000,
            cost_per_token=0.0001,
            speed_rating=9.0,
            quality_rating=7.0,
            capabilities=[TaskType.PLANNING, TaskType.COMPRESSION, TaskType.RESEARCH],
        )

        # Gemini Pro (복잡한 추론, 분석)
        self.models["gemini-pro"] = ModelConfig(
            name="gemini-pro",
            provider="google",
            model_id="gemini-pro-latest",
            temperature=0.2,
            max_tokens=4000,
            cost_per_token=0.0005,
            speed_rating=6.0,
            quality_rating=9.0,
            capabilities=[
                TaskType.DEEP_REASONING,
                TaskType.ANALYSIS,
                TaskType.SYNTHESIS,
            ],
        )

        # Gemini Flash (균형잡힌 성능)
        self.models["gemini-flash"] = ModelConfig(
            name="gemini-flash",
            provider="google",
            model_id="gemini-3.5-flash-lite",
            temperature=0.1,
            max_tokens=2000,
            cost_per_token=0.0002,
            speed_rating=8.0,
            quality_rating=8.0,
            capabilities=[
                TaskType.GENERATION,
                TaskType.VERIFICATION,
                TaskType.RESEARCH,
                TaskType.CREATIVE,
            ],
        )


    def _load_openrouter_models(self):
        """OpenRouter API에서 무료 모델들을 동적으로 로드 (선택적)."""
        try:
            openrouter_models = self._fetch_openrouter_models()
            for model_data in openrouter_models:
                if self._is_free_model(model_data):
                    model_name = self._generate_model_name(model_data)
                    model_config = self._create_model_config(model_data, model_name)
                    self.models[model_name] = model_config
                    logger.debug(f"Loaded OpenRouter model: {model_name} ({model_data['id']})")
        except Exception as e:
            logger.warning(f"Failed to load OpenRouter models: {e}")
            # 예외를 raise하지 않음 - Gemini만 사용하도록 함


    def _fetch_openrouter_models(self):
        """OpenRouter API에서 모델 목록을 가져옴."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)

        if response.status_code == 200:
            data = _parse_openrouter_json_response(response, "model list fetch")
            return data.get("data", [])
        else:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")


    def _is_free_model(self, model_data):
        """모델이 무료인지 확인."""
        model_id = model_data.get("id", "").lower()

        # 사용 중단된 모델 필터링
        deprecated_models = [
            "kwaipilot/kat-coder-pro:free",
            "mistralai/mistral-7b-instruct:free",
            "qwen/qwen3-4b:free",
            "google/gemma-3n-e2b-it:free",
            "google/gemma-3-4b-it:free",
            "meta-llama/llama-3.2-3b-instruct:free",
        ]
        if any(deprecated in model_id for deprecated in deprecated_models):
            logger.debug(f"Skipping deprecated model: {model_id}")
            return False

        pricing = model_data.get("pricing", {})
        prompt_price = pricing.get("prompt", "0")
        completion_price = pricing.get("completion", "0")

        # 무료 모델 조건: prompt와 completion 가격이 모두 0이거나 매우 낮음
        try:
            prompt_cost = float(prompt_price) if prompt_price else 0
            completion_cost = float(completion_price) if completion_price else 0
            return prompt_cost == 0 and completion_cost == 0
        except (ValueError, TypeError):
            return False


    def _generate_model_name(self, model_data):
        """모델 데이터에서 고유한 이름 생성."""
        model_id = model_data.get("id", "")
        # "provider/model-name:tag" 형식을 "provider-model-name-tag"로 변환
        name = model_id.replace("/", "-").replace(":", "-").replace("_", "-")
        return name


    def _create_model_config(self, model_data, model_name):
        """모델 데이터에서 ModelConfig 생성."""
        model_id = model_data.get("id", "")
        context_length = model_data.get("context_length", 4000)

        # 모델별 기본 설정
        capabilities = self._determine_capabilities(model_id)
        speed_rating = self._estimate_speed_rating(model_id)
        quality_rating = self._estimate_quality_rating(model_id)

        return ModelConfig(
            name=model_name,
            provider="openrouter",
            model_id=model_id,
            temperature=0.1,
            max_tokens=min(context_length, 2000),
            cost_per_token=0.0,  # 무료 모델
            speed_rating=speed_rating,
            quality_rating=quality_rating,
            capabilities=capabilities,
        )


    def _determine_capabilities(self, model_id):
        """모델 ID를 기반으로 capabilities 결정."""
        capabilities = [TaskType.GENERATION]  # 기본

        # 모델명에 따른 capabilities 추가
        if any(keyword in model_id.lower() for keyword in ["reasoning", "reason", "think"]):
            capabilities.extend([TaskType.DEEP_REASONING, TaskType.ANALYSIS])

        if any(keyword in model_id.lower() for keyword in ["code", "coder", "programming"]):
            capabilities.extend([TaskType.RESEARCH, TaskType.COMPRESSION])

        if any(keyword in model_id.lower() for keyword in ["verify", "check", "validate"]):
            capabilities.extend([TaskType.VERIFICATION])

        if any(keyword in model_id.lower() for keyword in ["plan", "planning", "strategy"]):
            capabilities.append(TaskType.PLANNING)

        # 기본적으로 모든 작업 가능
        if not any(
            keyword in model_id.lower() for keyword in ["reasoning", "code", "verify", "plan"]
        ):
            capabilities = [
                TaskType.PLANNING,
                TaskType.DEEP_REASONING,
                TaskType.VERIFICATION,
                TaskType.GENERATION,
                TaskType.COMPRESSION,
                TaskType.RESEARCH,
                TaskType.CREATIVE,
            ]

        return list(set(capabilities))  # 중복 제거


    def _estimate_speed_rating(self, model_id):
        """모델 ID를 기반으로 속도 등급 추정."""
        if any(keyword in model_id.lower() for keyword in ["flash", "fast", "lite", "small"]):
            return 8.0
        elif any(keyword in model_id.lower() for keyword in ["large", "big", "70b", "72b"]):
            return 6.0
        else:
            return 7.0


    def _estimate_quality_rating(self, model_id):
        """모델 ID를 기반으로 품질 등급 추정."""
        if any(keyword in model_id.lower() for keyword in ["70b", "72b", "large", "pro"]):
            return 9.0
        elif any(keyword in model_id.lower() for keyword in ["27b", "medium"]):
            return 7.5
        else:
            return 8.0


    def get_openrouter_fallback_models(self) -> List[str]:
        """OpenRouter 호환 유효 무료 모델 후보 목록 반환 (deprecated 모델 제외)."""
        fallback_candidates = []

        # 1. 동적 로드된 무료 OpenRouter 모델 수집
        free_dynamic_models = [
            m.model_id
            for m in self.models.values()
            if m.provider == "openrouter" and m.cost_per_token == 0.0
        ]

        deprecated = [
            "kwaipilot/kat-coder-pro:free",
            "mistralai/mistral-7b-instruct:free",
            "qwen/qwen3-4b:free",
            "google/gemma-3n-e2b-it:free",
            "google/gemma-3-4b-it:free",
            "meta-llama/llama-3.2-3b-instruct:free",
        ]

        for m_id in free_dynamic_models:
            if not any(dep in m_id.lower() for dep in deprecated) and m_id not in fallback_candidates:
                fallback_candidates.append(m_id)

        # 2. 알려진 검증된 무료 OpenRouter 기본 모델 후보군 (fallback diversity)
        default_candidates = [
            "tencent/hy3:free",
            "qwen/qwen3-coder:free",
            "deepseek/deepseek-r1:free",
        ]
        for m_id in default_candidates:
            if m_id not in fallback_candidates:
                fallback_candidates.append(m_id)

        return fallback_candidates

    def _get_valid_openrouter_model_id(self, model_id: str, model_name: str) -> str:
        """OpenRouter에 실제 존재하는 모델 ID 반환."""
        # 이미 OpenRouter 형식인 경우 (provider/model:tag)
        if "/" in model_id:
            return model_id

        # Gemini 모델 및 미인식 모델 요청 시 사용 가능한 무료 모델 목록 동적 선택
        fallback_models = self.get_openrouter_fallback_models()
        if "gemini" in model_name.lower() or "gemini" in model_id.lower():
            if fallback_models:
                return fallback_models[0]

        logger.warning(
            f"Model ID {model_id} not in OpenRouter format. "
            f"Using fallback for agent service stability: {fallback_models[0]}"
        )
        return fallback_models[0]


    def _get_valid_groq_model_id(self, model_id: str) -> str:
        """Groq에 실제 존재하는 모델 ID 반환."""
        # Groq 프로덕션 모델 목록 (2026-07 기준).
        # mixtral-8x7b-32768은 2025-03-20 단종, llama-3.2-*-preview 계열은
        # preview 슬롯이라 프로덕션에 부적합/단종. llama-3.1-8b-instant와
        # llama-3.3-70b-versatile도 2026-06-17 단종 공지가 나가 gpt-oss로 대체.
        valid_groq_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "groq/compound",
        ]

        # 이미 유효한 Groq 모델 ID인 경우
        if model_id in valid_groq_models:
            return model_id

        # OpenRouter 형식인 경우 Groq 모델 ID 추출
        if "/" in model_id:
            parts = model_id.split("/")
            if len(parts) == 2:
                groq_id = parts[1].split(":")[0]  # 태그 제거
                for valid in valid_groq_models:
                    if valid.split("/")[-1] == groq_id:
                        return valid

        # Provider-prefixed IDs (e.g., "openai/gpt-oss-20b") are invalid for the
        # Groq API. Strip the prefix and validate the bare model name.
        if "/" in model_id:
            bare_id = model_id.split("/", 1)[1].split(":")[0]
            if bare_id in valid_groq_models:
                return bare_id

        # 최소 Fallback 정책: LLM 모델 요청 실패 시에만 fallback 사용
        # Fallback은 Agent 서비스 안정성을 위해 필수적이지만, 명확한 로깅과 함께 최소한으로만 사용됩니다.
        logger.warning(
            f"Model ID {model_id} not a valid Groq model. "
            f"Using minimal fallback for agent service stability: openai/gpt-oss-20b"
        )
        return "openai/gpt-oss-20b"


    def _load_groq_models(self):
        """Groq 모델 로딩."""
        # Groq 프로덕션 모델만 사용 (2026-07 기준, console.groq.com/docs/models).
        # 이전에 쓰던 llama-3.1-8b-instant/llama-3.3-70b-versatile은 단종 공지가
        # 나갔고 mixtral-8x7b-32768은 이미 오래전에 단종되어 gpt-oss로 교체.
        groq_models = [
            {
                "name": "openai/gpt-oss-20b",
                "model_id": "openai/gpt-oss-20b",
                "speed_rating": 9.5,
                "quality_rating": 8.0,
                "capabilities": [
                    TaskType.GENERATION,
                    TaskType.RESEARCH,
                    TaskType.ANALYSIS,
                    TaskType.CREATIVE,
                ],
                # Observed org-level Groq free-tier TPM limit for this model
                # is 8000 (#1339: a 10238-token request got a deterministic
                # 413 here). Lets the cascade skip it for oversized requests
                # instead of retrying into the same 413.
                "context_limit_tokens": 8000,
            },
            {
                "name": "openai/gpt-oss-120b",
                "model_id": "openai/gpt-oss-120b",
                "speed_rating": 8.5,
                "quality_rating": 9.0,
                "capabilities": [
                    TaskType.GENERATION,
                    TaskType.RESEARCH,
                    TaskType.ANALYSIS,
                    TaskType.PLANNING,
                    TaskType.DEEP_REASONING,
                    TaskType.CREATIVE,
                ],
            },
            {
                # Groq의 자체 에이전틱 모델. 웹서치/코드실행 등 내장 툴을 모델이
                # 스스로 호출하므로 다른 groq 모델이 모두 실패했을 때만 쓰는
                # 최후 fallback으로 등록 (gpt-oss 뒤에 위치).
                "name": "groq/compound",
                "model_id": "groq/compound",
                "speed_rating": 7.0,
                "quality_rating": 8.0,
                "capabilities": [
                    TaskType.GENERATION,
                    TaskType.RESEARCH,
                    TaskType.CREATIVE,
                ],
            },
        ]

        for model_data in groq_models:
            self.models[model_data["name"]] = ModelConfig(
                name=model_data["name"],
                provider="groq",
                model_id=model_data["model_id"],
                temperature=0.1,
                max_tokens=2000,
                cost_per_token=0.0,  # Groq는 무료 티어 제공
                speed_rating=model_data["speed_rating"],
                quality_rating=model_data["quality_rating"],
                capabilities=model_data["capabilities"],
                context_limit_tokens=model_data.get("context_limit_tokens"),
            )
            logger.debug(f"Loaded Groq model: {model_data['name']} ({model_data['model_id']})")


    def _load_openai_models(self):
        """OpenAI/GPT 모델 로딩."""
        # 주요 GPT 모델들
        gpt_models = [
            {
                "name": "gpt-5-mini",
                "model_id": "gpt-5-mini",
                "speed_rating": 8.0,
                "quality_rating": 8.5,
                "capabilities": [
                    TaskType.GENERATION,
                    TaskType.VERIFICATION,
                    TaskType.RESEARCH,
                ],
            },
            {
                "name": "gpt-5-nano",
                "model_id": "gpt-5-nano",
                "speed_rating": 7.0,
                "quality_rating": 9.5,
                "capabilities": [
                    TaskType.DEEP_REASONING,
                    TaskType.ANALYSIS,
                    TaskType.SYNTHESIS,
                ],
            },
            {
                "name": "gpt-4o-mini",
                "model_id": "gpt-4o-mini",
                "speed_rating": 9.0,
                "quality_rating": 7.0,
                "capabilities": [
                    TaskType.PLANNING,
                    TaskType.COMPRESSION,
                    TaskType.RESEARCH,
                ],
            },
        ]

        for model_data in gpt_models:
            self.models[model_data["name"]] = ModelConfig(
                name=model_data["name"],
                provider="openai",
                model_id=model_data["model_id"],
                temperature=0.1,
                max_tokens=2000,
                cost_per_token=0.0001,  # GPT는 유료
                speed_rating=model_data["speed_rating"],
                quality_rating=model_data["quality_rating"],
                capabilities=model_data["capabilities"],
            )
            logger.debug(f"Loaded OpenAI/GPT model: {model_data['name']} ({model_data['model_id']})")


    def _load_nvidia_models(self):
        """NVIDIA NIM 모델 로딩."""
        # 주의: 동일 model_id를 별명으로 중복 등록하면 provider cascade가
        # 같은 모델을 연달아 호출해 429를 자초하므로 단일 항목만 유지한다.
        # z-ai/glm-5.2는 2026-08-21 NVIDIA NIM에서 EOL(410 Gone)되어 카탈로그에서
        # 완전히 제거됨 (https://integrate.api.nvidia.com/v1/models, 2026-08-22 확인).
        nvidia_models = [
            {
                "name": "nvidia/nemotron-3-ultra-550b-a55b",
                "model_id": "nvidia/nemotron-3-ultra-550b-a55b",
                "speed_rating": 8.5,
                "quality_rating": 9.0,
                "capabilities": [
                    TaskType.PLANNING,
                    TaskType.DEEP_REASONING,
                    TaskType.VERIFICATION,
                    TaskType.GENERATION,
                    TaskType.COMPRESSION,
                    TaskType.RESEARCH,
                    TaskType.ANALYSIS,
                    TaskType.SYNTHESIS,
                    TaskType.CREATIVE,
                ],
            },
        ]

        for model_data in nvidia_models:
            self.models[model_data["name"]] = ModelConfig(
                name=model_data["name"],
                provider="nvidia",
                model_id=model_data["model_id"],
                temperature=0.2,
                # 파일 쓰기 tool call은 인자가 길어 4000이면 JSON이 잘린다
                max_tokens=16384,
                cost_per_token=0.0,
                speed_rating=model_data["speed_rating"],
                quality_rating=model_data["quality_rating"],
                capabilities=model_data["capabilities"],
            )
            logger.debug(f"Loaded NVIDIA NIM model: {model_data['name']} ({model_data['model_id']})")


    def _load_cerebras_models(self):
        """Cerebras 모델 로딩 (api.cerebras.ai 직접 API, OpenAI 호환).

        gemma-4-31b는 2026-07 기준 Cerebras 공개 엔드포인트에서 preview 등급이다
        (zai-glm-4.7과 동급, gpt-oss-120b만 production 등급). Cerebras 자체 문서가
        "preview 모델은 평가 목적으로만 사용하고 프로덕션에 쓰지 말 것"이라 명시하니,
        rate limit이 낮거나(무료 티어 5 req/min) 예고 없이 내려갈 수 있음을 감안할 것.
        멀티모달(텍스트+이미지) 지원이 필요해 프로덕션 모델 대신 선택된 상태.
        inference-docs.cerebras.ai/models/overview, /models/gemma-4-31b 참고.
        """
        cerebras_models = [
            {
                "name": "cerebras/gemma-4-31b",
                "model_id": "gemma-4-31b",
                "speed_rating": 8.5,  # ~1850 tok/s (gpt-oss-120b의 ~3000 tok/s보다 낮음)
                "quality_rating": 8.0,
                "capabilities": [
                    TaskType.GENERATION,
                    TaskType.RESEARCH,
                    TaskType.ANALYSIS,
                    TaskType.PLANNING,
                    TaskType.DEEP_REASONING,
                ],
            },
        ]

        for model_data in cerebras_models:
            self.models[model_data["name"]] = ModelConfig(
                name=model_data["name"],
                provider="cerebras",
                model_id=model_data["model_id"],
                temperature=0.2,
                max_tokens=4000,
                cost_per_token=0.0,
                speed_rating=model_data["speed_rating"],
                quality_rating=model_data["quality_rating"],
                capabilities=model_data["capabilities"],
            )
            logger.debug(f"Loaded Cerebras model: {model_data['name']} ({model_data['model_id']})")


    def refresh_openrouter_models(self):
        """OpenRouter 모델 목록을 새로고침."""
        logger.info("Refreshing OpenRouter models...")
        # 기존 OpenRouter 모델들 제거
        openrouter_models = [
            name for name, config in self.models.items() if config.provider == "openrouter"
        ]
        for model_name in openrouter_models:
            del self.models[model_name]

        # 새로 로드
        self._load_openrouter_models()
        logger.info(
            f"Refreshed OpenRouter models: {len([name for name, config in self.models.items() if config.provider == 'openrouter'])} models loaded"
        )


    def _initialize_clients(self):
        """모델 클라이언트 초기화."""
        try:
            if genai is not None:
                # llm_config.api_key tracks LLM_PROVIDER's key (e.g. NVIDIA_API_KEY when
                # LLM_PROVIDER=nvidia), not necessarily Google's -- the Gemini SDK needs
                # an actual Google key regardless of which provider is primary (#1517-class bug).
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or self.llm_config.api_key)
            else:
                logger.warning(
                    "google-generativeai library not installed. Install with: pip install google-generativeai"
                )

            for model_name, model_config in self.models.items():
                if model_config.provider == "google":
                    if genai is None:
                        continue
                    # Google Generative AI 클라이언트
                    self.model_clients[model_name] = genai.GenerativeModel(model_config.model_id)

                    # LangChain 클라이언트 (선택적)
                    # See SAFETY_SETTINGS_BLOCK_NONE above: safety_settings is intentionally
                    # not passed here, ChatGoogleGenerativeAI doesn't accept it directly.
                    if ChatGoogleGenerativeAI is not None:
                        self.model_clients[f"{model_name}_langchain"] = ChatGoogleGenerativeAI(
                            model=model_config.model_id,
                            temperature=model_config.temperature,
                            max_tokens=model_config.max_tokens,
                            google_api_key=self.llm_config.api_key,
                        )

                elif model_config.provider == "openrouter":
                    # OpenRouter 클라이언트는 HTTP 요청으로 직접 처리
                    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
                    if not openrouter_api_key:
                        raise ValueError(f"OpenRouter API key not found for {model_name}")
                    # OpenRouter는 HTTP 요청으로 직접 처리하므로 클라이언트 저장하지 않음
                    logger.debug(f"OpenRouter model {model_name} configured for HTTP requests")

                elif model_config.provider == "groq":
                    # Groq 클라이언트 초기화
                    try:
                        from groq import Groq

                        groq_api_key = os.getenv("GROQ_API_KEY")
                        if not groq_api_key:
                            raise ValueError(f"GROQ_API_KEY not found for {model_name}")
                        self.model_clients[model_name] = Groq(api_key=groq_api_key)
                        logger.debug(f"Groq model {model_name} configured")
                    except ImportError:
                        logger.warning("groq library not installed. Install with: pip install groq")
                    except Exception as e:
                        logger.warning(f"Failed to initialize Groq client for {model_name}: {e}")

                elif model_config.provider == "openai":
                    # OpenAI 클라이언트 초기화
                    try:
                        from openai import OpenAI

                        openai_api_key = os.getenv("OPENAI_API_KEY")
                        if not openai_api_key:
                            raise ValueError(f"OPENAI_API_KEY not found for {model_name}")
                        self.model_clients[model_name] = OpenAI(api_key=openai_api_key)
                        logger.debug(f"OpenAI/GPT model {model_name} configured")
                    except ImportError:
                        logger.warning(
                            "openai library not installed. Install with: pip install openai"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to initialize OpenAI client for {model_name}: {e}")

                elif model_config.provider == "nvidia":
                    # NVIDIA NIM 클라이언트 초기화
                    try:
                        from openai import OpenAI

                        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
                        if not nvidia_api_key:
                            raise ValueError(f"NVIDIA_API_KEY not found for {model_name}")
                        self.model_clients[model_name] = OpenAI(
                            api_key=nvidia_api_key,
                            base_url="https://integrate.api.nvidia.com/v1",
                            timeout=180.0,
                            max_retries=1,
                        )
                        logger.debug(f"NVIDIA NIM model {model_name} configured")
                    except ImportError:
                        logger.warning(
                            "openai library not installed. Install with: pip install openai"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to initialize NVIDIA NIM client for {model_name}: {e}")

                elif model_config.provider == "cerebras":
                    # Cerebras 클라이언트 초기화 (OpenAI 호환 API, base_url만 다름)
                    try:
                        from openai import OpenAI

                        cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
                        if not cerebras_api_key:
                            raise ValueError(f"CEREBRAS_API_KEY not found for {model_name}")
                        self.model_clients[model_name] = OpenAI(
                            api_key=cerebras_api_key,
                            base_url="https://api.cerebras.ai/v1",
                        )
                        logger.debug(f"Cerebras model {model_name} configured")
                    except ImportError:
                        logger.warning(
                            "openai library not installed. Install with: pip install openai"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to initialize Cerebras client for {model_name}: {e}")

            logger.info("Model clients initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize model clients: {e}")
            raise

