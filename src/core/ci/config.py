"""LLM config bootstrap shared by the CI gate agents."""

from __future__ import annotations

import os

# CI 잡은 LLM_PROVIDER/LLM_MODEL과 API 키만 주입하므로, 설정 로더가 요구하는
# 나머지 필수 변수에 안전한 기본값을 채운다. 시크릿 성격의 키에는 기본값을 두지 않는다.
_CONFIG_ENV_DEFAULTS = {
    "LLM_TEMPERATURE": "0.2",
    "LLM_MAX_TOKENS": "8192",
    "BUDGET_LIMIT": "5.0",
    "ENABLE_COST_OPTIMIZATION": "true",
}
_MODEL_ROLE_KEYS = (
    "PLANNING_MODEL",
    "REASONING_MODEL",
    "VERIFICATION_MODEL",
    "GENERATION_MODEL",
    "COMPRESSION_MODEL",
)


def ensure_config_loaded() -> None:
    """MultiModelOrchestrator 생성 전에 전역 LLM 설정을 1회 로드한다."""
    from src.core import researcher_config

    if researcher_config.config is not None:
        return
    for key, value in _CONFIG_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    default_model = os.getenv("LLM_MODEL")
    if default_model:
        for key in _MODEL_ROLE_KEYS:
            os.environ.setdefault(key, default_model)
    researcher_config.load_config_from_env()
