"""Tests for OpenAI model configuration and GPT-6 lineup integration."""

from src.core.config import ALLOWED_MINI_MODELS
from src.core.llm_manager.model_registry import ModelRegistryMixin
from src.core.llm_manager.types import TaskType


class _Registry(ModelRegistryMixin):
    def __init__(self):
        self.models = {}


def test_openai_gpt6_lineup_and_legacy_compatibility():
    registry = _Registry()
    registry._load_openai_models()

    # 1. 최신 GPT-6 계열 모델 등록 확인
    assert "gpt-6-luna" in registry.models
    assert "gpt-6-sol" in registry.models
    assert "gpt-6.1-sol" in registry.models

    # 2. 기존 레거시 / 경량 호환 모델 누락 없음 확인
    assert "gpt-4o-mini" in registry.models
    assert "gpt-5-mini" in registry.models
    assert "gpt-5-nano" in registry.models

    # 3. 비용 검증 (Sol은 Luna 대비 대폭 높은 cost_per_token 책정)
    luna = registry.models["gpt-6-luna"]
    sol = registry.models["gpt-6-sol"]
    assert sol.cost_per_token >= luna.cost_per_token * 50
    assert luna.cost_per_token <= 0.0001

    # 4. 허용된 mini 모델에 gpt-6-luna 포함 확인
    assert "gpt-6-luna" in ALLOWED_MINI_MODELS
    assert "gpt-4o-mini" in ALLOWED_MINI_MODELS
