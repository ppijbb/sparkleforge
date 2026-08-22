"""TaskType.CREATIVE must be routable without falling through select_model()'s
whole capability-search maze to the hardcoded "gemini-flash-lite" default --
that fallback crashes with "Model gemini-flash-lite not found" in any
deployment where Gemini isn't loaded (see PR #1226 review). The configured
primary model (nvidia/nemotron-3-ultra-550b-a55b) must declare CREATIVE itself so the
select_model() fast path can return it directly, and at least one other
provider must declare it too so _try_fallback_models() has somewhere to go
if the primary model fails.
"""

from src.core.llm_manager.model_registry import ModelRegistryMixin
from src.core.llm_manager.types import TaskType


class _Registry(ModelRegistryMixin):
    def __init__(self):
        self.models = {}


def test_nvidia_primary_model_declares_creative():
    registry = _Registry()
    registry._load_nvidia_models()

    assert TaskType.CREATIVE in registry.models["nvidia/nemotron-3-ultra-550b-a55b"].capabilities


def test_creative_has_a_fallback_candidate_outside_nvidia():
    registry = _Registry()
    registry._load_nvidia_models()
    registry._load_google_models()

    fallback_providers = {
        config.provider
        for name, config in registry.models.items()
        if name != "nvidia/nemotron-3-ultra-550b-a55b" and TaskType.CREATIVE in config.capabilities
    }

    assert fallback_providers, "CREATIVE needs a fallback candidate outside nvidia"
