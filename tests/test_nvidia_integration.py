import asyncio
import os
import sys
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.llm_manager import MultiModelOrchestrator, Provider, TaskType


class TestNvidiaIntegration:
    """NVIDIA NIM provider 및 nvidia/nemotron-3-ultra-550b-a55b 모델 통합 검증 테스트."""

    @pytest.fixture
    def mock_env_vars(self):
        """NVIDIA 관련 환경 변수 모킹 및 타 API 키 간섭 제거."""
        from src.core.researcher_config import load_config_from_env
        
        # 백업
        orig_or = os.environ.get("OPENROUTER_API_KEY")
        orig_openai = os.environ.get("OPENAI_API_KEY")
        orig_groq = os.environ.get("GROQ_API_KEY")
        
        # 패치 환경 설정
        env_dict = {
            "NVIDIA_API_KEY": "nvapi-test_key",
            "LLM_PROVIDER": "nvidia",
            "LLM_MODEL": "nvidia/nemotron-3-ultra-550b-a55b",
            "OPENROUTER_API_KEY": orig_or or "or-test_key",
            "OPENAI_API_KEY": "openai-test_key",
            "GROQ_API_KEY": "groq-test_key",
        }
        
        with patch.dict(os.environ, env_dict):
            load_config_from_env()
            yield

    def test_nvidia_models_loaded(self, mock_env_vars):
        """NVIDIA NIM 모델이 로드되었는지 확인."""
        orchestrator = MultiModelOrchestrator()
        
        # 모델 목록에 등록되었는지 확인
        assert "nvidia/nemotron-3-ultra-550b-a55b" in orchestrator.models
        
        config = orchestrator.models["nvidia/nemotron-3-ultra-550b-a55b"]
        assert config.provider == "nvidia"
        assert config.model_id == "nvidia/nemotron-3-ultra-550b-a55b"
        assert TaskType.DEEP_REASONING in config.capabilities

    @pytest.mark.asyncio
    async def test_nvidia_client_initialization(self, mock_env_vars):
        """NVIDIA NIM용 OpenAI 클라이언트 초기화 및 API 호출 검증."""
        orchestrator = MultiModelOrchestrator()
        
        # mock 클라이언트 직접 주입
        mock_client = MagicMock()
        orchestrator.model_clients["nvidia/nemotron-3-ultra-550b-a55b"] = mock_client
        orchestrator.model_clients["nemotron-3-ultra-550b-a55b"] = mock_client
        
        # API 응답 모킹
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="NVIDIA GLM Response",
                    tool_calls=[]
                )
            )
        ]
        mock_response.usage = MagicMock(total_tokens=100)
        mock_client.chat.completions.create.return_value = mock_response

        # execute 실행 (포지셔널 바인딩 오류 방지를 위해 명시적 키워드 인자 전달)
        result = await orchestrator.execute_with_model(
            prompt="Hello, GLM!",
            task_type=TaskType.DEEP_REASONING,
            model_name="nvidia/nemotron-3-ultra-550b-a55b",
            use_cascade=False
        )

        assert result.content == "NVIDIA GLM Response"
        assert result.metadata["provider"] == "nvidia"
        assert result.metadata["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
        
        # completions.create가 올바른 파라미터로 호출되었는지 검증
        mock_client.chat.completions.create.assert_called_once_with(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[{"role": "user", "content": "Hello, GLM!"}],
            temperature=0.2,
            max_tokens=16384
        )

    @pytest.mark.asyncio
    async def test_nvidia_model_fallback(self, mock_env_vars):
        """NVIDIA 모델 실행 실패 시 폴백 메커니즘 검증."""
        orchestrator = MultiModelOrchestrator()
        
        # mock_clients에 Mock 클라이언트 주입 (두 앨리어스 모두 등록)
        mock_nvidia_client = MagicMock()
        mock_nvidia_client.chat.completions.create.side_effect = Exception("API Error")
        orchestrator.model_clients["nvidia/nemotron-3-ultra-550b-a55b"] = mock_nvidia_client
        orchestrator.model_clients["nemotron-3-ultra-550b-a55b"] = mock_nvidia_client
        
        # _try_fallback_models 자체를 모킹하여 진입 확인 및 가짜 결과 반환
        mock_fallback = AsyncMock()
        mock_fallback.return_value = (
            {
                "content": "Gemini Fallback Response",
                "confidence": 0.8,
                "metadata": {
                    "provider": "google",
                    "model": "gemini-flash"
                }
            },
            "gemini-flash"
        )
        orchestrator._try_fallback_models = mock_fallback
        
        # execute 실행 (nvidia/nemotron-3-ultra-550b-a55b 호출 실패 -> fallback 호출 유도)
        result = await orchestrator.execute_with_model(
            prompt="Try fallback",
            task_type=TaskType.GENERATION,
            model_name="nvidia/nemotron-3-ultra-550b-a55b",
            use_cascade=False
        )
        
        assert result.content == "Gemini Fallback Response"
        assert result.model_used == "gemini-flash"
        
        # _try_fallback_models가 올바른 skip_providers 리스트와 함께 호출되었는지 검증
        mock_fallback.assert_called_once()
        call_args = mock_fallback.call_args[1]
        assert "nvidia" in call_args["skip_providers"]
