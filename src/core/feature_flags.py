"""
Feature Flags - 모든 기능 기본 활성화

변경 사항:
- 모든 기능이 기본적으로 활성화됨
- 선택적 기능은 작업 중 Human-in-Loop로 처리
- 환경 변수는 비활성화용으로만 사용 (선택 사항)
"""

import os
import logging

logger = logging.getLogger(__name__)


class FeatureFlags:
    """
    기능 플래그 - 모든 기능 기본 활성화
    
    변경 사항:
    - 모든 기능이 기본적으로 활성화됨
    - 선택적 기능은 작업 중 Human-in-Loop로 처리
    - 환경 변수는 비활성화용으로만 사용 (DISABLE_* 형식)
    """
    
    # 모든 기능 기본 활성화 (환경 변수로 비활성화 가능)
    ENABLE_MCP_STABILITY = os.getenv("DISABLE_MCP_STABILITY", "true").lower() != "true"
    ENABLE_GUARDRAILS = os.getenv("DISABLE_GUARDRAILS", "true").lower() != "true"
    ENABLE_AGENT_TOOLS = os.getenv("DISABLE_AGENT_TOOLS", "true").lower() != "true"
    ENABLE_YAML_CONFIG = os.getenv("DISABLE_YAML_CONFIG", "true").lower() != "true"
    ENABLE_MCP_HEALTH_BACKGROUND = os.getenv("DISABLE_MCP_HEALTH_BACKGROUND", "true").lower() != "true"
    
    @classmethod
    def get_all_flags(cls) -> dict:
        """모든 기능 플래그 상태 반환"""
        return {
            "mcp_stability": cls.ENABLE_MCP_STABILITY,
            "guardrails": cls.ENABLE_GUARDRAILS,
            "agent_tools": cls.ENABLE_AGENT_TOOLS,
            "yaml_config": cls.ENABLE_YAML_CONFIG,
            "mcp_health_background": cls.ENABLE_MCP_HEALTH_BACKGROUND,
        }
    
    @classmethod
    def log_status(cls):
        """현재 활성화된 기능 플래그 로깅"""
        flags = cls.get_all_flags()
        active_flags = [name for name, enabled in flags.items() if enabled]
        disabled_flags = [name for name, enabled in flags.items() if not enabled]
        
        if active_flags:
            logger.info(f"✅ Active features (default): {', '.join(active_flags)}")
        if disabled_flags:
            logger.info(f"⚠️ Disabled features (via DISABLE_* env vars): {', '.join(disabled_flags)}")

