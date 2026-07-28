"""Forge Master Subsystem - Agent Operating System Meta-Orchestrator

외부 AI CLI 도구들을 24/7 중앙 관리 및 제어하고,
토큰 최소화, 세션 보존, 적대적 평가(Zero-Trust Audit)를 총괄하는 메타 오케스트레이션 패키지
"""

from .adversarial_evaluator import AdversarialEvaluationResult, AdversarialEvaluator
from .controller import ForgeMasterController
from .router import ForgeMasterRouter, ToolGoalAssignment
from .session_manager import AgentSession, ForgeMasterSessionManager
from .token_minimizer import TokenMinimizer

__all__ = [
    "ForgeMasterController",
    "ForgeMasterRouter",
    "ToolGoalAssignment",
    "ForgeMasterSessionManager",
    "AgentSession",
    "TokenMinimizer",
    "AdversarialEvaluator",
    "AdversarialEvaluationResult",
]
