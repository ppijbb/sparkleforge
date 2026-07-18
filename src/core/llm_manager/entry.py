"""Public entry points: get_llm_orchestrator singleton, execute_llm_task, CLI-agent support.

Split out of the former monolithic llm_manager.py (issue #582). This is the
module most other files in the repo actually import from -- TaskType and
execute_llm_task alone are imported by ~30 files -- so the module-level
_llm_orchestrator singleton lives here (not in orchestrator.py) to guarantee
every get_llm_orchestrator() call site across the repo keeps sharing the
same one instance after the split.
"""

import logging
from typing import Any, Dict, List

from src.core.llm_manager.orchestrator import MultiModelOrchestrator
from src.core.llm_manager.types import ModelResult, TaskType
from src.core.prompt_refiner_wrapper import refine_llm_call
from src.core.prompt_security import validate_llm_output

logger = logging.getLogger(__name__)

_llm_orchestrator = None


def get_llm_orchestrator() -> "MultiModelOrchestrator":
    """Get or initialize global LLM orchestrator."""
    global _llm_orchestrator
    if _llm_orchestrator is None:
        _llm_orchestrator = MultiModelOrchestrator()
    return _llm_orchestrator



def _build_active_skills_system_block(
    system_message: str | None, prompt: str, kwargs: Dict[str, Any]
) -> str:
    """SkillManager/Selector에서 활성 스킬을 수집해 system_message 하단에 [Active Agent Skills & Rules] 블록을 붙인다.
    USE_PROGRESSIVE_SKILL_DISCLOSURE=true 이면 요약만 주입하고, 상세는 get_skill_instructions 도구로 로드하도록 안내.
    """
    try:
        from src.core.skills_manager import get_skill_manager
        from src.core.skills_selector import get_skill_selector
    except ImportError:
        return system_message or ""

    import os

    sm = get_skill_manager()
    parts: List[str] = []
    use_progressive = os.getenv("USE_PROGRESSIVE_SKILL_DISCLOSURE", "false").lower() == "true"

    # 1) CLI --skills로 지정된 스킬 (우선)
    forced_ids = list(sm.get_forced_skills())
    for skill_id in forced_ids:
        if use_progressive:
            parts.append(sm.get_skill_summary(skill_id))
        else:
            skill = sm.load_skill(skill_id)
            if skill and skill.instructions:
                parts.append(f"[{skill_id}]\n{skill.instructions}")
                if getattr(skill, "examples", None):
                    parts.append(f"Examples:\n{skill.examples}")

    # 2) 글로벌 룰 (.cursorrules, .agentrules, .cursor/rules/*.mdc) - 항상 전체 주입
    for skill in sm.get_global_rules_skills():
        if skill.instructions:
            parts.append(f"[{skill.metadata.skill_id}]\n{skill.instructions}")

    # 3) 현재 prompt/query 기준 선택 스킬 (중복 제외)
    seen_ids = set(forced_ids) | {s.metadata.skill_id for s in sm.get_global_rules_skills()}
    query_for_selector = kwargs.get("user_query") or prompt[:500] if prompt else ""
    if query_for_selector:
        for match in get_skill_selector().select_skills_for_task(query_for_selector, max_skills=3):
            if match.skill_id in seen_ids:
                continue
            seen_ids.add(match.skill_id)
            if use_progressive:
                parts.append(sm.get_skill_summary(match.skill_id))
            else:
                skill = sm.load_skill(match.skill_id)
                if skill and skill.instructions:
                    parts.append(f"[{match.skill_id}]\n{skill.instructions}")
                    if getattr(skill, "examples", None):
                        parts.append(f"Examples:\n{skill.examples}")

    if not parts:
        return system_message or ""

    if use_progressive:
        block = (
            "\n\n---\n\n[Active Agent Skills – Summaries]\n\n"
            + "\n\n".join(parts)
            + "\n\nTo load full instructions for a skill, use the get_skill_instructions tool with skill_id."
        )
    else:
        block = "\n\n---\n\n[Active Agent Skills & Rules]\n\n" + "\n\n".join(parts)
    return (system_message or "") + block



@refine_llm_call
async def execute_llm_task(
    prompt: str,
    task_type: TaskType,
    model_name: str = None,
    system_message: str = None,
    use_ensemble: bool = False,
    agent_name: str | None = None,
    **kwargs,
) -> ModelResult:
    """LLM 작업 실행 (API 모델 + CLI 에이전트 지원)."""
    try:
        system_message = _build_active_skills_system_block(system_message, prompt, kwargs)
        if not agent_name:
            from src.core.agent_security import get_current_agent_name

            agent_name = get_current_agent_name()

        if agent_name:
            from src.core.agent_security import get_agent_security_manager

            _sec = get_agent_security_manager()
            if not _sec.check_rate_limit(agent_name):
                return ModelResult(
                    content="[RATE_LIMIT] LLM call limit exceeded for this agent execution.",
                    model_used="none",
                    execution_time=0.0,
                    confidence=0.0,
                    cost=0.0,
                    metadata={"rate_limited": True, "agent": agent_name},
                )

        # Provider가 opencode면 무조건 OpenCode(Kimi K 2.5)로 라우팅
        from src.core.researcher_config import get_llm_config

        if get_llm_config().provider == "opencode":
            result = await _execute_cli_agent_task(
                prompt, task_type, "open_code", system_message, **kwargs
            )
            ok, final_content = validate_llm_output(result.content or "")
            if not ok:
                return ModelResult(
                    content=final_content,
                    model_used=result.model_used,
                    execution_time=result.execution_time,
                    confidence=0.0,
                    cost=result.cost,
                    metadata={**result.metadata, "output_validated_rejected": True},
                )
            if final_content != (result.content or ""):
                result = ModelResult(
                    content=final_content,
                    model_used=result.model_used,
                    execution_time=result.execution_time,
                    confidence=result.confidence,
                    cost=result.cost,
                    metadata=result.metadata,
                )
            if agent_name:
                from src.core.agent_security import get_agent_security_manager

                _sec_out = get_agent_security_manager()
                out_check = _sec_out.enforce_output(agent_name, result.content or "")
                if out_check.filtered_text != (result.content or ""):
                    result = ModelResult(
                        content=out_check.filtered_text,
                        model_used=result.model_used,
                        execution_time=result.execution_time,
                        confidence=result.confidence if out_check.is_allowed else 0.0,
                        cost=result.cost,
                        metadata={**result.metadata, "agent_security_filtered": True},
                    )
            return result

        # CLI 에이전트 체크 (model_name이 CLI 에이전트인 경우)
        if model_name and _is_cli_agent(model_name):
            result = await _execute_cli_agent_task(
                prompt, task_type, model_name, system_message, **kwargs
            )
            ok, final_content = validate_llm_output(result.content or "")
            if not ok:
                return ModelResult(
                    content=final_content,
                    model_used=result.model_used,
                    execution_time=result.execution_time,
                    confidence=0.0,
                    cost=result.cost,
                    metadata={**result.metadata, "output_validated_rejected": True},
                )
            if final_content != (result.content or ""):
                result = ModelResult(
                    content=final_content,
                    model_used=result.model_used,
                    execution_time=result.execution_time,
                    confidence=result.confidence,
                    cost=result.cost,
                    metadata=result.metadata,
                )
            return result

        # 기존 API 모델 처리
        if use_ensemble:
            result = await get_llm_orchestrator().weighted_ensemble(
                prompt, task_type, model_name, system_message, **kwargs
            )
        else:
            result = await get_llm_orchestrator().execute_with_model(
                prompt, task_type, model_name, system_message, **kwargs
            )
        # Output validation (prompt leakage / sensitive pattern filter)
        ok, final_content = validate_llm_output(result.content or "")
        if not ok:
            return ModelResult(
                content=final_content,
                model_used=result.model_used,
                execution_time=result.execution_time,
                confidence=0.0,
                cost=result.cost,
                metadata={**result.metadata, "output_validated_rejected": True},
            )
        if final_content != (result.content or ""):
            result = ModelResult(
                content=final_content,
                model_used=result.model_used,
                execution_time=result.execution_time,
                confidence=result.confidence,
                cost=result.cost,
                metadata=result.metadata,
            )

        if agent_name:
            from src.core.agent_security import get_agent_security_manager

            _sec_out = get_agent_security_manager()
            out_check = _sec_out.enforce_output(agent_name, result.content or "")
            if out_check.filtered_text != (result.content or ""):
                result = ModelResult(
                    content=out_check.filtered_text,
                    model_used=result.model_used,
                    execution_time=result.execution_time,
                    confidence=result.confidence if out_check.is_allowed else 0.0,
                    cost=result.cost,
                    metadata={**result.metadata, "agent_security_filtered": True},
                )

        return result
    except Exception as e:
        logger.error(f"LLM task execution failed: {e}")
        raise



def get_best_model_for_task(task_type: TaskType) -> str:
    """작업에 최적 모델 반환."""
    return get_llm_orchestrator().get_best_model_for_task(task_type)


def get_model_performance_stats() -> Dict[str, Any]:
    """모델 성능 통계 반환."""
    return get_llm_orchestrator().get_model_performance_stats()


# CLI 에이전트 지원 함수들
def _is_cli_agent(model_name: str) -> bool:
    """모델 이름이 CLI 에이전트인지 확인"""
    cli_agents = {
        "claude_code",
        "open_code",
        "gemini_cli",
        "cline_cli",
        "claudecode",
        "opencode",
        "gemini-cli",
        "cline-cli",
    }
    return model_name.lower() in cli_agents


async def _execute_cli_agent_task(
    prompt: str,
    task_type: TaskType,
    agent_name: str,
    system_message: str = None,
    **kwargs,
) -> ModelResult:
    """CLI 에이전트 작업 실행"""
    from src.core.cli_agents.cli_agent_manager import get_cli_agent_manager

    try:
        cli_manager = get_cli_agent_manager()

        # 시스템 메시지와 프롬프트 결합
        full_query = prompt
        if system_message:
            full_query = f"{system_message}\n\n{prompt}"

        # 작업 유형에 따른 추가 파라미터 설정
        agent_kwargs = kwargs.copy()

        # 작업 유형별 특화 파라미터
        if task_type == TaskType.GENERATION:
            agent_kwargs.setdefault("mode", "generate")
        elif task_type == TaskType.ANALYSIS:
            agent_kwargs.setdefault("mode", "analyze")
        elif task_type == TaskType.RESEARCH:
            agent_kwargs.setdefault("mode", "chat")

        # CLI 에이전트 실행
        result = await cli_manager.execute_with_agent(agent_name, full_query, **agent_kwargs)

        # ModelResult 형식으로 변환 (content, model_used, execution_time, cost)
        meta = result.get("metadata", {})
        exec_time = meta.get("execution_time", 0.0)
        return ModelResult(
            content=result.get("response", ""),
            model_used=f"cli:{agent_name}",
            execution_time=exec_time,
            confidence=result.get("confidence", 0.0),
            cost=0.0,
            metadata={
                "agent_type": "cli",
                "agent_name": agent_name,
                "task_type": task_type.value,
                "execution_time": exec_time,
                **meta,
            },
        )

    except Exception as e:
        logger.error(f"CLI agent execution failed: {agent_name} - {e}")
        return ModelResult(
            content="",
            model_used=f"cli:{agent_name}",
            execution_time=0.0,
            confidence=0.0,
            cost=0.0,
            metadata={"agent_type": "cli", "agent_name": agent_name, "error": str(e)},
        )
