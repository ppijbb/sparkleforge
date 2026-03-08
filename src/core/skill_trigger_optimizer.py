"""Skill Trigger Optimizer - eval/improve 루프로 스킬 description 최적화.

Agent Skills Spec 패턴: 테스트 쿼리로 트리거 여부 평가 후 LLM으로 description 개선 제안.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _get_skills_dir() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    return root / "skills"


class SkillTriggerOptimizer:
    """스킬 description을 반복 최적화하여 트리거 정확도 개선."""

    def __init__(self, skill_manager: Any = None):
        from src.core.skills_manager import get_skill_manager

        self.skill_manager = skill_manager or get_skill_manager()
        self._selector = None

    def _get_selector(self):
        if self._selector is None:
            from src.core.skills_selector import get_skill_selector

            self._selector = get_skill_selector()
        return self._selector

    async def run_trigger_eval(
        self, skill_id: str, test_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """테스트 쿼리에 대해 스킬 트리거 여부 평가.

        test_queries: [{"query": str, "should_trigger": bool}, ...]
        Returns: {"results": [{"query", "should_trigger", "pass", "triggered"}], "summary": {"passed", "total"}}
        """
        selector = self._get_selector()
        results = []
        for item in test_queries:
            query = item.get("query", "")
            should_trigger = item.get("should_trigger", True)
            triggered = False
            try:
                matches = await selector.select_skills_proactively(
                    query, context=None, max_skills=10
                )
                triggered = any(m.skill_id == skill_id for m in matches)
            except Exception as e:
                logger.warning("run_trigger_eval select_skills_proactively failed: %s", e)
            pass_ = triggered == should_trigger
            results.append(
                {
                    "query": query,
                    "should_trigger": should_trigger,
                    "triggered": triggered,
                    "pass": pass_,
                }
            )
        passed = sum(1 for r in results if r["pass"])
        return {
            "results": results,
            "summary": {"passed": passed, "total": len(results)},
        }

    async def improve_description(
        self, skill_id: str, eval_results: Dict[str, Any]
    ) -> str:
        """LLM으로 description 개선 제안."""
        skill = self.skill_manager.load_skill(skill_id)
        if not skill:
            return ""
        current_description = skill.metadata.description or ""
        results = eval_results.get("results", [])
        failed_triggers = [
            r for r in results if r.get("should_trigger") and not r.get("pass")
        ]
        false_triggers = [
            r for r in results if not r.get("should_trigger") and not r.get("pass")
        ]

        prompt = f"""You are optimizing a skill description for trigger accuracy.
Skill ID: {skill_id}
Current description: {current_description}

Eval results: {len(results)} total, {eval_results.get('summary', {}).get('passed', 0)} passed.

Queries that SHOULD have triggered but did NOT (improve to catch these):
{json.dumps([r.get('query') for r in failed_triggers[:10]], ensure_ascii=False, indent=2)}

Queries that should NOT have triggered but did (improve to avoid these):
{json.dumps([r.get('query') for r in false_triggers[:10]], ensure_ascii=False, indent=2)}

Output a single improved description (one paragraph, max 1024 chars) that:
- Triggers for the first list and avoids triggering for the second.
- Uses imperative phrasing ("Use this skill when...").
- Stays concise. Output ONLY the new description, no preamble."""

        try:
            from src.core.llm_manager import TaskType, execute_llm_task

            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.ANALYSIS,
                model_name=None,
                system_message="You output only the improved skill description text.",
            )
            text = (result.content or "").strip()
            if len(text) > 1024:
                text = text[:1021] + "..."
            return text
        except Exception as e:
            logger.warning("improve_description LLM failed: %s", e)
            return current_description

    async def optimize_loop(
        self, skill_id: str, test_queries: List[Dict[str, Any]], iterations: int = 3
    ) -> str:
        """eval -> improve 반복 루프. 최종 description 반환."""
        best_description = None
        skill = self.skill_manager.load_skill(skill_id)
        if skill:
            best_description = skill.metadata.description or ""

        for i in range(iterations):
            eval_results = await self.run_trigger_eval(skill_id, test_queries)
            if eval_results["summary"]["passed"] == eval_results["summary"]["total"]:
                logger.info(
                    "SkillTriggerOptimizer: %s passed all at iteration %d",
                    skill_id,
                    i + 1,
                )
                break
            improved = await self.improve_description(skill_id, eval_results)
            if not improved or improved == best_description:
                break
            best_description = improved
            self._apply_description_to_skill(skill_id, improved)
            logger.info(
                "SkillTriggerOptimizer: %s iteration %d applied new description",
                skill_id,
                i + 1,
            )

        return best_description or ""

    def _apply_description_to_skill(self, skill_id: str, new_description: str) -> None:
        """SKILL.md에 새 description 반영 (frontmatter 또는 본문)."""
        skills_dir = _get_skills_dir()
        skill_path = skills_dir / skill_id / "SKILL.md"
        if not skill_path.exists():
            logger.warning("SKILL.md not found for skill %s", skill_id)
            return
        try:
            content = skill_path.read_text(encoding="utf-8")
            if content.strip().startswith("---"):
                parts = content.strip().split("---", 2)
                if len(parts) >= 3:
                    import yaml

                    try:
                        fm = yaml.safe_load(parts[1].strip()) or {}
                        fm["description"] = new_description
                        import io

                        buf = io.StringIO()
                        yaml.dump(fm, buf, default_flow_style=False, allow_unicode=True)
                        new_content = "---\n" + buf.getvalue().strip() + "\n---\n" + parts[2]
                        skill_path.write_text(new_content, encoding="utf-8")
                    except Exception as e:
                        logger.warning("_apply_description_to_skill YAML failed: %s", e)
            else:
                logger.debug("SKILL.md has no frontmatter, skip apply")
        except Exception as e:
            logger.warning("_apply_description_to_skill failed: %s", e)


def get_skill_trigger_optimizer(skill_manager: Any = None) -> SkillTriggerOptimizer:
    """전역 SkillTriggerOptimizer 인스턴스 반환."""
    return SkillTriggerOptimizer(skill_manager=skill_manager)
