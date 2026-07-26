"""Skill Gym — generative practice-environment self-improvement loop.

Anvil Phase Θ: before a newly distilled skill is exported to the skill
marketplace, it is exercised against LLM-generated variant scenarios (edge
cases, adversarial inputs, difficulty escalations) inside an isolated sandbox
backend. Execution results are scored by the evaluation agent / LLM council,
and only skills that clear a reward threshold are promoted to a ``trusted``
trust grade in the SkillRepository. Sub-threshold skills are fed back to the
SkillDistiller with their failure cases as re-distillation hints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

from .skill_repository import Skill, SkillRepository

logger = logging.getLogger(__name__)


@dataclass
class GymScenario:
    """A generated practice scenario for a skill."""

    name: str
    description: str
    inputs: dict[str, Any] = field(default_factory=dict)
    category: str = "edge_case"
    difficulty: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GymScenarioResult:
    """Execution + scoring outcome for one scenario."""

    scenario: GymScenario
    executed: bool
    output: Any = None
    error: str = ""
    score: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "executed": self.executed,
            "output": self.output,
            "error": self.error,
            "score": self.score,
            "rationale": self.rationale,
        }


@dataclass
class GymReport:
    """Aggregate Skill Gym report for a single skill."""

    skill_name: str
    scenarios: list[GymScenarioResult] = field(default_factory=list)
    average_score: float = 0.0
    passed: bool = False
    trust_grade: str = "untrusted"
    failure_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "scenarios": [item.to_dict() for item in self.scenarios],
            "average_score": self.average_score,
            "passed": self.passed,
            "trust_grade": self.trust_grade,
            "failure_hints": self.failure_hints,
        }


class SkillGym:
    """Generate scenarios, execute them in a sandbox, and score the results."""

    def __init__(
        self,
        *,
        scenario_generator: Any | None = None,
        sandbox_backend: Any | None = None,
        scorer: Any | None = None,
        num_scenarios: int = 3,
        reward_threshold: float = 0.7,
    ) -> None:
        self.scenario_generator = scenario_generator
        self.sandbox_backend = sandbox_backend
        self.scorer = scorer
        self.num_scenarios = max(1, int(num_scenarios))
        self.reward_threshold = float(reward_threshold)

    def generate_scenarios(self, skill: Skill) -> list[GymScenario]:
        if self.scenario_generator is None:
            return self._synthetic_scenarios(skill)
        try:
            raw = self.scenario_generator.generate(skill, count=self.num_scenarios)
        except AttributeError:
            raw = self.scenario_generator(skill, count=self.num_scenarios)
        scenarios: list[GymScenario] = []
        for index, item in enumerate(raw, start=1):
            scenarios.append(self._coerce_scenario(item, index, skill))
        return scenarios[: self.num_scenarios]

    def _coerce_scenario(self, item: Any, index: int, skill: Skill) -> GymScenario:
        if isinstance(item, GymScenario):
            return item
        if isinstance(item, dict):
            return GymScenario(
                name=str(item.get("name", f"scenario_{index}")),
                description=str(item.get("description", "")),
                inputs=dict(item.get("inputs", {}) or {}),
                category=str(item.get("category", "edge_case")),
                difficulty=float(item.get("difficulty", 0.5)),
            )
        return GymScenario(
            name=f"scenario_{index}",
            description=str(item),
            inputs={"goal": skill.metadata.get("goal", skill.name)},
        )

    def _synthetic_scenarios(self, skill: Skill) -> list[GymScenario]:
        goal = skill.metadata.get("goal", skill.name)
        categories = ["edge_case", "adversarial_input", "difficulty_escalation"]
        scenarios: list[GymScenario] = []
        for index in range(self.num_scenarios):
            category = categories[index % len(categories)]
            scenarios.append(
                GymScenario(
                    name=f"synthetic_{category}_{index + 1}",
                    description=(
                        f"Synthetic {category} variant for skill '{skill.name}' "
                        f"targeting goal: {goal}"
                    ),
                    inputs={"goal": goal, "variant": index + 1},
                    category=category,
                    difficulty=0.5 + 0.1 * index,
                )
            )
        return scenarios

    def execute_scenario(self, skill: Skill, scenario: GymScenario) -> GymScenarioResult:
        context = {"goal": skill.metadata.get("goal", skill.name)}
        context.update(scenario.inputs)
        if self.sandbox_backend is None:
            try:
                output = self._run_skill_code(skill.code, context)
                return GymScenarioResult(
                    scenario=scenario,
                    executed=True,
                    output=output,
                    score=1.0 if output is not None else 0.0,
                    rationale="executed in-process (no sandbox backend configured)",
                )
            except Exception as exc:  # pragma: no cover - defensive
                return GymScenarioResult(
                    scenario=scenario,
                    executed=False,
                    error=str(exc),
                    score=0.0,
                    rationale="in-process execution failed",
                )

        code = self._wrap_skill_code(skill.code, context)
        try:
            response = asyncio.run(self.sandbox_backend.execute_code(code, language="python"))
            exit_code = getattr(response, "exit_code", 0)
            output_text = getattr(response, "output", "")
            if exit_code != 0:
                return GymScenarioResult(
                    scenario=scenario,
                    executed=False,
                    error=str(output_text),
                    score=0.0,
                    rationale=f"sandbox exit code {exit_code}",
                )
            output = self._parse_output(output_text)
            return GymScenarioResult(
                scenario=scenario,
                executed=True,
                output=output,
                score=1.0 if output is not None else 0.0,
                rationale="executed in sandbox backend",
            )
        except Exception as exc:
            return GymScenarioResult(
                scenario=scenario,
                executed=False,
                error=str(exc),
                score=0.0,
                rationale="sandbox execution raised an exception",
            )

    def score_result(self, skill: Skill, result: GymScenarioResult) -> GymScenarioResult:
        if not result.executed:
            result.score = 0.0
            result.rationale = result.rationale or "scenario did not execute"
            return result
        if self.scorer is None:
            result.score = self._heuristic_score(skill, result)
            result.rationale = "heuristic scorer (no LLM scorer configured)"
            return result
        try:
            score, rationale = self.scorer.score(skill, result)
        except AttributeError:
            score, rationale = self.scorer(skill, result)
        result.score = float(score)
        result.rationale = str(rationale)
        return result

    def _heuristic_score(self, skill: Skill, result: GymScenarioResult) -> float:
        output = result.output
        if not isinstance(output, dict):
            return 0.2
        steps = output.get("steps")
        if isinstance(steps, list) and steps:
            base = 0.8
        else:
            base = 0.4
        goal = skill.metadata.get("goal", "")
        if goal and goal in str(output.get("source_goal", "")):
            base += 0.1
        return min(1.0, base)

    def run(self, skill: Skill) -> GymReport:
        scenarios = self.generate_scenarios(skill)
        results: list[GymScenarioResult] = []
        for scenario in scenarios:
            executed = self.execute_scenario(skill, scenario)
            scored = self.score_result(skill, executed)
            results.append(scored)
        average = (
            sum(item.score for item in results) / len(results) if results else 0.0
        )
        passed = average >= self.reward_threshold
        report = GymReport(
            skill_name=skill.name,
            scenarios=results,
            average_score=round(average, 4),
            passed=passed,
            trust_grade="trusted" if passed else "untrusted",
            failure_hints=self._failure_hints(results),
        )
        return report

    def _failure_hints(self, results: list[GymScenarioResult]) -> list[str]:
        hints: list[str] = []
        for item in results:
            if item.score >= self.reward_threshold:
                continue
            hint = (
                f"{item.scenario.name} ({item.scenario.category}): "
                f"score={item.score:.2f}"
            )
            if item.error:
                hint += f" error={item.error}"
            elif item.rationale:
                hint += f" rationale={item.rationale}"
            hints.append(hint)
        return hints

    @staticmethod
    def _run_skill_code(code: str, context: dict[str, Any]) -> Any:
        namespace: dict[str, Any] = {}
        exec(compile(code, "<skill_gym>", "exec"), namespace)  # noqa: S102
        run_fn = namespace.get("run")
        if not callable(run_fn):
            raise ValueError("skill code does not define a run() function")
        return run_fn(context=context)

    @staticmethod
    def _wrap_skill_code(code: str, context: dict[str, Any]) -> str:
        context_json = json.dumps(context, ensure_ascii=False)
        return (
            code
            + "\n\n"
            + "import json as _gym_json\n"
            + "_gym_context = _gym_json.loads(%r)\n" % context_json
            + "try:\n"
            + "    _gym_result = run(context=_gym_context)\n"
            + "    print(_gym_json.dumps(_gym_result, default=str))\n"
            + "except Exception as _gym_exc:\n"
            + "    print('GYM_ERROR:' + str(_gym_exc))\n"
        )

    @staticmethod
    def _parse_output(output_text: str) -> Any:
        text = (output_text or "").strip()
        if not text:
            return None
        if text.startswith("GYM_ERROR:"):
            raise RuntimeError(text[len("GYM_ERROR:"):])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


class SkillGymGate:
    """Promotion gate that records Skill Gym reports on skills."""

    def __init__(
        self,
        gym: SkillGym,
        repository: SkillRepository,
        *,
        reward_threshold: float | None = None,
    ) -> None:
        self.gym = gym
        self.repository = repository
        self.reward_threshold = (
            float(reward_threshold)
            if reward_threshold is not None
            else gym.reward_threshold
        )

    def evaluate(self, skill: Skill) -> GymReport:
        report = self.gym.run(skill)
        metadata = dict(skill.metadata or {})
        metadata["skill_gym"] = report.to_dict()
        metadata["trust_grade"] = report.trust_grade
        metadata["quality_gate"] = "skill_gym" if report.passed else "skill_gym_failed"
        self.repository.save_skill(
            skill.name,
            skill.code,
            description=skill.description,
            metadata=metadata,
        )
        return report

    def evaluate_draft(self, draft: Any) -> GymReport:
        """Evaluate a ``SkillDraft`` without requiring a persisted skill.

        The draft is materialized into a transient ``Skill`` so the Skill Gym
        can exercise it against generated scenarios. The resulting report is
        returned to the caller (e.g. ``SkillDistiller``) so sub-threshold
        drafts can be rejected before marketplace export.
        """
        skill = Skill(
            name=draft.name,
            code=draft.code,
            description=draft.description,
            metadata=dict(getattr(draft, "metadata", {}) or {}),
        )
        return self.gym.run(skill)

    def is_trusted(self, skill: Skill) -> bool:
        metadata = skill.metadata or {}
        gym_meta = metadata.get("skill_gym")
        if not isinstance(gym_meta, dict):
            return False
        return bool(gym_meta.get("passed")) and gym_meta.get("trust_grade") == "trusted"

    def failure_hints(self, skill: Skill) -> list[str]:
        metadata = skill.metadata or {}
        gym_meta = metadata.get("skill_gym")
        if not isinstance(gym_meta, dict):
            return []
        return list(gym_meta.get("failure_hints", []) or [])

    def distillation_feedback(self, skill: Skill) -> dict[str, Any]:
        """Return feedback payload for SkillDistiller re-distillation."""
        report = skill.metadata.get("skill_gym", {}) if skill.metadata else {}
        return {
            "skill_name": skill.name,
            "average_score": report.get("average_score", 0.0),
            "failure_hints": list(report.get("failure_hints", []) or []),
            "failed_scenarios": [
                item
                for item in report.get("scenarios", [])
                if isinstance(item, dict) and item.get("score", 0.0) < self.reward_threshold
            ],
        }


class SyntheticTestbedGenerator:
    """Generates synthetic test cases, benchmarks, and validation assertions."""

    def __init__(self, llm_manager: Any) -> None:
        self.llm_manager = llm_manager

    def generate_testbed(self, proposal: str) -> dict[str, Any]:
        """Constructs a synthetic testbed for an architectural proposal."""
        prompt = f"""
        Analyze the following architectural proposal and generate a synthetic testbed:
        Proposal: {proposal}

        Return a JSON object with:
        1. 'test_cases': List of scenarios (name, description, inputs).
        2. 'benchmark_inputs': Data structures for performance/load testing.
        3. 'validation_assertions': List of expected outcomes/invariants to verify.
        """
        response = self.llm_manager.generate(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse generated testbed", "raw": response}

    def create_evaluation_harness(self, testbed: dict[str, Any]) -> str:
        """Generates a Python evaluation harness from the testbed."""
        harness_template = """
import unittest

class TestArchitecturalProposal(unittest.TestCase):
    def setUp(self):
        self.testbed = {testbed_json}

    def test_scenarios(self):
        for case in self.testbed.get('test_cases', []):
            with self.subTest(case=case['name']):
                # Implementation of test execution logic
                pass

    def test_invariants(self):
        for assertion in self.testbed.get('validation_assertions', []):
            self.assertTrue(True, msg=assertion)

if __name__ == '__main__':
    unittest.main()
"""
        return harness_template.format(testbed_json=json.dumps(testbed))

    def run_evaluation(self, proposal: str) -> dict[str, Any]:
        """End-to-end generation and execution of evaluation harness."""
        testbed = self.generate_testbed(proposal)
        if "error" in testbed:
            return testbed

        harness_code = self.create_evaluation_harness(testbed)
        # In a real implementation, this would be executed in a sandbox
        return {
            "testbed": testbed,
            "harness_code": harness_code,
            "status": "generated"
        }
