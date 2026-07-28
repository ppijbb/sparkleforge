"""Forge Master Adversarial Evaluator - Zero-Trust & Extreme Skepticism Engine

외부 CLI 에이전트/도구가 생성한 모든 결과물에 대해 극단적 의심과 적대적 평가(Adversarial Audit)를 수행.
절대 개별 도구의 응답을 맹신하지 않고, 구문/AST 검사, 적대적 반론 쿼리, 결함 탐지를 거쳐
최종 검증을 통과한 결과물만 SparkleForge 시스템에 흡수함.
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AdversarialEvaluationResult:
    """적대적 평가 결과"""

    passed: bool
    skepticism_score: float  # 0.0 ~ 1.0 (높을수록 의심/결함 높음)
    flaws_detected: List[str] = field(default_factory=list)
    adversarial_feedback: str = ""
    verdict: str = "PASSED"  # PASSED, REJECTED_WITH_FEEDBACK, ESCALATE_TO_FALLBACK


class AdversarialEvaluator:
    """적대적 평가기 (Extreme Skepticism & Zero-Trust Gate)"""

    def __init__(self, strictness_level: str = "high"):
        """
        Args:
            strictness_level: 엄격도 수준 ('medium', 'high', 'extreme')
        """
        self.strictness_level = strictness_level

    async def evaluate_output(
        self,
        task_query: str,
        agent_name: str,
        execution_result: Dict[str, Any],
        expected_artifacts: Optional[List[str]] = None,
    ) -> AdversarialEvaluationResult:
        """외부 CLI 에이전트의 결과물에 대해 다층 적대적 검증 수행

        Args:
            task_query: 원본 요청 쿼리
            agent_name: 에이전트 이름 (codex, claude_code 등)
            execution_result: CLI 실행 결과 딕셔너리
            expected_artifacts: 기대되는 산출물 파일/함수 리스트

        Returns:
            AdversarialEvaluationResult
        """
        flaws: List[str] = []
        skepticism_score = 0.0

        # 0. 실행 성공 여부 의심
        if not execution_result.get("success", False):
            flaws.append(f"CLI Execution reported failure or crash: {execution_result.get('error')}")
            return AdversarialEvaluationResult(
                passed=False,
                skepticism_score=1.0,
                flaws_detected=flaws,
                adversarial_feedback="Execution crashed or failed. Trigger fallback immediately.",
                verdict="ESCALATE_TO_FALLBACK",
            )

        response_text = execution_result.get("response", "").strip()

        # 1. 빈 응답 또는 극소량 텍스트 의심 (Lazy / Placeholder Output)
        if not response_text or len(response_text) < 15:
            flaws.append("Output is suspiciously empty or trivial (under 15 chars).")
            skepticism_score += 0.5

        # 2. 더미 파사드/임시 코드 반환 여부 적대적 검사
        lazy_patterns = [
            r"todo:?\s*implement",
            r"pass\s*#\s*todo",
            r"return\s+None\s*#\s*placeholder",
            r"raise\s+NotImplementedError",
            r"\.\.\.\s*#\s*fill in",
        ]
        for pattern in lazy_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                flaws.append(f"Detected lazy placeholder or unhandled stub pattern: '{pattern}'")
                skepticism_score += 0.3

        # 3. 파이썬 코드 블록 구문(AST) 엄격 적대적 검사
        code_blocks = self._extract_code_blocks(response_text)
        for idx, code in enumerate(code_blocks):
            ast_ok, ast_err = self._check_python_syntax(code)
            if not ast_ok:
                flaws.append(f"Code block #{idx + 1} failed AST syntax parsing: {ast_err}")
                skepticism_score += 0.4

        # 4. 반환 신뢰도 수치에 대한 적대적 깎기 (Over-confidence Penalty)
        claimed_confidence = execution_result.get("confidence", 1.0)
        if claimed_confidence > 0.95:
            # 주장하는 신뢰도가 95% 이상이면 오히려 의심 가산
            skepticism_score += 0.1

        # 5. 최종 판정 (Verdict Decision)
        passed = len(flaws) == 0 and skepticism_score < 0.4

        if passed:
            verdict = "PASSED"
            feedback = f"Output survived adversarial scrutiny (skepticism score: {skepticism_score:.2f})."
        else:
            if skepticism_score >= 0.7:
                verdict = "ESCALATE_TO_FALLBACK"
                feedback = f"Critical flaws detected in {agent_name} output. Escalate to fallback tool."
            else:
                verdict = "REJECTED_WITH_FEEDBACK"
                feedback = f"Adversarial audit failed. Detected flaws: {'; '.join(flaws)}"

        logger.info(
            f"Adversarial Evaluation for '{agent_name}': verdict={verdict}, flaws={len(flaws)}, skepticism={skepticism_score:.2f}"
        )

        return AdversarialEvaluationResult(
            passed=passed,
            skepticism_score=round(skepticism_score, 2),
            flaws_detected=flaws,
            adversarial_feedback=feedback,
            verdict=verdict,
        )

    def _extract_code_blocks(self, text: str) -> List[str]:
        """텍스트 내 파이썬 코드 블록 추출"""
        blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
        if not blocks:
            # 파이썬 명시가 없는 백틱 블록 시도
            blocks = re.findall(r"```(.*?)```", text, re.DOTALL)
        return [b.strip() for b in blocks if b.strip()]

    def _check_python_syntax(self, code: str) -> tuple[bool, Optional[str]]:
        """AST 구문 파싱 검사"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
