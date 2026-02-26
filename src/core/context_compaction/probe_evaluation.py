"""Probe-Based Evaluation for Context Compression

Agent-Skills-for-Context-Engineering: 압축 후 기능적 품질 검증.
Recall, Artifact, Continuation, Decision 프로브로 보존된 정보를 측정합니다.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """단일 프로브 유형별 평가 결과."""

    probe_type: str  # recall | artifact | continuation | decision
    question: str
    answer: str
    score: float  # 0.0 ~ 1.0
    expected_keywords: List[str] = field(default_factory=list)


@dataclass
class ProbeEvaluationResult:
    """전체 프로브 평가 결과."""

    recall_score: float
    artifact_score: float
    continuation_score: float
    decision_score: float
    overall_score: float
    probe_results: List[ProbeResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


# 프로브 유형별 기본 질문 (원본 컨텍스트에서 추출할 수 없을 때 사용)
DEFAULT_PROBE_QUESTIONS = {
    "recall": "What was the main problem or error that was being addressed? What was the original error message or key fact?",
    "artifact": "Which files or artifacts have been modified, created, or read? List any file paths or key artifacts mentioned.",
    "continuation": "What should we do next? What are the immediate next steps?",
    "decision": "What important decisions were made? What was decided about the approach or solution?",
}


def _extract_text_from_messages(messages: List[Dict[str, Any]], max_chars: int = 15000) -> str:
    """메시지 리스트에서 텍스트만 추출하여 하나의 문자열로."""
    parts = []
    total = 0
    for msg in messages:
        if total >= max_chars:
            break
        content = msg.get("content", "")
        if isinstance(content, str):
            part = content[: max_chars - total]
            parts.append(part)
            total += len(part)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    part = text[: max_chars - total]
                    parts.append(part)
                    total += len(part)
    return "\n\n".join(parts)


def _score_answer_by_keywords(answer: str, expected_keywords: List[str]) -> float:
    """답변에 기대 키워드가 포함된 비율로 0~1 점수."""
    if not expected_keywords:
        return 0.5  # 키워드 없으면 중립
    answer_lower = (answer or "").lower()
    found = sum(1 for k in expected_keywords if k.lower() in answer_lower)
    return min(1.0, found / len(expected_keywords) + 0.2)  # 일부만 있어도 부분 점수


async def run_probe_evaluation(
    compressed_messages: List[Dict[str, Any]],
    original_messages: List[Dict[str, Any]] | None,
    llm_client: Any | None,
    probe_questions: Dict[str, str] | None = None,
) -> ProbeEvaluationResult:
    """압축된 컨텍스트에 대해 프로브 기반 평가를 실행합니다.

    Args:
        compressed_messages: 압축 후 메시지 리스트 (요약 + 보호된 메시지)
        original_messages: 압축 전 원본 메시지 (프로브 기대 답 추출용, None이면 기본 질문만 사용)
        llm_client: LLM 클라이언트 (프로브 답변 생성 및 채점용)
        probe_questions: 프로브 유형별 질문 오버라이드

    Returns:
        ProbeEvaluationResult
    """
    questions = probe_questions or DEFAULT_PROBE_QUESTIONS.copy()
    context_text = _extract_text_from_messages(compressed_messages)
    probe_results: List[ProbeResult] = []
    scores: Dict[str, float] = {}

    for probe_type, question in questions.items():
        answer = ""
        score = 0.0
        expected_keywords: List[str] = []

        if llm_client and context_text:
            try:
                prompt = f"""Based only on the following context, answer this question in one or two short paragraphs. Be precise; if the context does not contain the information, say "Not mentioned in context."

Context:
{context_text[:8000]}

Question: {question}

Answer:"""
                if hasattr(llm_client, "generate"):
                    answer = await llm_client.generate(prompt)
                else:
                    answer = "Not mentioned in context."
            except Exception as e:
                logger.warning("Probe evaluation LLM call failed: %s", e)
                answer = ""

        if not answer:
            answer = "Not mentioned in context."
            score = 0.0
        else:
            # 원본에서 기대 키워드 추출 (간단 휴리스틱: 원본 텍스트에서 자주 나온 단어)
            if original_messages:
                orig_text = _extract_text_from_messages(original_messages, max_chars=3000)
                words = [w.strip(".,;:") for w in orig_text.split() if len(w.strip(".,;:")) > 4]
                from collections import Counter
                top = Counter(words).most_common(5)
                expected_keywords = [w for w, _ in top]
            score = _score_answer_by_keywords(answer, expected_keywords) if expected_keywords else 0.5
            # "Not mentioned"면 낮은 점수
            if "not mentioned" in answer.lower() or "not in context" in answer.lower():
                score = min(score, 0.3)

        scores[probe_type] = score
        probe_results.append(
            ProbeResult(
                probe_type=probe_type,
                question=question,
                answer=answer[:500],
                score=score,
                expected_keywords=expected_keywords,
            )
        )

    recall_score = scores.get("recall", 0.5)
    artifact_score = scores.get("artifact", 0.5)
    continuation_score = scores.get("continuation", 0.5)
    decision_score = scores.get("decision", 0.5)
    overall_score = (
        recall_score * 0.25
        + artifact_score * 0.25
        + continuation_score * 0.25
        + decision_score * 0.25
    )

    return ProbeEvaluationResult(
        recall_score=recall_score,
        artifact_score=artifact_score,
        continuation_score=continuation_score,
        decision_score=decision_score,
        overall_score=overall_score,
        probe_results=probe_results,
        metadata={
            "probe_questions_used": list(questions.keys()),
            "context_length": len(context_text),
        },
    )


def evaluate_context_completeness_with_probes(
    completeness_score: float,
    probe_result: ProbeEvaluationResult | None,
    weight_probes: float = 0.4,
) -> float:
    """기존 완전성 점수와 프로브 평가를 결합합니다.

    RecursiveContextManager.evaluate_context_completeness 등과 함께 사용.
    """
    if probe_result is None:
        return completeness_score
    combined = (1.0 - weight_probes) * completeness_score + weight_probes * probe_result.overall_score
    return min(1.0, max(0.0, combined))
