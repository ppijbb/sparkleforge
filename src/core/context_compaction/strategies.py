"""Context Compaction Strategies

컨텍스트 압축 전략 구현.
AgentPG 패턴을 참고하여 Prune, Summarize, Hybrid 전략을 제공합니다.
2025 트렌드: Forgetting(TTL·중요도 기반 삭제), Condensation(핵심만 축약).
Agent-Skills-for-Context-Engineering: Anchored Iterative Summarization (구조화된 요약 병합).
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from src.core.context_compaction.manager import CompactionConfig

logger = logging.getLogger(__name__)

# Anchored Iterative Summarization: 명시적 섹션 (정보 손실 방지)
STRUCTURED_SUMMARY_SECTIONS = [
    "Session Intent",
    "Files Modified",
    "Decisions Made",
    "Current State",
    "Next Steps",
]


class CompactionStrategy(ABC):
    """압축 전략 추상 클래스."""

    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client

    @abstractmethod
    def name(self) -> str:
        """전략 이름."""
        pass

    @abstractmethod
    async def compact(
        self, messages: List[Dict[str, Any]], config: CompactionConfig
    ) -> List[Dict[str, Any]]:
        """메시지 압축.

        Args:
            messages: 원본 메시지 리스트
            config: 압축 설정

        Returns:
            압축된 메시지 리스트
        """
        pass


class PruneStrategy(CompactionStrategy):
    """Prune 전략: Tool 출력 제거 (무료 압축).

    AgentPG 패턴: Tool 출력을 우선적으로 제거하여 토큰을 절감합니다.
    """

    def name(self) -> str:
        return "prune"

    async def compact(
        self, messages: List[Dict[str, Any]], config: CompactionConfig
    ) -> List[Dict[str, Any]]:
        """Tool 출력 제거로 압축.

        마지막 N개 메시지는 보호합니다.
        """
        if len(messages) <= config.preserve_last_n:
            logger.debug("Not enough messages to compact")
            return messages

        # 마지막 N개 메시지 보호
        protected = messages[-config.preserve_last_n :]
        to_compact = messages[: -config.preserve_last_n]

        # Tool 출력 제거
        pruned = []
        for msg in to_compact:
            # Tool 출력인지 확인
            if not self._is_tool_output(msg):
                pruned.append(msg)
            else:
                logger.debug(f"Pruned tool output message: {msg.get('id', 'unknown')}")

        # 보호된 메시지 + 압축된 메시지
        result = pruned + protected

        logger.info(
            f"Prune strategy: {len(messages)} -> {len(result)} messages "
            f"({len(messages) - len(result)} tool outputs removed)"
        )

        return result

    def _is_tool_output(self, message: Dict[str, Any]) -> bool:
        """Tool 출력 메시지인지 확인."""
        role = message.get("role", "")
        content = message.get("content", "")

        # Tool call 결과인지 확인
        if role == "tool" or role == "assistant":
            # Tool call 메타데이터 확인
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        return True
            elif isinstance(content, str):
                # Tool 출력 패턴 확인
                if "tool_result" in message.get("metadata", {}):
                    return True

        return False


class SummarizeStrategy(CompactionStrategy):
    """Summarize 전략: LLM으로 요약 생성 (유료).

    Agent-Skills-for-Context-Engineering: Anchored Iterative Summarization.
    - 명시적 섹션 (Session Intent, Files Modified, Decisions Made, Current State, Next Steps) 강제
    - 기존 요약이 있으면 새 메시지만 요약 후 섹션별 병합 (Incremental Merging)
    """

    def name(self) -> str:
        return "summarize"

    def __init__(self, llm_client: Any | None = None):
        super().__init__(llm_client)
        if llm_client is None:
            logger.warning("SummarizeStrategy requires LLM client")

    async def compact(
        self, messages: List[Dict[str, Any]], config: CompactionConfig
    ) -> List[Dict[str, Any]]:
        """LLM으로 요약 생성. 기존 요약이 있으면 새 내용만 요약 후 병합."""
        if self.llm_client is None:
            logger.warning("LLM client not available, falling back to prune")
            prune_strategy = PruneStrategy()
            return await prune_strategy.compact(messages, config)

        if len(messages) <= config.preserve_last_n:
            logger.debug("Not enough messages to compact")
            return messages

        protected = messages[-config.preserve_last_n :]
        to_summarize = messages[: -config.preserve_last_n]

        # Anchored Iterative: 기존 구조화 요약 추출 (첫 메시지가 compaction_summary인 경우)
        existing_summary = None
        new_messages = to_summarize
        meta = (to_summarize[0].get("metadata") or {}) if to_summarize else {}
        if (
            to_summarize
            and meta.get("type") == "compaction_summary"
            and self._has_structured_sections(to_summarize[0].get("content", ""))
        ):
            existing_summary = (
                to_summarize[0].get("content")
                if isinstance(to_summarize[0].get("content"), str)
                else ""
            )
            new_messages = to_summarize[1:]
            logger.debug("Incremental merge: using existing structured summary")
        if not new_messages and existing_summary:
            return [to_summarize[0]] + protected

        # 요약 생성 (전체 또는 새 메시지만, 병합 시 병합)
        summary = await self._generate_summary(
            new_messages, existing_structured_summary=existing_summary
        )

        summary_message = {
            "role": "system",
            "content": summary,
            "metadata": {
                "type": "compaction_summary",
                "original_count": len(to_summarize),
                "compressed_at": datetime.now().isoformat(),
                "strategy": "summarize",
                "anchored_iterative": True,
            },
        }

        result = [summary_message] + protected
        logger.info(
            f"Summarize strategy: {len(messages)} -> {len(result)} messages "
            f"({len(to_summarize)} messages summarized)"
        )
        return result

    def _has_structured_sections(self, content: str) -> bool:
        """요약 본문이 구조화된 섹션을 포함하는지 확인."""
        if not content or not isinstance(content, str):
            return False
        for section in STRUCTURED_SUMMARY_SECTIONS:
            if f"## {section}" in content or f"## {section}:" in content:
                return True
        return False

    def _parse_structured_sections(self, content: str) -> Dict[str, str]:
        """구조화된 요약에서 섹션별 텍스트 추출."""
        sections: Dict[str, str] = {s: "" for s in STRUCTURED_SUMMARY_SECTIONS}
        pattern = re.compile(
            r"##\s*(" + "|".join(re.escape(s) for s in STRUCTURED_SUMMARY_SECTIONS) + r")\s*:?\s*\n(.*?)(?=##|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        for m in pattern.finditer(content):
            name = m.group(1).strip()
            for key in STRUCTURED_SUMMARY_SECTIONS:
                if key.lower() == name.lower():
                    sections[key] = m.group(2).strip()
                    break
        return sections

    def _merge_structured_sections(
        self, existing: Dict[str, str], new: Dict[str, str]
    ) -> str:
        """기존 요약과 새 요약을 섹션별로 병합. 새 내용이 있으면 추가/갱신."""
        out = []
        for section in STRUCTURED_SUMMARY_SECTIONS:
            existing_val = (existing.get(section) or "").strip()
            new_val = (new.get(section) or "").strip()
            if new_val:
                combined = f"{existing_val}\n{new_val}".strip() if existing_val else new_val
            else:
                combined = existing_val
            if combined:
                out.append(f"## {section}\n{combined}")
        return "\n\n".join(out) if out else ""

    async def _generate_summary(
        self,
        messages: List[Dict[str, Any]],
        existing_structured_summary: str | None = None,
    ) -> str:
        """Anchored Iterative: 구조화된 섹션으로 요약 생성 또는 병합."""
        content_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                content_parts.append(f"[{role}]: {content[:500]}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        content_parts.append(f"[{role}]: {text[:500]}")
        full_content = "\n".join(content_parts)
        max_content_len = 5000
        truncated = full_content[:max_content_len]

        section_list = "\n".join(f"- {s}" for s in STRUCTURED_SUMMARY_SECTIONS)
        if existing_structured_summary:
            existing_sections = self._parse_structured_sections(
                existing_structured_summary
            )
            prompt = f"""다음은 새로 추가된 대화 내용입니다. 이 내용만 보고 구조화 요약을 생성하세요.
기존 요약과 병합할 것이므로, 새 대화에서 나온 내용만 각 섹션에 채우세요.

필수 섹션 (반드시 모두 포함, 없으면 "없음" 또는 "-"):
{section_list}

새 대화 내용:
{truncated}

다음 형식으로만 응답하세요 (다른 설명 없이):
## Session Intent
...
## Files Modified
...
## Decisions Made
...
## Current State
...
## Next Steps
..."""
        else:
            prompt = f"""다음 대화 내용을 아래 필수 섹션 구조로 요약해주세요. 각 섹션을 반드시 채우세요.

필수 섹션:
{section_list}

- Session Intent: 사용자/세션이 달성하려는 목표
- Files Modified: 수정/생성/참조된 파일 목록 (없으면 "없음")
- Decisions Made: 중요한 결정 사항
- Current State: 현재 진행 상황
- Next Steps: 다음에 할 일

대화 내용:
{truncated}

다음 형식으로만 응답하세요 (다른 설명 없이):
## Session Intent
...
## Files Modified
...
## Decisions Made
...
## Current State
...
## Next Steps
..."""

        try:
            if hasattr(self.llm_client, "generate"):
                summary = await self.llm_client.generate(prompt)
            else:
                summary = f"## Session Intent\n(요약: {len(messages)}개 메시지)\n## Files Modified\n없음\n## Decisions Made\n없음\n## Current State\n진행 중\n## Next Steps\n없음"

            if existing_structured_summary and summary:
                existing_sections = self._parse_structured_sections(
                    existing_structured_summary
                )
                new_sections = self._parse_structured_sections(summary)
                summary = self._merge_structured_sections(
                    existing_sections, new_sections
                )
            return summary if summary else "요약 없음"
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"요약 생성 실패: {len(messages)}개 메시지"


class HybridStrategy(CompactionStrategy):
    """Hybrid 전략: Prune + Summarize (권장).

    AgentPG 패턴:
    1. Tool 출력 제거 (무료)
    2. 나머지를 요약 (유료)
    3. 마지막 40K 토큰 보호
    """

    def name(self) -> str:
        return "hybrid"

    def __init__(self, llm_client: Any | None = None):
        super().__init__(llm_client)
        self.prune_strategy = PruneStrategy()
        self.summarize_strategy = SummarizeStrategy(llm_client) if llm_client else None

    async def compact(
        self, messages: List[Dict[str, Any]], config: CompactionConfig
    ) -> List[Dict[str, Any]]:
        """Hybrid 압축: Prune 후 Summarize.

        AgentPG 패턴:
        1. 마지막 40K 토큰 보호
        2. Tool 출력 제거 (무료)
        3. 나머지 요약 (유료)
        """
        if len(messages) <= config.preserve_last_n:
            logger.debug("Not enough messages to compact")
            return messages

        # 1. 마지막 N개 메시지 보호 (또는 토큰 기반)
        protected = messages[-config.preserve_last_n :]
        to_compact = messages[: -config.preserve_last_n]

        if not to_compact:
            return protected

        # 2. Tool 출력 제거 (무료)
        pruned = await self.prune_strategy.compact(to_compact, config)

        # 3. 나머지 요약 (유료, LLM 클라이언트가 있는 경우만)
        if self.summarize_strategy and len(pruned) > 0:
            # 요약할 메시지가 충분히 많은 경우만 요약
            if len(pruned) > config.preserve_last_n:
                summarized = await self.summarize_strategy.compact(pruned, config)
                result = summarized + protected
            else:
                result = pruned + protected
        else:
            # 요약 불가능하면 Prune만 사용
            result = pruned + protected

        logger.info(
            f"Hybrid strategy: {len(messages)} -> {len(result)} messages "
            f"(pruned + summarized)"
        )

        return result


class ForgettingStrategy(CompactionStrategy):
    """Forgetting 전략: TTL·중요도 기반 삭제 (2025 메모리 트렌드).

    - 메시지 메타데이터의 timestamp가 forget_ttl_seconds 초과 시 제거
    - importance가 importance_threshold 미만이면 제거
    - 마지막 preserve_last_n 개는 항상 유지
    """

    def name(self) -> str:
        return "forgetting"

    async def compact(
        self, messages: List[Dict[str, Any]], config: CompactionConfig
    ) -> List[Dict[str, Any]]:
        if len(messages) <= getattr(config, "preserve_last_n", 20):
            return messages
        preserve_last_n = getattr(config, "preserve_last_n", 20)
        forget_ttl = getattr(config, "forget_ttl_seconds", None)
        importance_threshold = getattr(config, "importance_threshold", 0.0)
        protected = messages[-preserve_last_n:]
        to_filter = messages[:-preserve_last_n]
        now = time.time()
        kept = []
        for msg in to_filter:
            meta = msg.get("metadata") or {}
            ts = meta.get("timestamp") or meta.get("created_at")
            if isinstance(ts, (int, float)):
                if forget_ttl is not None and (now - ts) > forget_ttl:
                    continue
            imp = meta.get("importance", 1.0)
            if isinstance(imp, (int, float)) and imp < importance_threshold:
                continue
            kept.append(msg)
        result = kept + protected
        logger.info(
            f"Forgetting strategy: {len(messages)} -> {len(result)} messages "
            f"(ttl={forget_ttl}, importance>={importance_threshold})"
        )
        return result


class CondensationStrategy(CompactionStrategy):
    """Condensation 전략: 핵심만 추출하여 추가 축약 (2025 메모리 트렌드).

    요약/긴 메시지를 핵심 사실 위주로 한 번 더 압축합니다.
    LLM 클라이언트가 있으면 사용하고, 없으면 prune만 적용.
    """

    def name(self) -> str:
        return "condensation"

    def __init__(self, llm_client: Any | None = None):
        super().__init__(llm_client)
        self.prune = PruneStrategy()

    async def compact(
        self, messages: List[Dict[str, Any]], config: CompactionConfig
    ) -> List[Dict[str, Any]]:
        preserve_last_n = getattr(config, "preserve_last_n", 20)
        if len(messages) <= preserve_last_n or not self.llm_client:
            if not self.llm_client and len(messages) > preserve_last_n:
                return await self.prune.compact(messages, config)
            return messages
        protected = messages[-preserve_last_n:]
        to_condense = messages[:-preserve_last_n]
        condensed = []
        for msg in to_condense:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 800:
                try:
                    if hasattr(self.llm_client, "generate"):
                        short = await self.llm_client.generate(
                            f"Extract only key facts (bullet points, max 200 chars):\n{content[:2000]}"
                        )
                    else:
                        short = content[:400] + "..."
                    condensed.append(
                        {
                            **msg,
                            "content": short,
                            "metadata": {
                                **(msg.get("metadata") or {}),
                                "condensed": True,
                            },
                        }
                    )
                except Exception as e:
                    logger.debug(f"Condensation skip: {e}")
                    condensed.append(msg)
            else:
                condensed.append(msg)
        result = condensed + protected
        logger.info(
            f"Condensation strategy: {len(messages)} -> {len(result)} messages "
            f"(long messages condensed)"
        )
        return result
