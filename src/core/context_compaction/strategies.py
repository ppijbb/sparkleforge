"""
Context Compaction Strategies

컨텍스트 압축 전략 구현.
AgentPG 패턴을 참고하여 Prune, Summarize, Hybrid 전략을 제공합니다.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.context_compaction.manager import CompactionConfig

logger = logging.getLogger(__name__)


class CompactionStrategy(ABC):
    """압축 전략 추상 클래스."""
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
    
    @abstractmethod
    def name(self) -> str:
        """전략 이름."""
        pass
    
    @abstractmethod
    async def compact(
        self,
        messages: List[Dict[str, Any]],
        config: CompactionConfig
    ) -> List[Dict[str, Any]]:
        """
        메시지 압축.
        
        Args:
            messages: 원본 메시지 리스트
            config: 압축 설정
            
        Returns:
            압축된 메시지 리스트
        """
        pass


class PruneStrategy(CompactionStrategy):
    """
    Prune 전략: Tool 출력 제거 (무료 압축).
    
    AgentPG 패턴: Tool 출력을 우선적으로 제거하여 토큰을 절감합니다.
    """
    
    def name(self) -> str:
        return "prune"
    
    async def compact(
        self,
        messages: List[Dict[str, Any]],
        config: CompactionConfig
    ) -> List[Dict[str, Any]]:
        """
        Tool 출력 제거로 압축.
        
        마지막 N개 메시지는 보호합니다.
        """
        if len(messages) <= config.preserve_last_n:
            logger.debug("Not enough messages to compact")
            return messages
        
        # 마지막 N개 메시지 보호
        protected = messages[-config.preserve_last_n:]
        to_compact = messages[:-config.preserve_last_n]
        
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
    """
    Summarize 전략: LLM으로 요약 생성 (유료).
    
    AgentPG 패턴: 8-section 구조로 요약합니다.
    """
    
    def name(self) -> str:
        return "summarize"
    
    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(llm_client)
        if llm_client is None:
            logger.warning("SummarizeStrategy requires LLM client")
    
    async def compact(
        self,
        messages: List[Dict[str, Any]],
        config: CompactionConfig
    ) -> List[Dict[str, Any]]:
        """
        LLM으로 요약 생성.
        
        마지막 N개 메시지는 보호하고, 나머지를 요약합니다.
        """
        if self.llm_client is None:
            logger.warning("LLM client not available, falling back to prune")
            prune_strategy = PruneStrategy()
            return await prune_strategy.compact(messages, config)
        
        if len(messages) <= config.preserve_last_n:
            logger.debug("Not enough messages to compact")
            return messages
        
        # 마지막 N개 메시지 보호
        protected = messages[-config.preserve_last_n:]
        to_summarize = messages[:-config.preserve_last_n]
        
        # 요약 생성
        summary = await self._generate_summary(to_summarize)
        
        # 요약 메시지 생성
        summary_message = {
            "role": "system",
            "content": summary,
            "metadata": {
                "type": "compaction_summary",
                "original_count": len(to_summarize),
                "compressed_at": datetime.now().isoformat(),
                "strategy": "summarize"
            }
        }
        
        # 보호된 메시지 + 요약
        result = [summary_message] + protected
        
        logger.info(
            f"Summarize strategy: {len(messages)} -> {len(result)} messages "
            f"({len(to_summarize)} messages summarized)"
        )
        
        return result
    
    async def _generate_summary(self, messages: List[Dict[str, Any]]) -> str:
        """8-section 구조로 요약 생성 (Claude Code 패턴)."""
        # 메시지 내용 추출
        content_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if isinstance(content, str):
                content_parts.append(f"[{role}]: {content[:500]}")  # 처음 500자만
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        content_parts.append(f"[{role}]: {text[:500]}")
        
        full_content = "\n".join(content_parts)
        
        # 8-section 구조 요약 프롬프트
        prompt = f"""다음 대화 내용을 8-section 구조로 요약해주세요:

1. 주요 주제 (Main Topics)
2. 중요한 결정 사항 (Key Decisions)
3. 실행된 작업 (Actions Taken)
4. 발견된 정보 (Information Discovered)
5. 해결된 문제 (Problems Solved)
6. 남은 질문 (Remaining Questions)
7. 다음 단계 (Next Steps)
8. 중요 인사이트 (Key Insights)

대화 내용:
{full_content[:5000]}  # 최대 5000자

요약:"""
        
        try:
            # LLM 호출 (간단한 예시, 실제로는 LLM 클라이언트 사용)
            if hasattr(self.llm_client, "generate"):
                summary = await self.llm_client.generate(prompt)
            else:
                # 폴백: 간단한 요약
                summary = f"요약: {len(messages)}개의 메시지가 {len(full_content)}자로 요약되었습니다."
            
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"요약 생성 실패: {len(messages)}개 메시지"


class HybridStrategy(CompactionStrategy):
    """
    Hybrid 전략: Prune + Summarize (권장).
    
    AgentPG 패턴:
    1. Tool 출력 제거 (무료)
    2. 나머지를 요약 (유료)
    3. 마지막 40K 토큰 보호
    """
    
    def name(self) -> str:
        return "hybrid"
    
    def __init__(self, llm_client: Optional[Any] = None):
        super().__init__(llm_client)
        self.prune_strategy = PruneStrategy()
        self.summarize_strategy = SummarizeStrategy(llm_client) if llm_client else None
    
    async def compact(
        self,
        messages: List[Dict[str, Any]],
        config: CompactionConfig
    ) -> List[Dict[str, Any]]:
        """
        Hybrid 압축: Prune 후 Summarize.
        
        AgentPG 패턴:
        1. 마지막 40K 토큰 보호
        2. Tool 출력 제거 (무료)
        3. 나머지 요약 (유료)
        """
        if len(messages) <= config.preserve_last_n:
            logger.debug("Not enough messages to compact")
            return messages
        
        # 1. 마지막 N개 메시지 보호 (또는 토큰 기반)
        protected = messages[-config.preserve_last_n:]
        to_compact = messages[:-config.preserve_last_n]
        
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

