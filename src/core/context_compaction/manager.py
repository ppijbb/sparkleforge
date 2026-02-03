"""
Context Compaction Manager

컨텍스트 압축 관리자 (AgentPG 패턴 참고).
85% 임계값에서 자동 압축을 수행합니다.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.core.db.database_driver import Transaction
from src.core.db.transaction_manager import get_transaction_manager
from src.core.context_compaction.strategies import (
    CompactionStrategy,
    HybridStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class CompactionConfig:
    """압축 설정."""
    trigger_threshold: float = 0.85  # 85% 임계값
    target_tokens: int = 0  # 목표 토큰 수 (0이면 자동 계산)
    preserve_last_n: int = 20  # 마지막 N개 메시지 보호
    preserve_last_tokens: int = 40000  # 마지막 40K 토큰 보호 (AgentPG 패턴)
    strategy: str = "hybrid"  # "prune" | "summarize" | "hybrid"
    auto_compact: bool = True  # 자동 압축 활성화
    max_context_tokens: int = 200000  # 최대 컨텍스트 토큰 수


@dataclass
class CompactionResult:
    """압축 결과."""
    session_id: str
    original_message_count: int
    compressed_message_count: int
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    strategy_used: str
    preserved_messages: int
    archived_messages: int
    created_at: datetime
    metadata: Dict[str, Any]


class CompactionManager:
    """
    컨텍스트 압축 관리자.
    
    AgentPG 패턴을 참고하여 구현:
    - 85% 임계값에서 자동 압축
    - 마지막 40K 토큰 보호
    - Hybrid Strategy (Prune + Summarize)
    """
    
    def __init__(
        self,
        config: Optional[CompactionConfig] = None,
        llm_client: Optional[Any] = None
    ):
        """
        초기화.
        
        Args:
            config: 압축 설정
            llm_client: LLM 클라이언트 (요약 생성용)
        """
        self.config = config or CompactionConfig()
        self.llm_client = llm_client
        self.tx_manager = get_transaction_manager()
        
        # 전략 등록
        self.strategies: Dict[str, CompactionStrategy] = {}
        self._register_default_strategies()
        
        logger.info(
            f"CompactionManager initialized: "
            f"threshold={self.config.trigger_threshold}, "
            f"strategy={self.config.strategy}"
        )
    
    def _register_default_strategies(self):
        """기본 전략 등록."""
        if self.llm_client:
            self.strategies["hybrid"] = HybridStrategy(self.llm_client)
            self.strategies["summarize"] = SummarizeStrategy(self.llm_client)
        self.strategies["prune"] = PruneStrategy()
    
    def register_strategy(self, name: str, strategy: CompactionStrategy):
        """전략 등록."""
        self.strategies[name] = strategy
        logger.info(f"Compaction strategy registered: {name}")
    
    async def should_compact(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        token_count: Optional[int] = None
    ) -> bool:
        """
        압축 필요 여부 확인.
        
        Args:
            session_id: 세션 ID
            messages: 메시지 리스트
            token_count: 현재 토큰 수 (None이면 자동 계산)
            
        Returns:
            압축 필요 여부
        """
        if not self.config.auto_compact:
            return False
        
        # 토큰 수 계산
        if token_count is None:
            token_count = self._estimate_tokens(messages)
        
        threshold_tokens = int(self.config.max_context_tokens * self.config.trigger_threshold)
        
        should_compact = token_count >= threshold_tokens
        
        if should_compact:
            logger.info(
                f"Compaction needed for session {session_id}: "
                f"{token_count} tokens >= {threshold_tokens} threshold"
            )
        
        return should_compact
    
    async def compact(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        strategy_name: Optional[str] = None,
        tx: Optional[Transaction] = None
    ) -> CompactionResult:
        """
        컨텍스트 압축 실행.
        
        Args:
            session_id: 세션 ID
            messages: 메시지 리스트
            strategy_name: 사용할 전략 이름 (None이면 설정값 사용)
            tx: 트랜잭션 (None이면 자동 생성)
            
        Returns:
            압축 결과
        """
        strategy_name = strategy_name or self.config.strategy
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown compaction strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        
        # 트랜잭션 관리
        if tx is None:
            async with self.tx_manager.transaction() as new_tx:
                return await self._compact_in_transaction(
                    session_id, messages, strategy, new_tx
                )
        else:
            return await self._compact_in_transaction(
                session_id, messages, strategy, tx
            )
    
    async def _compact_in_transaction(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        strategy: CompactionStrategy,
        tx: Transaction
    ) -> CompactionResult:
        """트랜잭션 내에서 압축 실행."""
        original_count = len(messages)
        original_tokens = self._estimate_tokens(messages)
        
        # 전략 실행
        compressed_messages = await strategy.compact(
            messages,
            self.config
        )
        
        compressed_count = len(compressed_messages)
        compressed_tokens = self._estimate_tokens(compressed_messages)
        tokens_saved = original_tokens - compressed_tokens
        
        # 원본 메시지 아카이브 (롤백 지원)
        archived_count = original_count - compressed_count
        
        result = CompactionResult(
            session_id=session_id,
            original_message_count=original_count,
            compressed_message_count=compressed_count,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            strategy_used=strategy.name(),
            preserved_messages=self.config.preserve_last_n,
            archived_messages=archived_count,
            created_at=datetime.now(),
            metadata={
                "compression_ratio": compressed_tokens / original_tokens if original_tokens > 0 else 0.0,
                "token_reduction_percent": (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0.0
            }
        )
        
        logger.info(
            f"Compaction completed for session {session_id}: "
            f"{original_count} -> {compressed_count} messages, "
            f"{original_tokens} -> {compressed_tokens} tokens "
            f"({tokens_saved} saved, {result.metadata['token_reduction_percent']:.1f}% reduction)"
        )
        
        return result
    
    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """토큰 수 추정 (간단한 추정)."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                # 간단한 추정: 영어 기준 단어 수의 1.3배
                word_count = len(content.split())
                total += int(word_count * 1.3)
            elif isinstance(content, list):
                # Content blocks
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        word_count = len(text.split())
                        total += int(word_count * 1.3)
        return total


# 전역 인스턴스
_compaction_manager: Optional[CompactionManager] = None


def get_compaction_manager() -> Optional[CompactionManager]:
    """전역 압축 관리자 인스턴스 반환."""
    return _compaction_manager


def set_compaction_manager(manager: CompactionManager) -> None:
    """전역 압축 관리자 설정."""
    global _compaction_manager
    _compaction_manager = manager
    logger.info("Compaction manager set")

