"""
Realtime Memory Processor

실시간 메모리 처리 시스템.
메시지 스트림에서 실시간으로 메모리를 추출하고 업데이트합니다.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.memory_types import BaseMemory
from src.core.memory_extraction import get_memory_extractor, get_memory_consolidator
from src.core.adaptive_memory import get_adaptive_memory
from src.core.db.database_driver import Transaction
from src.core.db.transaction_manager import get_transaction_manager

logger = logging.getLogger(__name__)


class RealtimeMemoryProcessor:
    """
    실시간 메모리 프로세서.
    
    메시지가 생성될 때마다 즉시 메모리를 추출하고 업데이트합니다.
    백그라운드 처리와 달리 낮은 지연 시간을 제공합니다.
    """
    
    def __init__(
        self,
        use_lightweight_model: bool = True,
        batch_size: int = 1,
        enable_incremental: bool = True
    ):
        """
        초기화.
        
        Args:
            use_lightweight_model: 경량 모델 사용 여부 (빠른 처리)
            batch_size: 배치 크기 (실시간이므로 보통 1)
            enable_incremental: 증분 업데이트 활성화
        """
        self.use_lightweight_model = use_lightweight_model
        self.batch_size = batch_size
        self.enable_incremental = enable_incremental
        
        # 메모리 추출기/통합기
        self.extractor = get_memory_extractor()
        self.consolidator = get_memory_consolidator()
        self.adaptive_memory = get_adaptive_memory()
        self.tx_manager = get_transaction_manager()
        
        # 처리 통계
        self.stats = {
            "processed_messages": 0,
            "extracted_memories": 0,
            "updated_memories": 0,
            "errors": 0,
            "avg_processing_time_ms": 0.0
        }
        
        logger.info("RealtimeMemoryProcessor initialized")
    
    async def process_message(
        self,
        session_id: str,
        user_id: str,
        message: Dict[str, Any],
        tx: Optional[Transaction] = None
    ) -> List[BaseMemory]:
        """
        메시지를 실시간으로 처리하여 메모리 추출 및 업데이트.
        
        Args:
            session_id: 세션 ID
            user_id: 사용자 ID
            message: 메시지 데이터
            tx: 트랜잭션 (None이면 자동 생성)
            
        Returns:
            추출된 메모리 리스트
        """
        import time
        start_time = time.time()
        
        try:
            # 트랜잭션 관리
            if tx is None:
                async with self.tx_manager.transaction() as new_tx:
                    return await self._process_message_in_transaction(
                        session_id, user_id, message, new_tx, start_time
                    )
            else:
                return await self._process_message_in_transaction(
                    session_id, user_id, message, tx, start_time
                )
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to process message in realtime: {e}", exc_info=True)
            return []
    
    async def _process_message_in_transaction(
        self,
        session_id: str,
        user_id: str,
        message: Dict[str, Any],
        tx: Transaction,
        start_time: float
    ) -> List[BaseMemory]:
        """트랜잭션 내에서 메시지 처리."""
        # 1. 빠른 메모리 추출 (경량 모델 사용)
        conversation_history = [message]  # 단일 메시지 또는 최근 N개
        
        extracted_memories = await self.extractor.extract_memories(
            conversation_history=conversation_history,
            session_id=session_id,
            user_id=user_id,
            turn=None  # 실시간이므로 turn 정보 없음
        )
        
        if not extracted_memories:
            logger.debug(f"No memories extracted from message in session {session_id}")
            return []
        
        # 2. 기존 메모리 로드 (통합을 위해)
        existing_memories = await self._load_existing_memories(user_id, tx)
        
        # 3. 증분 통합 (기존 메모리와 통합)
        if self.enable_incremental and existing_memories:
            consolidation_result = await self.consolidator.consolidate_memories(
                new_memories=extracted_memories,
                existing_memories=existing_memories,
                user_id=user_id
            )
            consolidated = consolidation_result["consolidated"]
        else:
            consolidated = extracted_memories
        
        # 4. 트랜잭션 내에서 메모리 저장
        saved_count = 0
        for memory in consolidated:
            success = await self.adaptive_memory.store_memory_async(memory, tx=tx)
            if success:
                saved_count += 1
        
        # 통계 업데이트
        processing_time_ms = (time.time() - start_time) * 1000
        self.stats["processed_messages"] += 1
        self.stats["extracted_memories"] += len(extracted_memories)
        self.stats["updated_memories"] += saved_count
        
        # 평균 처리 시간 업데이트
        current_avg = self.stats["avg_processing_time_ms"]
        count = self.stats["processed_messages"]
        self.stats["avg_processing_time_ms"] = (
            (current_avg * (count - 1) + processing_time_ms) / count
        )
        
        logger.debug(
            f"Realtime memory processed: {len(extracted_memories)} extracted, "
            f"{saved_count} saved in {processing_time_ms:.2f}ms"
        )
        
        return consolidated
    
    async def process_batch(
        self,
        session_id: str,
        user_id: str,
        messages: List[Dict[str, Any]],
        tx: Optional[Transaction] = None
    ) -> List[BaseMemory]:
        """
        여러 메시지를 배치로 처리.
        
        Args:
            session_id: 세션 ID
            user_id: 사용자 ID
            messages: 메시지 리스트
            tx: 트랜잭션
            
        Returns:
            추출된 메모리 리스트
        """
        all_memories = []
        
        if tx is None:
            async with self.tx_manager.transaction() as new_tx:
                for message in messages:
                    memories = await self._process_message_in_transaction(
                        session_id, user_id, message, new_tx, time.time()
                    )
                    all_memories.extend(memories)
        else:
            for message in messages:
                memories = await self._process_message_in_transaction(
                    session_id, user_id, message, tx, time.time()
                )
                all_memories.extend(memories)
        
        return all_memories
    
    async def _load_existing_memories(
        self,
        user_id: str,
        tx: Optional[Transaction] = None
    ) -> List[BaseMemory]:
        """기존 메모리 로드."""
        try:
            # AdaptiveMemory에서 사용자별 메모리 조회
            # 현재는 인메모리이므로 간단히 처리
            # 나중에 데이터베이스 저장소로 마이그레이션 시 트랜잭션 사용
            # TODO: 실제 메모리 저장소에서 로드
            # Issue: High priority - 실제 메모리 저장소에서 사용자별 메모리 로드 구현 필요
            # 현재는 빈 리스트 반환 (임시 구현)
            return []
        except Exception as e:
            logger.debug(f"Failed to load existing memories: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """처리 통계 반환."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """통계 초기화."""
        self.stats = {
            "processed_messages": 0,
            "extracted_memories": 0,
            "updated_memories": 0,
            "errors": 0,
            "avg_processing_time_ms": 0.0
        }


# 전역 인스턴스
_realtime_processor: Optional[RealtimeMemoryProcessor] = None


def get_realtime_memory_processor() -> RealtimeMemoryProcessor:
    """전역 실시간 메모리 프로세서 인스턴스 반환."""
    global _realtime_processor
    if _realtime_processor is None:
        _realtime_processor = RealtimeMemoryProcessor()
    return _realtime_processor


def set_realtime_memory_processor(processor: RealtimeMemoryProcessor) -> None:
    """전역 실시간 메모리 프로세서 설정."""
    global _realtime_processor
    _realtime_processor = processor
    logger.info("Realtime memory processor set")

