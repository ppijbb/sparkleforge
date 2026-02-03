"""
컨텍스트 엔지니어링 시스템

백서 요구사항: 4단계 사이클 (Fetch → Prepare → Invoke → Upload)
- Fetch Context: 메모리, RAG, 세션 히스토리 조회
- Prepare Context: 동적 프롬프트 구성
- Invoke LLM and Tools: 실행
- Upload Context: 백그라운드 저장

토큰 한계 내에서 입력·출력·도구 설명을 최적화하는 설계
컨텍스트 윈도우 내 토큰 수 제한 관리, 중요도 기반 컨텍스트 압축 및 요약
서브 에이전트 간 컨텍스트 교환 최적화
Context Rot 방지: Recursive summarization, Selective pruning, Token-based truncation
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextPriority(Enum):
    """컨텍스트 우선순위."""
    CRITICAL = 5  # 반드시 보존해야 하는 정보
    HIGH = 4      # 중요한 정보
    MEDIUM = 3    # 보통 중요도
    LOW = 2       # 낮은 중요도
    OPTIONAL = 1  # 선택적 정보


class ContextType(Enum):
    """컨텍스트 유형."""
    SYSTEM_PROMPT = "system_prompt"
    USER_QUERY = "user_query"
    TOOL_RESULTS = "tool_results"
    AGENT_MEMORY = "agent_memory"
    CONVERSATION_HISTORY = "conversation_history"
    METADATA = "metadata"


@dataclass
class ContextChunk:
    """컨텍스트 청크."""
    content: str
    content_type: ContextType
    priority: ContextPriority
    timestamp: float
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)  # 의존하는 다른 청크들
    compressed: bool = False
    original_size: int = 0

    def __post_init__(self):
        self.original_size = len(self.content)
        self.token_count = self._estimate_tokens()

    def _estimate_tokens(self) -> int:
        """토큰 수 추정 (대략적)."""
        # 간단한 추정: 영어 기준 단어 수의 1.3배
        word_count = len(self.content.split())
        return int(word_count * 1.3)

    def compress(self, max_tokens: int) -> 'ContextChunk':
        """청크 압축."""
        if self.token_count <= max_tokens or self.compressed:
            return self

        # 압축 로직
        compressed_content = self._compress_content(max_tokens)
        compressed_chunk = ContextChunk(
            content=compressed_content,
            content_type=self.content_type,
            priority=self.priority,
            timestamp=self.timestamp,
            metadata=self.metadata.copy(),
            dependencies=self.dependencies.copy(),
            compressed=True,
            original_size=self.original_size
        )
        compressed_chunk.metadata['compression_ratio'] = len(compressed_content) / len(self.content)

        return compressed_chunk

    def _compress_content(self, max_tokens: int) -> str:
        """내용 압축."""
        if self.content_type == ContextType.TOOL_RESULTS:
            return self._compress_tool_results(max_tokens)
        elif self.content_type == ContextType.CONVERSATION_HISTORY:
            return self._compress_conversation(max_tokens)
        elif self.content_type == ContextType.AGENT_MEMORY:
            return self._compress_memory(max_tokens)
        else:
            # 기본 압축: 중요 부분 추출
            return self._extract_important_parts(max_tokens)

    def _compress_tool_results(self, max_tokens: int) -> str:
        """도구 결과 압축."""
        try:
            # JSON 파싱 시도
            data = json.loads(self.content)
            if isinstance(data, dict) and 'results' in data:
                results = data['results']
                if isinstance(results, list):
                    # 상위 3개 결과만 유지
                    compressed_results = results[:3]
                    summary = f"총 {len(results)}개 결과 중 {len(compressed_results)}개 표시"
                    return json.dumps({
                        'results': compressed_results,
                        'summary': summary,
                        'total_count': len(results)
                    }, ensure_ascii=False)
        except:
            pass

        # 기본 압축
        return self._extract_important_parts(max_tokens)

    def _compress_conversation(self, max_tokens: int) -> str:
        """대화 히스토리 압축."""
        try:
            # JSON 파싱 시도
            messages = json.loads(self.content)
            if isinstance(messages, list):
                # 최근 5개 메시지만 유지
                recent_messages = messages[-5:] if len(messages) > 5 else messages
                summary = f"총 {len(messages)}개 메시지 중 최근 {len(recent_messages)}개 표시"
                return json.dumps({
                    'messages': recent_messages,
                    'summary': summary,
                    'total_count': len(messages)
                }, ensure_ascii=False)
        except:
            pass

        return self._extract_important_parts(max_tokens)

    def _compress_memory(self, max_tokens: int) -> str:
        """메모리 압축."""
        try:
            memory = json.loads(self.content)
            if isinstance(memory, dict):
                # 중요한 키만 유지
                important_keys = ['goals', 'findings', 'decisions', 'status']
                compressed_memory = {
                    k: v for k, v in memory.items()
                    if k in important_keys
                }
                return json.dumps(compressed_memory, ensure_ascii=False)
        except:
            pass

        return self._extract_important_parts(max_tokens)

    def _extract_important_parts(self, max_tokens: int) -> str:
        """중요 부분 추출."""
        content = self.content
        target_chars = max_tokens * 3  # 대략적인 문자 수

        if len(content) <= target_chars:
            return content

        # 처음과 끝 부분 유지 (중요한 정보가 양쪽에 있을 수 있음)
        start_chars = int(target_chars * 0.6)
        end_chars = target_chars - start_chars

        compressed = content[:start_chars] + "\n...\n" + content[-end_chars:]
        return compressed


@dataclass
class ContextWindowConfig:
    """컨텍스트 윈도우 설정."""
    max_tokens: int = 8000
    min_tokens: int = 1000
    system_prompt_tokens: int = 1000
    tool_descriptions_tokens: int = 1500
    conversation_tokens: int = 4000
    metadata_tokens: int = 500

    # 압축 설정
    enable_auto_compression: bool = True
    compression_threshold: float = 0.8  # 80% 사용 시 압축 시작
    importance_based_preservation: bool = True

    # 최적화 설정
    enable_token_estimation: bool = True
    adaptive_allocation: bool = True


class ContextEngineer:
    """
    컨텍스트 엔지니어링 시스템.

    백서 요구사항: 4단계 사이클 (Fetch → Prepare → Invoke → Upload)
    토큰 한계를 효율적으로 관리하고, 중요도 기반으로 컨텍스트를 최적화.
    Context Rot 방지: Recursive summarization, Selective pruning, Token-based truncation
    """

    def __init__(self, config: Optional[ContextWindowConfig] = None):
        """초기화."""
        self.config = config or ContextWindowConfig()
        self.context_chunks: List[ContextChunk] = []
        self.token_usage_history: List[Dict[str, Any]] = []
        self.compression_stats: Dict[str, int] = defaultdict(int)
        
        # Context Engineering 사이클 상태
        self.current_cycle: Optional[Dict[str, Any]] = None

        logger.info("ContextEngineer initialized with config: "
                   f"max_tokens={self.config.max_tokens}, "
                   f"auto_compression={self.config.enable_auto_compression}")
    
    async def fetch_context(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch Context 단계 (백서 요구사항).
        
        메모리, RAG, 세션 히스토리를 조회합니다.
        
        Args:
            user_query: 사용자 쿼리
            session_id: 세션 ID
            user_id: 사용자 ID
            
        Returns:
            조회된 컨텍스트
        """
        try:
            fetched_context = {
                "memories": [],
                "rag_documents": [],
                "session_history": [],
                "metadata": {}
            }
            
            # 1. 메모리 조회
            if user_id:
                try:
                    from src.core.memory_service import get_background_memory_service
                    # TODO: 실제 메모리 저장소에서 조회 (현재는 빈 리스트)
                    # Issue: High priority - 실제 메모리 저장소에서 관련 메모리 조회 구현 필요
                    # memories = await memory_storage.get_relevant_memories(user_id, user_query)
                    fetched_context["memories"] = []
                except Exception as e:
                    logger.debug(f"Memory retrieval failed: {e}")
            
            # 2. RAG 문서 조회 (외부 지식)
            # TODO: RAG 시스템 통합
            # Issue: High priority - RAG 시스템 통합 필요 (벡터 DB, 임베딩 검색 등)
            # 현재는 빈 리스트 반환
            fetched_context["rag_documents"] = []
            
            # 3. 세션 히스토리 조회
            if session_id:
                try:
                    from src.core.shared_memory import get_shared_memory
                    shared_memory = get_shared_memory()
                    session_memories = shared_memory.list_session_memories(session_id)
                    fetched_context["session_history"] = list(session_memories.values())
                except Exception as e:
                    logger.debug(f"Session history retrieval failed: {e}")
            
            logger.debug(f"Fetched context: {len(fetched_context['memories'])} memories, "
                        f"{len(fetched_context['rag_documents'])} RAG docs, "
                        f"{len(fetched_context['session_history'])} session entries")
            
            return fetched_context
            
        except Exception as e:
            logger.error(f"Failed to fetch context: {e}")
            return {"memories": [], "rag_documents": [], "session_history": [], "metadata": {}}
    
    async def prepare_context(
        self,
        user_query: str,
        fetched_context: Dict[str, Any],
        available_tokens: int
    ) -> Dict[str, Any]:
        """
        Prepare Context 단계 (백서 요구사항).
        
        동적 프롬프트를 구성합니다.
        
        Args:
            user_query: 사용자 쿼리
            fetched_context: Fetch 단계에서 조회한 컨텍스트
            available_tokens: 사용 가능한 토큰 수
            
        Returns:
            구성된 컨텍스트
        """
        try:
            # Fetch된 컨텍스트를 ContextChunk로 변환
            # 1. 메모리 추가
            for memory in fetched_context.get("memories", []):
                await self.add_context(
                    content=str(memory.get("content", "")),
                    content_type=ContextType.AGENT_MEMORY,
                    priority=ContextPriority.HIGH,
                    metadata={"source": "memory", "memory_id": memory.get("id")}
                )
            
            # 2. RAG 문서 추가
            for doc in fetched_context.get("rag_documents", []):
                await self.add_context(
                    content=str(doc.get("content", "")),
                    content_type=ContextType.TOOL_RESULTS,
                    priority=ContextPriority.MEDIUM,
                    metadata={"source": "rag", "doc_id": doc.get("id")}
                )
            
            # 3. 세션 히스토리 추가
            for entry in fetched_context.get("session_history", [])[:10]:  # 최근 10개만
                if isinstance(entry, dict):
                    await self.add_context(
                        content=str(entry.get("value", "")),
                        content_type=ContextType.CONVERSATION_HISTORY,
                        priority=ContextPriority.MEDIUM,
                        metadata={"source": "session_history"}
                    )
            
            # 4. 사용자 쿼리 추가
            await self.add_context(
                content=user_query,
                content_type=ContextType.USER_QUERY,
                priority=ContextPriority.CRITICAL
            )
            
            # 5. 컨텍스트 최적화
            optimized = await self.optimize_context(available_tokens=available_tokens)
            
            logger.debug(f"Prepared context: {len(optimized['chunks'])} chunks, "
                        f"{optimized['stats']['total_tokens']} tokens")
            
            return optimized
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return {"chunks": [], "stats": {}, "allocation": {}}
    
    async def upload_context(
        self,
        session_id: str,
        context_data: Dict[str, Any],
        agent_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Upload Context 단계 (백서 요구사항).
        
        백그라운드로 컨텍스트를 저장합니다.
        
        Args:
            session_id: 세션 ID
            context_data: 컨텍스트 데이터
            agent_state: AgentState (선택적)
            
        Returns:
            성공 여부
        """
        try:
            # 백그라운드 저장 (비동기, non-blocking)
            # 실제로는 세션 저장소에 저장하되, 사용자 경험을 차단하지 않음
            import asyncio
            
            async def _background_upload():
                try:
                    from src.core.session_manager import get_session_manager
                    session_manager = get_session_manager()
                    
                    if agent_state:
                        await session_manager.save_session(
                            session_id=session_id,
                            agent_state=agent_state,
                            metadata={"uploaded_at": datetime.now().isoformat()}
                        )
                    logger.debug(f"Context uploaded in background for session {session_id}")
                except Exception as e:
                    logger.warning(f"Background context upload failed: {e}")
            
            # 백그라운드 태스크로 실행 (await하지 않음)
            asyncio.create_task(_background_upload())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload context: {e}")
            return False
    
    async def execute_context_cycle(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        available_tokens: int = 8000
    ) -> Dict[str, Any]:
        """
        Context Engineering 4단계 사이클 실행 (백서 요구사항).
        
        Args:
            user_query: 사용자 쿼리
            session_id: 세션 ID
            user_id: 사용자 ID
            available_tokens: 사용 가능한 토큰 수
            
        Returns:
            사이클 실행 결과
        """
        try:
            cycle_result = {
                "fetch": None,
                "prepare": None,
                "invoke": None,  # LLM 호출은 외부에서 수행
                "upload": None,
                "cycle_id": f"cycle_{int(time.time() * 1000)}"
            }
            
            # 1. Fetch Context
            fetched = await self.fetch_context(user_query, session_id, user_id)
            cycle_result["fetch"] = fetched
            
            # 2. Prepare Context
            prepared = await self.prepare_context(user_query, fetched, available_tokens)
            cycle_result["prepare"] = prepared
            
            # 3. Invoke LLM and Tools (외부에서 수행, 여기서는 준비만)
            # 실제 LLM 호출은 AgentOrchestrator에서 수행
            cycle_result["invoke"] = {
                "status": "ready",
                "prepared_context": prepared
            }
            
            # 4. Upload Context (백그라운드)
            # 실제 업로드는 세션 종료 시 수행
            
            self.current_cycle = cycle_result
            logger.info(f"Context Engineering cycle completed: {cycle_result['cycle_id']}")
            
            return cycle_result
            
        except Exception as e:
            logger.error(f"Context Engineering cycle failed: {e}")
            return {"error": str(e)}

    async def add_context(
        self,
        content: str,
        content_type: ContextType,
        priority: ContextPriority = ContextPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Set[str]] = None
    ) -> str:
        """
        컨텍스트 추가.

        Args:
            content: 컨텍스트 내용
            content_type: 컨텍스트 유형
            priority: 우선순위
            metadata: 메타데이터
            dependencies: 의존 관계

        Returns:
            청크 ID
        """
        chunk_id = f"{content_type.value}_{int(time.time() * 1000)}_{len(self.context_chunks)}"

        chunk = ContextChunk(
            content=content,
            content_type=content_type,
            priority=priority,
            timestamp=time.time(),
            metadata=metadata or {},
            dependencies=dependencies or set()
        )

        self.context_chunks.append(chunk)
        logger.debug(f"Added context chunk: {chunk_id}, tokens={chunk.token_count}, type={content_type.value}")

        return chunk_id

    async def optimize_context(self, available_tokens: Optional[int] = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        컨텍스트 최적화 (캐시 및 점진적 로딩 지원).

        Args:
            available_tokens: 사용 가능한 토큰 수 (None이면 설정값 사용)
            use_cache: 캐시 사용 여부

        Returns:
            최적화 결과
        """
        max_tokens = available_tokens or self.config.max_tokens

        # 캐시 확인
        if use_cache:
            try:
                from src.core.context_cache import get_context_cache
                context_cache = get_context_cache()
                
                # 캐시 키 생성
                context_types = [chunk.content_type.value for chunk in self.context_chunks]
                query = ""  # 쿼리는 별도로 전달 필요
                token_allocation = self._calculate_token_allocation(max_tokens)
                cache_key = context_cache.generate_cache_key(context_types, query, token_allocation)
                
                cached = context_cache.get(cache_key, query)
                if cached:
                    logger.info(f"Using cached context: {cache_key}")
                    return {
                        'chunks': cached.context_chunks,
                        'stats': {
                            'total_chunks': len(cached.context_chunks),
                            'total_tokens': sum(chunk.get('token_count', 0) for chunk in cached.context_chunks),
                            'available_tokens': max_tokens,
                            'usage_ratio': 0.0,
                            'compressed_chunks': 0,
                            'optimization_applied': True,
                            'cached': True
                        },
                        'allocation': cached.token_allocation
                    }
            except Exception as e:
                logger.debug(f"Cache lookup failed: {e}")

        # 현재 토큰 사용량 계산
        total_tokens = sum(chunk.token_count for chunk in self.context_chunks)
        usage_ratio = total_tokens / max_tokens if max_tokens > 0 else 1.0

        logger.info(f"Context optimization: {total_tokens}/{max_tokens} tokens ({usage_ratio:.1%})")

        # 압축 필요 여부 확인
        if usage_ratio > self.config.compression_threshold and self.config.enable_auto_compression:
            await self._compress_context(max_tokens)

        # 최적화된 컨텍스트 생성
        optimized_context = await self._build_optimized_context(max_tokens)

        # 통계 기록
        stats = {
            'total_chunks': len(self.context_chunks),
            'total_tokens': total_tokens,
            'available_tokens': max_tokens,
            'usage_ratio': usage_ratio,
            'compressed_chunks': sum(1 for c in self.context_chunks if c.compressed),
            'optimization_applied': usage_ratio > self.config.compression_threshold,
            'cached': False
        }

        self.token_usage_history.append({
            'timestamp': time.time(),
            'stats': stats,
            'optimized_tokens': sum(len(c.content.split()) * 1.3 for c in optimized_context['chunks'])
        })

        # 캐시에 저장
        if use_cache:
            try:
                from src.core.context_cache import get_context_cache
                context_cache = get_context_cache()
                chunks_data = [
                    {
                        "content": chunk.content,
                        "content_type": chunk.content_type.value,
                        "priority": chunk.priority.value,
                        "token_count": chunk.token_count
                    }
                    for chunk in optimized_context['chunks']
                ]
                context_cache.put(
                    cache_key=cache_key,
                    context_chunks=chunks_data,
                    token_allocation=optimized_context['allocation'],
                    query=query
                )
            except Exception as e:
                logger.debug(f"Cache store failed: {e}")

        logger.info(f"Context optimized: {stats['compressed_chunks']} chunks compressed, "
                   f"final ratio: {stats['usage_ratio']:.1%}")

        return {
            'chunks': optimized_context['chunks'],
            'stats': stats,
            'allocation': optimized_context['allocation']
        }
    
    async def load_context_progressively(
        self,
        initial_tokens: int,
        expand_tokens: int,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        점진적 컨텍스트 로딩.
        
        Args:
            initial_tokens: 초기 로딩 토큰 수
            expand_tokens: 확장 시 추가 토큰 수
            max_tokens: 최대 토큰 수
            
        Returns:
            점진적 로딩 결과
        """
        try:
            # 1단계: 핵심 컨텍스트만 로딩
            core_chunks = await self._select_core_chunks(initial_tokens)
            
            result = {
                'stage': 'initial',
                'chunks': core_chunks,
                'tokens_used': sum(chunk.token_count for chunk in core_chunks),
                'can_expand': len(core_chunks) < len(self.context_chunks)
            }
            
            logger.info(f"Progressive loading stage 1: {len(core_chunks)} core chunks loaded")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load context progressively: {e}")
            return {'stage': 'error', 'chunks': [], 'tokens_used': 0, 'can_expand': False}
    
    async def expand_context(
        self,
        current_chunks: List[ContextChunk],
        additional_tokens: int
    ) -> List[ContextChunk]:
        """
        컨텍스트 확장.
        
        Args:
            current_chunks: 현재 로드된 청크
            additional_tokens: 추가 토큰 수
            
        Returns:
            확장된 청크 목록
        """
        try:
            current_keys = {id(chunk) for chunk in current_chunks}
            remaining_chunks = [
                chunk for chunk in self.context_chunks
                if id(chunk) not in current_keys
            ]
            
            # 우선순위별 정렬
            remaining_chunks.sort(key=lambda c: (-c.priority.value, -c.timestamp))
            
            # 추가 토큰 내에서 선택
            expanded = []
            used_tokens = 0
            
            for chunk in remaining_chunks:
                if used_tokens + chunk.token_count <= additional_tokens:
                    expanded.append(chunk)
                    used_tokens += chunk.token_count
                else:
                    break
            
            logger.info(f"Context expanded: {len(expanded)} additional chunks")
            return current_chunks + expanded
            
        except Exception as e:
            logger.error(f"Failed to expand context: {e}")
            return current_chunks
    
    async def _select_core_chunks(self, max_tokens: int) -> List[ContextChunk]:
        """핵심 청크 선택 (초기 로딩용)."""
        # CRITICAL 및 HIGH 우선순위만 선택
        core_chunks = [
            chunk for chunk in self.context_chunks
            if chunk.priority.value >= ContextPriority.HIGH.value
        ]
        
        # 토큰 제한 내에서 선택
        selected = []
        used_tokens = 0
        
        core_chunks.sort(key=lambda c: (-c.priority.value, -c.timestamp))
        
        for chunk in core_chunks:
            if used_tokens + chunk.token_count <= max_tokens:
                selected.append(chunk)
                used_tokens += chunk.token_count
            else:
                # 부분 포함 (압축)
                if chunk.priority == ContextPriority.CRITICAL:
                    compressed = chunk.compress(max_tokens - used_tokens)
                    if compressed.token_count > 0:
                        selected.append(compressed)
                break
        
        return selected

    async def _compress_context(self, max_tokens: int):
        """
        컨텍스트 압축 (백서 요구사항: Context Rot 방지).
        
        전략:
        1. Selective pruning: 낮은 우선순위 청크 제거
        2. Recursive summarization: 재귀적 요약
        3. Token-based truncation: 토큰 기반 잘라내기
        """
        logger.info("Starting context compression (Context Rot prevention)...")

        # 1. Selective Pruning: 낮은 우선순위 청크 제거
        current_tokens = sum(c.token_count for c in self.context_chunks)
        target_tokens = int(max_tokens * 0.7)  # 70% 목표
        
        if current_tokens > target_tokens:
            # 우선순위별 정렬 (낮은 우선순위부터)
            sorted_chunks = sorted(
                self.context_chunks,
                key=lambda c: (c.priority.value, -c.timestamp)
            )
            
            # OPTIONAL 및 LOW 우선순위 청크 제거
            chunks_to_remove = []
            for chunk in sorted_chunks:
                if current_tokens <= target_tokens:
                    break
                
                if chunk.priority in [ContextPriority.OPTIONAL, ContextPriority.LOW]:
                    chunks_to_remove.append(chunk)
                    current_tokens -= chunk.token_count
                    self.compression_stats['chunks_pruned'] = self.compression_stats.get('chunks_pruned', 0) + 1
            
            # 제거
            for chunk in chunks_to_remove:
                self.context_chunks.remove(chunk)
            
            logger.debug(f"Selective pruning: {len(chunks_to_remove)} chunks removed")

        # 2. Recursive Summarization: 재귀적 요약
        if current_tokens > target_tokens:
            # MEDIUM 우선순위 청크 요약
            for chunk in self.context_chunks:
                if current_tokens <= target_tokens:
                    break
                
                if chunk.priority == ContextPriority.MEDIUM and not chunk.compressed:
                    original_tokens = chunk.token_count
                    # 50% 압축 시도
                    compressed_chunk = chunk.compress(int(chunk.token_count * 0.5))
                    
                    if compressed_chunk.token_count < original_tokens:
                        idx = self.context_chunks.index(chunk)
                        self.context_chunks[idx] = compressed_chunk
                        current_tokens -= (original_tokens - compressed_chunk.token_count)
                        
                        self.compression_stats['chunks_compressed'] = self.compression_stats.get('chunks_compressed', 0) + 1
                        self.compression_stats['tokens_saved'] = self.compression_stats.get('tokens_saved', 0) + (original_tokens - compressed_chunk.token_count)
                        
                        logger.debug(f"Recursive summarization: {original_tokens} -> {compressed_chunk.token_count} tokens")

        # 3. Token-based Truncation: 토큰 기반 잘라내기
        if current_tokens > max_tokens:
            # 남은 토큰을 각 청크에 할당
            remaining_tokens = max_tokens
            for chunk in sorted(self.context_chunks, key=lambda c: -c.priority.value):
                if remaining_tokens <= 0:
                    # 더 이상 공간이 없으면 제거
                    self.context_chunks.remove(chunk)
                    continue
                
                if chunk.token_count > remaining_tokens:
                    # 잘라내기
                    truncated_chunk = chunk.compress(remaining_tokens)
                    idx = self.context_chunks.index(chunk)
                    self.context_chunks[idx] = truncated_chunk
                    remaining_tokens = 0
                else:
                    remaining_tokens -= chunk.token_count

        logger.info(f"Context compression completed: "
                   f"{self.compression_stats.get('chunks_pruned', 0)} pruned, "
                   f"{self.compression_stats.get('chunks_compressed', 0)} compressed, "
                   f"{self.compression_stats.get('tokens_saved', 0)} tokens saved")

    async def _build_optimized_context(self, max_tokens: int) -> Dict[str, Any]:
        """최적화된 컨텍스트 구축."""
        # 토큰 할당 계산
        allocation = self._calculate_token_allocation(max_tokens)

        # 유형별 청크 분류
        chunks_by_type = defaultdict(list)
        for chunk in self.context_chunks:
            chunks_by_type[chunk.content_type].append(chunk)

        # 각 유형별 최적화
        optimized_chunks = []

        for content_type, chunks in chunks_by_type.items():
            allocated_tokens = allocation.get(content_type.value, 0)
            if allocated_tokens > 0:
                selected_chunks = await self._select_chunks_for_type(chunks, allocated_tokens)
                optimized_chunks.extend(selected_chunks)

        # 우선순위별 정렬 (중요한 것 먼저)
        optimized_chunks.sort(key=lambda c: (-c.priority.value, -c.timestamp))

        return {
            'chunks': optimized_chunks,
            'allocation': allocation
        }

    def _calculate_token_allocation(self, max_tokens: int) -> Dict[str, int]:
        """토큰 할당 계산 (의존성 그래프 및 스마트 조립 지원)."""
        # MemoryLearner에서 추천 조합 가져오기 시도
        try:
            from src.core.memory_learner import get_memory_learner
            memory_learner = get_memory_learner()
            
            # 현재 컨텍스트 타입 추출
            context_types = [chunk.content_type.value for chunk in self.context_chunks]
            unique_types = list(set(context_types))
            
            # 추천 조합 가져오기
            recommended = memory_learner.get_recommended_combination(unique_types, max_tokens)
            if recommended and recommended.token_allocation:
                logger.debug("Using learned token allocation pattern")
                return recommended.token_allocation
        except Exception as e:
            logger.debug(f"MemoryLearner not available or failed: {e}")
        
        if not self.config.adaptive_allocation:
            # 고정 할당
            return {
                'system_prompt': self.config.system_prompt_tokens,
                'user_query': int(max_tokens * 0.05),
                'tool_results': int(max_tokens * 0.4),
                'agent_memory': int(max_tokens * 0.2),
                'conversation_history': int(max_tokens * 0.25),
                'metadata': self.config.metadata_tokens
            }

        # 적응형 할당 (의존성 그래프 고려)
        allocation = {}

        # 시스템 프롬프트 고정
        allocation['system_prompt'] = min(self.config.system_prompt_tokens, int(max_tokens * 0.15))

        # 도구 설명 고정
        allocation['tool_descriptions'] = min(self.config.tool_descriptions_tokens, int(max_tokens * 0.2))

        # 남은 토큰 분배 (의존성 고려)
        remaining_tokens = max_tokens - allocation['system_prompt'] - allocation['tool_descriptions']

        # 의존성 그래프 기반 가중치 계산
        type_weights = self._calculate_type_weights_with_dependencies()

        # 유형별 청크 수에 따라 동적 할당
        type_counts = defaultdict(int)
        for chunk in self.context_chunks:
            type_counts[chunk.content_type] += 1

        total_chunks = sum(type_counts.values())

        if total_chunks > 0:
            total_weight = sum(type_weights.get(ct.value, 1.0) for ct in type_counts.keys())
            for content_type, count in type_counts.items():
                if content_type.value not in ['system_prompt', 'tool_descriptions']:
                    weight = type_weights.get(content_type.value, 1.0)
                    ratio = (count / total_chunks) * (weight / total_weight) if total_weight > 0 else count / total_chunks
                    allocation[content_type.value] = int(remaining_tokens * ratio)

        return allocation
    
    def _calculate_type_weights_with_dependencies(self) -> Dict[str, float]:
        """의존성 그래프를 고려한 타입별 가중치 계산."""
        weights = defaultdict(float)
        
        # 기본 가중치
        default_weights = {
            'user_query': 1.0,
            'system_prompt': 1.0,
            'tool_results': 0.8,
            'agent_memory': 0.6,
            'conversation_history': 0.7,
            'metadata': 0.3
        }
        
        # 의존성 기반 가중치 조정
        for chunk in self.context_chunks:
            chunk_type = chunk.content_type.value
            base_weight = default_weights.get(chunk_type, 0.5)
            
            # 의존성이 많을수록 가중치 증가
            dependency_bonus = len(chunk.dependencies) * 0.1
            weights[chunk_type] = max(weights[chunk_type], base_weight + dependency_bonus)
        
        return dict(weights)

    async def _select_chunks_for_type(self, chunks: List[ContextChunk], allocated_tokens: int) -> List[ContextChunk]:
        """유형별 청크 선택."""
        if not chunks:
            return []

        # 우선순위별 정렬
        sorted_chunks = sorted(chunks, key=lambda c: (-c.priority.value, -c.timestamp))

        selected_chunks = []
        used_tokens = 0

        for chunk in sorted_chunks:
            if used_tokens + chunk.token_count <= allocated_tokens:
                selected_chunks.append(chunk)
                used_tokens += chunk.token_count
            elif chunk.priority == ContextPriority.CRITICAL:
                # CRITICAL은 강제 포함 (압축)
                compressed_chunk = chunk.compress(allocated_tokens - used_tokens)
                if compressed_chunk.token_count > 0:
                    selected_chunks.append(compressed_chunk)
                    used_tokens += compressed_chunk.token_count

        return selected_chunks

    async def exchange_context_with_sub_agent(
        self,
        sub_agent_id: str,
        shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        서브 에이전트와 컨텍스트 교환.

        Args:
            sub_agent_id: 서브 에이전트 ID
            shared_context: 공유할 컨텍스트

        Returns:
            교환 결과
        """
        # 공유 컨텍스트를 청크로 변환
        shared_chunks = []
        for key, value in shared_context.items():
            if isinstance(value, (str, int, float, bool)):
                content = str(value)
            elif isinstance(value, (list, dict)):
                content = json.dumps(value, ensure_ascii=False)
            else:
                continue

            chunk = ContextChunk(
                content=content,
                content_type=ContextType.AGENT_MEMORY,
                priority=ContextPriority.HIGH,
                timestamp=time.time(),
                metadata={'source': sub_agent_id, 'key': key}
            )
            shared_chunks.append(chunk)

        # 중복 제거 및 통합
        await self._merge_shared_chunks(shared_chunks)

        # 응답 컨텍스트 생성
        response_context = await self._generate_response_context()

        logger.info(f"Context exchanged with sub-agent {sub_agent_id}: "
                   f"{len(shared_chunks)} chunks received, "
                   f"{len(response_context.get('chunks', []))} chunks sent")

        return response_context

    async def _merge_shared_chunks(self, shared_chunks: List[ContextChunk]):
        """공유 청크 통합."""
        for shared_chunk in shared_chunks:
            # 중복 확인
            duplicate_found = False
            for existing_chunk in self.context_chunks:
                if (existing_chunk.content_type == shared_chunk.content_type and
                    existing_chunk.metadata.get('key') == shared_chunk.metadata.get('key') and
                    abs(existing_chunk.timestamp - shared_chunk.timestamp) < 60):  # 1분 내
                    # 최신 것으로 업데이트
                    if shared_chunk.timestamp > existing_chunk.timestamp:
                        idx = self.context_chunks.index(existing_chunk)
                        self.context_chunks[idx] = shared_chunk
                    duplicate_found = True
                    break

            if not duplicate_found:
                self.context_chunks.append(shared_chunk)

    async def _generate_response_context(self) -> Dict[str, Any]:
        """응답 컨텍스트 생성."""
        # 중요한 청크들만 선택
        important_chunks = [
            chunk for chunk in self.context_chunks
            if chunk.priority in [ContextPriority.CRITICAL, ContextPriority.HIGH]
        ]

        # 최근 청크 우선 (최대 10개)
        recent_chunks = sorted(important_chunks, key=lambda c: -c.timestamp)[:10]

        return {
            'chunks': recent_chunks,
            'summary': {
                'total_chunks': len(self.context_chunks),
                'shared_chunks': len(recent_chunks),
                'compression_applied': any(c.compressed for c in recent_chunks)
            }
        }

    def get_context_stats(self) -> Dict[str, Any]:
        """컨텍스트 통계."""
        type_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        total_tokens = 0
        compressed_count = 0

        for chunk in self.context_chunks:
            type_counts[chunk.content_type] += 1
            priority_counts[chunk.priority] += 1
            total_tokens += chunk.token_count
            if chunk.compressed:
                compressed_count += 1

        return {
            'total_chunks': len(self.context_chunks),
            'total_tokens': total_tokens,
            'compressed_chunks': compressed_count,
            'type_distribution': dict(type_counts),
            'priority_distribution': dict(priority_counts),
            'compression_stats': dict(self.compression_stats),
            'token_usage_history': self.token_usage_history[-10:]  # 최근 10개
        }

    async def cleanup_old_context(self, max_age_hours: int = 24):
        """오래된 컨텍스트 정리."""
        cutoff_time = time.time() - (max_age_hours * 3600)

        original_count = len(self.context_chunks)
        self.context_chunks = [
            chunk for chunk in self.context_chunks
            if chunk.timestamp > cutoff_time or chunk.priority == ContextPriority.CRITICAL
        ]

        removed_count = original_count - len(self.context_chunks)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old context chunks (>{max_age_hours}h)")

        return removed_count


# 전역 컨텍스트 엔지니어 인스턴스
_context_engineer = None

def get_context_engineer() -> ContextEngineer:
    """전역 컨텍스트 엔지니어 인스턴스 반환."""
    global _context_engineer
    if _context_engineer is None:
        _context_engineer = ContextEngineer()
    return _context_engineer

def set_context_engineer(engineer: ContextEngineer):
    """전역 컨텍스트 엔지니어 설정."""
    global _context_engineer
    _context_engineer = engineer
