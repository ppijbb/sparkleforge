"""Context compaction configuration (shared to avoid circular imports)."""

from dataclasses import dataclass


@dataclass
class CompactionConfig:
    """압축 설정."""

    trigger_threshold: float = 0.85  # 85% 임계값
    target_tokens: int = 0  # 목표 토큰 수 (0이면 자동 계산)
    preserve_last_n: int = 20  # 마지막 N개 메시지 보호
    preserve_last_tokens: int = 40000  # 마지막 40K 토큰 보호 (AgentPG 패턴)
    strategy: str = "hybrid"  # "prune" | "summarize" | "hybrid" | "forgetting" | "condensation"
    auto_compact: bool = True  # 자동 압축 활성화
    max_context_tokens: int = 200000  # 최대 컨텍스트 토큰 수
    forget_ttl_seconds: int | None = None  # Forgetting: 이 초보다 오래된 메시지 제거
    importance_threshold: float = 0.0  # Forgetting: 이 값 미만 importance 메시지 제거
    # V-1: 장기 컨텍스트 일반화 — 연구 모드뿐 아니라 work/coworker, 코딩 에이전트 등
    # 모든 에이전트 유형에서 동일한 컨텍스트 관리 정책을 적용하기 위한 축.
    # "research" | "work" | "coworker" | "coding" | "general"
    agent_mode: str = "general"
    # 에이전트 모드별 보호 우선순위: 코딩/워크 에이전트는 최근 코드 변경 맥락을 더 강하게 보존.
    preserve_recent_multiplier: float = 1.0
