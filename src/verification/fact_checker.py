#!/usr/bin/env python3
"""
Fact Checker for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade 팩트 체킹 및 교차 검증 시스템.
여러 출처 간 내용 일치성 검증, 상충되는 정보 탐지,
LLM 기반 사실 확인을 통해 정보의 신뢰성을 보장합니다.

2025년 10월 최신 기술 스택:
- Gemini 2.5 Flash Lite for fact checking
- MCP tools for external verification
- Cross-reference validation
- Production-grade reliability patterns
"""

import asyncio
import logging
import json
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import difflib
from collections import defaultdict

# MCP integration
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.mcp_integration import execute_tool
from src.core.llm_manager import execute_llm_task, TaskType
from src.core.reliability import execute_with_reliability
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class VerificationStage(Enum):
    """검증 단계."""
    SELF = "self"
    CROSS = "cross"
    EXTERNAL = "external"


class FactStatus(Enum):
    """팩트 상태."""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    CONFLICTING = "conflicting"
    UNVERIFIED = "unverified"
    DISPUTED = "disputed"


@dataclass
class FactCheckResult:
    """팩트 체크 결과."""
    fact_id: str
    original_text: str
    verification_stage: VerificationStage
    fact_status: FactStatus
    confidence_score: float  # 0.0-1.0
    supporting_sources: List[str]
    conflicting_sources: List[str]
    verification_details: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CrossReferenceResult:
    """교차 참조 결과."""
    fact_id: str
    source_matches: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]]
    consensus_score: float
    verification_confidence: float


class FactChecker:
    """
    Production-grade 팩트 체킹 및 교차 검증 시스템.
    
    Features:
    - 여러 출처 간 내용 일치성 검증
    - 상충되는 정보 탐지 및 플래깅
    - LLM 기반 사실 확인 (Gemini 2.5 Flash Lite 활용)
    - 불확실한 정보에 대한 신뢰도 하향 조정
    """
    
    def __init__(self):
        """팩트 체커 초기화."""
        self.verification_cache = {}  # 검증 결과 캐시
        self.conflict_patterns = self._initialize_conflict_patterns()
        self.verification_thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        }
        
        logger.info("FactChecker initialized with production-grade reliability")
    
    def _initialize_conflict_patterns(self) -> List[str]:
        """상충 패턴 초기화."""
        return [
            r'however|but|although|despite|in contrast',
            r'disagrees?|contradicts?|conflicts?',
            r'according to.*but|while.*suggests?',
            r'studies? show.*however|research indicates.*but',
            r'experts? say.*while|scientists? believe.*however'
        ]
    
    async def verify_fact(
        self,
        fact_text: str,
        sources: List[Dict[str, Any]],
        fact_id: Optional[str] = None
    ) -> FactCheckResult:
        """
        팩트를 검증합니다.
        
        Args:
            fact_text: 검증할 팩트 텍스트
            sources: 관련 출처들
            fact_id: 팩트 ID (선택사항)
            
        Returns:
            FactCheckResult: 검증 결과
        """
        if not fact_id:
            fact_id = f"fact_{int(datetime.now().timestamp())}"
        
        logger.info(f"Verifying fact: {fact_id}")
        
        try:
            # 1단계: Self-Verification (에이전트 자체 검증)
            self_result = await self._self_verification(fact_text, sources)
            
            # 2단계: Cross-Verification (다른 에이전트와 교차 검증)
            cross_result = await self._cross_verification(fact_text, sources)
            
            # 3단계: External-Verification (외부 출처와 대조)
            external_result = await self._external_verification(fact_text, sources)
            
            # 종합 결과 생성
            final_result = self._synthesize_verification_results(
                fact_id, fact_text, self_result, cross_result, external_result
            )
            
            # 캐시에 저장
            self.verification_cache[fact_id] = final_result
            
            logger.info(f"Fact verification completed: {fact_id} - Status: {final_result.fact_status.value}")
            return final_result
            
        except Exception as e:
            logger.error(f"Fact verification failed for {fact_id}: {e}")
            return self._create_failed_result(fact_id, fact_text, str(e))
    
    async def _self_verification(
        self,
        fact_text: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Self-Verification: 에이전트 자체 검증."""
        try:
            # LLM을 사용한 내부 일관성 검사
            consistency_prompt = f"""
            Analyze the following fact for internal consistency and logical validity:
            
            Fact: "{fact_text}"
            
            Sources: {json.dumps(sources, ensure_ascii=False, indent=2)}
            
            Evaluate:
            1. Internal logical consistency
            2. Factual plausibility
            3. Source reliability correlation
            4. Potential red flags or inconsistencies
            
            Return JSON format:
            {{
                "consistency_score": 0.0-1.0,
                "plausibility_score": 0.0-1.0,
                "red_flags": ["list of potential issues"],
                "confidence": 0.0-1.0,
                "reasoning": "detailed analysis"
            }}
            """
            
            result = await execute_llm_task(
                prompt=consistency_prompt,
                task_type=TaskType.ANALYSIS,
                system_message="You are an expert fact-checker specializing in logical consistency analysis."
            )
            
            content = result.content or ""
            if not content or not content.strip():
                logger.warning("Self-verification failed: Empty response from LLM")
                raise ValueError("Empty response from LLM")
            
            # JSON 추출 시도
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # JSON 블록이 없으면 전체를 파싱 시도
                try:
                    analysis = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(f"Self-verification failed: Invalid JSON format in response")
                    raise ValueError("Invalid JSON format")
            
            return {
                'stage': VerificationStage.SELF.value,
                'consistency_score': analysis.get('consistency_score', 0.5),
                'plausibility_score': analysis.get('plausibility_score', 0.5),
                'red_flags': analysis.get('red_flags', []),
                'confidence': analysis.get('confidence', 0.5),
                'reasoning': analysis.get('reasoning', ''),
                'model_used': result.model_used
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Self-verification failed: {e}")
            raise
        except Exception as e:
            logger.warning(f"Self-verification failed: {e}")
            raise
            return {
                'stage': VerificationStage.SELF.value,
                'consistency_score': 0.5,
                'plausibility_score': 0.5,
                'red_flags': [],
                'confidence': 0.3,
                'reasoning': f'Verification failed: {e}',
                'model_used': 'unknown'
            }
    
    async def _cross_verification(
        self,
        fact_text: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Cross-Verification: 다른 에이전트와 교차 검증."""
        try:
            # 여러 출처 간 내용 비교
            cross_ref_result = await self._cross_reference_sources(fact_text, sources)
            
            # 상충 정보 탐지
            conflicts = await self._detect_conflicts(fact_text, sources)
            
            # 일치도 계산
            consensus_score = self._calculate_consensus_score(cross_ref_result, conflicts)
            
            return {
                'stage': VerificationStage.CROSS.value,
                'consensus_score': consensus_score,
                'conflicts': conflicts,
                'cross_references': cross_ref_result,
                'confidence': min(1.0, consensus_score * 1.2),
                'reasoning': f'Cross-verified across {len(sources)} sources'
            }
            
        except Exception as e:
            logger.warning(f"Cross-verification failed: {e}")
            return {
                'stage': VerificationStage.CROSS.value,
                'consensus_score': 0.5,
                'conflicts': [],
                'cross_references': [],
                'confidence': 0.3,
                'reasoning': f'Cross-verification failed: {e}'
            }
    
    async def _external_verification(
        self,
        fact_text: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """External-Verification: 외부 출처와 대조."""
        try:
            # MCP 도구를 사용한 외부 검증
            external_sources = await self._search_external_sources(fact_text)
            
            # 외부 출처와의 일치도 분석
            external_match_score = await self._analyze_external_matches(
                fact_text, external_sources
            )
            
            # 신뢰할 수 있는 외부 출처 확인
            trusted_verification = await self._verify_with_trusted_sources(
                fact_text, external_sources
            )
            
            return {
                'stage': VerificationStage.EXTERNAL.value,
                'external_match_score': external_match_score,
                'trusted_verification': trusted_verification,
                'external_sources_count': len(external_sources),
                'confidence': (external_match_score + trusted_verification) / 2,
                'reasoning': f'Verified against {len(external_sources)} external sources'
            }
            
        except Exception as e:
            logger.warning(f"External verification failed: {e}")
            return {
                'stage': VerificationStage.EXTERNAL.value,
                'external_match_score': 0.5,
                'trusted_verification': 0.5,
                'external_sources_count': 0,
                'confidence': 0.3,
                'reasoning': f'External verification failed: {e}'
            }
    
    async def _cross_reference_sources(
        self,
        fact_text: str,
        sources: List[Dict[str, Any]]
    ) -> CrossReferenceResult:
        """출처 간 교차 참조를 수행합니다."""
        try:
            source_matches = []
            conflicts = []
            
            # 각 출처의 내용을 비교
            for i, source1 in enumerate(sources):
                for j, source2 in enumerate(sources[i+1:], i+1):
                    similarity = self._calculate_text_similarity(
                        source1.get('content', ''),
                        source2.get('content', '')
                    )
                    
                    if similarity > 0.7:
                        source_matches.append({
                            'source1': source1.get('url', f'source_{i}'),
                            'source2': source2.get('url', f'source_{j}'),
                            'similarity': similarity
                        })
                    elif similarity < 0.3:
                        conflicts.append({
                            'source1': source1.get('url', f'source_{i}'),
                            'source2': source2.get('url', f'source_{j}'),
                            'similarity': similarity,
                            'conflict_type': 'content_mismatch'
                        })
            
            consensus_score = self._calculate_consensus_from_matches(source_matches, conflicts)
            
            return CrossReferenceResult(
                fact_id=f"cross_ref_{int(datetime.now().timestamp())}",
                source_matches=source_matches,
                conflicts=conflicts,
                consensus_score=consensus_score,
                verification_confidence=min(1.0, consensus_score * 1.1)
            )
            
        except Exception as e:
            logger.warning(f"Cross-reference failed: {e}")
            return CrossReferenceResult(
                fact_id="failed",
                source_matches=[],
                conflicts=[],
                consensus_score=0.5,
                verification_confidence=0.3
            )
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도를 계산합니다."""
        if not text1 or not text2:
            return 0.0
        
        # difflib를 사용한 시퀀스 매칭
        matcher = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
        return matcher.ratio()
    
    async def _detect_conflicts(self, fact_text: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """상충 정보를 탐지합니다."""
        conflicts = []
        
        try:
            # LLM을 사용한 상충 탐지
            conflict_prompt = f"""
            Analyze the following fact and sources for potential conflicts or contradictions:
            
            Fact: "{fact_text}"
            
            Sources: {json.dumps(sources, ensure_ascii=False, indent=2)}
            
            Identify:
            1. Direct contradictions between sources
            2. Conflicting data or statistics
            3. Opposing viewpoints or interpretations
            4. Temporal inconsistencies
            
            Return JSON format:
            {{
                "conflicts": [
                    {{
                        "type": "contradiction|data_conflict|viewpoint_conflict|temporal_conflict",
                        "description": "description of the conflict",
                        "sources_involved": ["source1", "source2"],
                        "severity": "low|medium|high",
                        "confidence": 0.0-1.0
                    }}
                ],
                "overall_conflict_score": 0.0-1.0
            }}
            """
            
            result = await execute_llm_task(
                prompt=conflict_prompt,
                task_type=TaskType.ANALYSIS,
                system_message="You are an expert at detecting conflicts and contradictions in information."
            )
            
            content = result.content or ""
            if not content or not content.strip():
                logger.warning("Conflict detection failed: Empty response from LLM")
                return []
            
            # JSON 추출 시도
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # JSON 블록이 없으면 전체를 파싱 시도
                try:
                    analysis = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(f"Conflict detection failed: Invalid JSON format in response")
                    return []
            
            conflicts = analysis.get('conflicts', [])
            
        except json.JSONDecodeError as e:
            logger.warning(f"Conflict detection failed: JSON decode error - {e}")
        except Exception as e:
            logger.warning(f"Conflict detection failed: {e}")
        
        return conflicts
    
    async def _search_external_sources(self, fact_text: str) -> List[Dict[str, Any]]:
        """외부 출처를 검색합니다."""
        try:
            # MCP 도구를 사용한 외부 검색
            search_result = await execute_tool("g-search", {
                "query": fact_text,
                "max_results": 5
            })
            
            if search_result.get('success', False):
                return search_result.get('data', {}).get('results', [])
            
            return []
            
        except Exception as e:
            logger.warning(f"External source search failed: {e}")
            return []
    
    async def _analyze_external_matches(
        self,
        fact_text: str,
        external_sources: List[Dict[str, Any]]
    ) -> float:
        """외부 출처와의 일치도를 분석합니다."""
        if not external_sources:
            return 0.5
        
        total_similarity = 0.0
        valid_sources = 0
        
        for source in external_sources:
            content = source.get('content', '') or source.get('snippet', '')
            if content:
                similarity = self._calculate_text_similarity(fact_text, content)
                total_similarity += similarity
                valid_sources += 1
        
        return total_similarity / valid_sources if valid_sources > 0 else 0.5
    
    async def _verify_with_trusted_sources(
        self,
        fact_text: str,
        external_sources: List[Dict[str, Any]]
    ) -> float:
        """신뢰할 수 있는 출처로 검증합니다."""
        trusted_domains = [
            'edu', 'gov', 'org', 'reuters.com', 'bbc.com', 'ap.org',
            'nature.com', 'science.org', 'scholar.google.com'
        ]
        
        trusted_matches = 0
        total_trusted = 0
        
        for source in external_sources:
            url = source.get('url', '')
            if any(domain in url for domain in trusted_domains):
                total_trusted += 1
                content = source.get('content', '') or source.get('snippet', '')
                if content:
                    similarity = self._calculate_text_similarity(fact_text, content)
                    if similarity > 0.6:
                        trusted_matches += 1
        
        return trusted_matches / total_trusted if total_trusted > 0 else 0.5
    
    def _calculate_consensus_score(
        self,
        cross_ref_result: CrossReferenceResult,
        conflicts: List[Dict[str, Any]]
    ) -> float:
        """합의 점수를 계산합니다."""
        base_score = cross_ref_result.consensus_score
        
        # 상충이 있으면 점수 감소
        if conflicts:
            conflict_penalty = min(0.3, len(conflicts) * 0.1)
            base_score -= conflict_penalty
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_consensus_from_matches(
        self,
        matches: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]]
    ) -> float:
        """매치 결과에서 합의 점수를 계산합니다."""
        if not matches and not conflicts:
            return 0.5
        
        if not matches:
            return 0.2  # 상충만 있는 경우
        
        # 매치의 평균 유사도
        avg_similarity = sum(match['similarity'] for match in matches) / len(matches)
        
        # 상충 페널티
        conflict_penalty = min(0.4, len(conflicts) * 0.1)
        
        return max(0.0, min(1.0, avg_similarity - conflict_penalty))
    
    def _synthesize_verification_results(
        self,
        fact_id: str,
        fact_text: str,
        self_result: Dict[str, Any],
        cross_result: Dict[str, Any],
        external_result: Dict[str, Any]
    ) -> FactCheckResult:
        """검증 결과를 종합합니다."""
        # 가중 평균으로 최종 신뢰도 계산
        weights = {'self': 0.3, 'cross': 0.4, 'external': 0.3}
        
        final_confidence = (
            weights['self'] * self_result.get('confidence', 0.5) +
            weights['cross'] * cross_result.get('confidence', 0.5) +
            weights['external'] * external_result.get('confidence', 0.5)
        )
        
        # 팩트 상태 결정
        fact_status = self._determine_fact_status(
            final_confidence,
            cross_result.get('consensus_score', 0.5),
            len(cross_result.get('conflicts', []))
        )
        
        # 지원 출처 및 상충 출처 수집
        supporting_sources = []
        conflicting_sources = []
        
        if cross_result.get('cross_references'):
            for ref in cross_result['cross_references']:
                if ref.get('similarity', 0) > 0.7:
                    supporting_sources.append(ref.get('source1', ''))
                    supporting_sources.append(ref.get('source2', ''))
        
        if cross_result.get('conflicts'):
            for conflict in cross_result['conflicts']:
                conflicting_sources.append(conflict.get('source1', ''))
                conflicting_sources.append(conflict.get('source2', ''))
        
        return FactCheckResult(
            fact_id=fact_id,
            original_text=fact_text,
            verification_stage=VerificationStage.EXTERNAL,  # 최종 단계
            fact_status=fact_status,
            confidence_score=final_confidence,
            supporting_sources=list(set(supporting_sources)),
            conflicting_sources=list(set(conflicting_sources)),
            verification_details={
                'self_verification': self_result,
                'cross_verification': cross_result,
                'external_verification': external_result
            },
            timestamp=datetime.now(timezone.utc),
            metadata={
                'verification_method': 'comprehensive',
                'total_sources_checked': len(supporting_sources) + len(conflicting_sources)
            }
        )
    
    def _determine_fact_status(
        self,
        confidence: float,
        consensus_score: float,
        conflict_count: int
    ) -> FactStatus:
        """팩트 상태를 결정합니다."""
        if conflict_count > 2:
            return FactStatus.DISPUTED
        elif confidence >= self.verification_thresholds['high_confidence'] and consensus_score > 0.8:
            return FactStatus.VERIFIED
        elif confidence >= self.verification_thresholds['medium_confidence'] and consensus_score > 0.6:
            return FactStatus.PARTIALLY_VERIFIED
        elif conflict_count > 0:
            return FactStatus.CONFLICTING
        else:
            return FactStatus.UNVERIFIED
    
    def _create_failed_result(
        self,
        fact_id: str,
        fact_text: str,
        error: str
    ) -> FactCheckResult:
        """실패한 검증 결과를 생성합니다."""
        return FactCheckResult(
            fact_id=fact_id,
            original_text=fact_text,
            verification_stage=VerificationStage.SELF,
            fact_status=FactStatus.UNVERIFIED,
            confidence_score=0.0,
            supporting_sources=[],
            conflicting_sources=[],
            verification_details={'error': error},
            timestamp=datetime.now(timezone.utc),
            metadata={'verification_failed': True}
        )
    
    async def batch_verify_facts(
        self,
        facts: List[Tuple[str, List[Dict[str, Any]]]]
    ) -> List[FactCheckResult]:
        """여러 팩트를 배치로 검증합니다."""
        tasks = [
            self.verify_fact(fact_text, sources, f"fact_{i}")
            for i, (fact_text, sources) in enumerate(facts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외 처리
        verified_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch verification failed for fact {i}: {result}")
                verified_results.append(
                    self._create_failed_result(f"fact_{i}", facts[i][0], str(result))
                )
            else:
                verified_results.append(result)
        
        return verified_results
