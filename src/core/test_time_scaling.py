"""Test-Time Scaling with Last-K-Fusion

DeepResearch 영감을 받은 테스트 타임 스케일링 시스템.
병렬 rollout 실행 후 융합 에이전트로 최적 결과 합성.

핵심 특징:
- Parallel rollout execution with diverse parameters
- Last-K fusion for result synthesis
- Confidence-weighted result selection
- Rollout diversity guarantee
"""

import asyncio
import hashlib
import logging
import random
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RolloutStatus(Enum):
    """Rollout 상태."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class FusionStrategy(Enum):
    """융합 전략."""

    BEST_CONFIDENCE = "best_confidence"  # 가장 높은 confidence 선택
    WEIGHTED_AVERAGE = "weighted_average"  # 가중 평균
    MAJORITY_VOTE = "majority_vote"  # 다수결
    LAST_K_FUSION = "last_k_fusion"  # Last-K 융합 (LLM 기반)
    ENSEMBLE = "ensemble"  # 앙상블 조합


class RolloutConfig(BaseModel):
    """Rollout 설정."""

    rollout_id: str = Field(description="Rollout ID")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-P")
    top_k: int = Field(default=40, ge=1, description="Top-K")
    max_tokens: int = Field(default=4096, description="최대 토큰")
    seed: int | None = Field(default=None, description="랜덤 시드")

    # 다양성 제어
    diversity_factor: float = Field(default=0.0, description="다양성 팩터")

    class Config:
        arbitrary_types_allowed = True


class RolloutResult(BaseModel):
    """Rollout 결과."""

    rollout_id: str = Field(description="Rollout ID")
    config: RolloutConfig = Field(description="사용된 설정")
    status: RolloutStatus = Field(default=RolloutStatus.PENDING)

    # 결과
    output: str = Field(default="", description="출력 텍스트")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="신뢰도")

    # 메타데이터
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    duration_seconds: float = Field(default=0.0)
    tokens_used: int = Field(default=0)

    # 품질 지표
    coherence_score: float = Field(default=0.0, description="일관성 점수")
    completeness_score: float = Field(default=0.0, description="완성도 점수")
    relevance_score: float = Field(default=0.0, description="관련성 점수")

    # 오류
    error: str | None = Field(default=None)

    @property
    def quality_score(self) -> float:
        """통합 품질 점수."""
        return (
            self.coherence_score * 0.3 + self.completeness_score * 0.4 + self.relevance_score * 0.3
        )

    class Config:
        arbitrary_types_allowed = True


class FusionResult(BaseModel):
    """융합 결과."""

    strategy: FusionStrategy = Field(description="사용된 융합 전략")
    output: str = Field(description="융합된 출력")
    confidence: float = Field(description="최종 신뢰도")

    # 메타데이터
    num_rollouts_used: int = Field(default=0, description="사용된 rollout 수")
    rollout_ids: List[str] = Field(default_factory=list, description="사용된 rollout ID들")

    # 품질 지표
    quality_score: float = Field(default=0.0)
    diversity_score: float = Field(default=0.0, description="결과 다양성 점수")
    agreement_score: float = Field(default=0.0, description="결과 일치도")

    # 분석
    analysis: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class ParallelRolloutExecutor:
    """병렬 Rollout 실행기.

    다양한 파라미터로 K개의 병렬 추론 실행.
    """

    def __init__(
        self,
        num_rollouts: int = 3,
        timeout_seconds: float = 120.0,
        diversity_strategy: str = "temperature_variation",
    ):
        self.num_rollouts = num_rollouts
        self.timeout_seconds = timeout_seconds
        self.diversity_strategy = diversity_strategy

        # 기본 temperature 범위
        self.temperature_range = (0.3, 1.0)

        logger.info(
            f"ParallelRolloutExecutor initialized: "
            f"num_rollouts={num_rollouts}, timeout={timeout_seconds}s"
        )

    async def execute(
        self,
        prompt: str,
        executor_fn: Callable,
        configs: List[RolloutConfig] | None = None,
        context: Dict[str, Any] | None = None,
    ) -> List[RolloutResult]:
        """병렬 rollout 실행.

        Args:
            prompt: 실행할 프롬프트
            executor_fn: 실행 함수 (async)
            configs: 커스텀 설정 (없으면 자동 생성)
            context: 추가 컨텍스트

        Returns:
            Rollout 결과 리스트
        """
        # 설정 생성
        if configs is None:
            configs = self._generate_diverse_configs()

        logger.info(f"Starting {len(configs)} parallel rollouts")

        # 병렬 실행
        tasks = [self._execute_single(prompt, config, executor_fn, context) for config in configs]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 정리
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_result = RolloutResult(
                    rollout_id=configs[i].rollout_id,
                    config=configs[i],
                    status=RolloutStatus.FAILED,
                    error=str(result),
                )
                valid_results.append(failed_result)
            else:
                valid_results.append(result)

        successful = sum(1 for r in valid_results if r.status == RolloutStatus.COMPLETED)
        logger.info(f"Rollouts completed: {successful}/{len(configs)} successful")

        return valid_results

    async def _execute_single(
        self,
        prompt: str,
        config: RolloutConfig,
        executor_fn: Callable,
        context: Dict[str, Any] | None,
    ) -> RolloutResult:
        """단일 rollout 실행."""
        result = RolloutResult(
            rollout_id=config.rollout_id,
            config=config,
            status=RolloutStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            # 타임아웃 적용
            async with asyncio.timeout(self.timeout_seconds):
                # 실행
                output = await executor_fn(
                    prompt=prompt,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_tokens,
                    context=context,
                )

            result.output = output.get("content", "") if isinstance(output, dict) else str(output)
            result.confidence = output.get("confidence", 0.5) if isinstance(output, dict) else 0.5
            result.tokens_used = output.get("tokens", 0) if isinstance(output, dict) else 0
            result.status = RolloutStatus.COMPLETED

            # 품질 점수 추정
            result.coherence_score = self._estimate_coherence(result.output)
            result.completeness_score = self._estimate_completeness(result.output)
            result.relevance_score = self._estimate_relevance(result.output, prompt)

        except TimeoutError:
            result.status = RolloutStatus.TIMEOUT
            result.error = f"Timeout after {self.timeout_seconds}s"

        except Exception as e:
            result.status = RolloutStatus.FAILED
            result.error = str(e)

        finally:
            result.completed_at = datetime.now()
            if result.started_at:
                result.duration_seconds = (result.completed_at - result.started_at).total_seconds()

        return result

    def _generate_diverse_configs(self) -> List[RolloutConfig]:
        """다양한 설정 자동 생성."""
        configs = []

        if self.diversity_strategy == "temperature_variation":
            # Temperature 변동
            temps = self._get_temperature_spread(self.num_rollouts)

            for i, temp in enumerate(temps):
                configs.append(
                    RolloutConfig(
                        rollout_id=f"rollout_{i}_{hashlib.md5(str(temp).encode()).hexdigest()[:6]}",
                        temperature=temp,
                        top_p=0.9,
                        diversity_factor=i / max(self.num_rollouts - 1, 1),
                    )
                )

        elif self.diversity_strategy == "full_variation":
            # 전체 파라미터 변동
            for i in range(self.num_rollouts):
                temp = self.temperature_range[0] + (
                    self.temperature_range[1] - self.temperature_range[0]
                ) * i / max(self.num_rollouts - 1, 1)

                top_p = 0.7 + 0.25 * i / max(self.num_rollouts - 1, 1)

                configs.append(
                    RolloutConfig(
                        rollout_id=f"rollout_{i}",
                        temperature=temp,
                        top_p=top_p,
                        seed=random.randint(0, 2**32 - 1),
                        diversity_factor=i / max(self.num_rollouts - 1, 1),
                    )
                )

        else:
            # 기본: 동일 설정
            for i in range(self.num_rollouts):
                configs.append(
                    RolloutConfig(
                        rollout_id=f"rollout_{i}",
                        temperature=0.7,
                        seed=random.randint(0, 2**32 - 1),
                    )
                )

        return configs

    def _get_temperature_spread(self, n: int) -> List[float]:
        """Temperature 분포 생성."""
        if n == 1:
            return [0.7]

        temps = []
        for i in range(n):
            t = self.temperature_range[0] + (
                self.temperature_range[1] - self.temperature_range[0]
            ) * i / (n - 1)
            temps.append(round(t, 2))

        return temps

    def _estimate_coherence(self, text: str) -> float:
        """일관성 추정."""
        if not text:
            return 0.0

        # 간단한 휴리스틱
        sentences = text.split(".")
        if len(sentences) < 2:
            return 0.5

        # 문장 길이 일관성
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.5

        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)

        # 낮은 분산 = 높은 일관성
        coherence = max(0.3, 1.0 - min(variance / 100, 0.7))

        return coherence

    def _estimate_completeness(self, text: str) -> float:
        """완성도 추정."""
        if not text:
            return 0.0

        # 길이 기반
        word_count = len(text.split())

        if word_count < 50:
            return 0.3
        elif word_count < 200:
            return 0.5 + 0.2 * (word_count - 50) / 150
        else:
            return min(0.9, 0.7 + 0.2 * (word_count - 200) / 500)

    def _estimate_relevance(self, text: str, prompt: str) -> float:
        """관련성 추정."""
        if not text or not prompt:
            return 0.0

        # 키워드 매칭
        prompt_words = set(prompt.lower().split())
        text_words = set(text.lower().split())

        common = prompt_words & text_words
        relevance = len(common) / max(len(prompt_words), 1)

        return min(1.0, relevance * 2)  # 스케일 조정


class FusionAgent:
    """융합 에이전트.

    여러 rollout 결과를 통합하여 최적의 결과 생성.
    """

    def __init__(
        self,
        default_strategy: FusionStrategy = FusionStrategy.LAST_K_FUSION,
        min_confidence_threshold: float = 0.3,
    ):
        self.default_strategy = default_strategy
        self.min_confidence_threshold = min_confidence_threshold

        logger.info(f"FusionAgent initialized: strategy={default_strategy.value}")

    async def fuse(
        self,
        results: List[RolloutResult],
        strategy: FusionStrategy | None = None,
        k: int = 3,
        llm_fuser: Callable | None = None,
    ) -> FusionResult:
        """Rollout 결과 융합.

        Args:
            results: Rollout 결과들
            strategy: 융합 전략
            k: Last-K에서 사용할 최근 결과 수
            llm_fuser: LLM 기반 융합 함수 (Last-K-Fusion용)

        Returns:
            융합된 결과
        """
        strategy = strategy or self.default_strategy

        # 성공한 결과만 필터링
        valid_results = [
            r
            for r in results
            if r.status == RolloutStatus.COMPLETED and r.confidence >= self.min_confidence_threshold
        ]

        if not valid_results:
            return FusionResult(
                strategy=strategy,
                output="No valid results to fuse",
                confidence=0.0,
                num_rollouts_used=0,
            )

        # 품질 기준 정렬
        sorted_results = sorted(valid_results, key=lambda r: r.quality_score, reverse=True)

        # 전략별 융합
        if strategy == FusionStrategy.BEST_CONFIDENCE:
            return self._fuse_best_confidence(sorted_results)

        elif strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return self._fuse_weighted_average(sorted_results)

        elif strategy == FusionStrategy.MAJORITY_VOTE:
            return self._fuse_majority_vote(sorted_results)

        elif strategy == FusionStrategy.LAST_K_FUSION:
            return await self._fuse_last_k(sorted_results[:k], llm_fuser)

        elif strategy == FusionStrategy.ENSEMBLE:
            return self._fuse_ensemble(sorted_results)

        else:
            return self._fuse_best_confidence(sorted_results)

    def _fuse_best_confidence(self, results: List[RolloutResult]) -> FusionResult:
        """가장 높은 confidence 결과 선택."""
        best = max(results, key=lambda r: r.confidence)

        return FusionResult(
            strategy=FusionStrategy.BEST_CONFIDENCE,
            output=best.output,
            confidence=best.confidence,
            num_rollouts_used=1,
            rollout_ids=[best.rollout_id],
            quality_score=best.quality_score,
            analysis={"selected": best.rollout_id, "reason": "highest_confidence"},
        )

    def _fuse_weighted_average(self, results: List[RolloutResult]) -> FusionResult:
        """신뢰도 가중 평균 (텍스트는 최고 품질 선택)."""
        # 텍스트는 가장 높은 품질 것을 사용
        best = max(results, key=lambda r: r.quality_score)

        # 신뢰도는 가중 평균
        total_weight = sum(r.quality_score for r in results)
        if total_weight > 0:
            weighted_conf = sum(r.confidence * r.quality_score for r in results) / total_weight
        else:
            weighted_conf = best.confidence

        return FusionResult(
            strategy=FusionStrategy.WEIGHTED_AVERAGE,
            output=best.output,
            confidence=weighted_conf,
            num_rollouts_used=len(results),
            rollout_ids=[r.rollout_id for r in results],
            quality_score=best.quality_score,
            analysis={
                "method": "weighted_average",
                "weights": {r.rollout_id: r.quality_score for r in results},
            },
        )

    def _fuse_majority_vote(self, results: List[RolloutResult]) -> FusionResult:
        """다수결 (출력 유사도 기반)."""
        # 출력 해시로 그룹화
        output_groups: Dict[str, List[RolloutResult]] = {}

        for result in results:
            # 간단한 정규화
            normalized = " ".join(result.output.lower().split())
            output_hash = hashlib.md5(normalized[:500].encode()).hexdigest()

            if output_hash not in output_groups:
                output_groups[output_hash] = []
            output_groups[output_hash].append(result)

        # 가장 큰 그룹 선택
        largest_group = max(output_groups.values(), key=len)
        best_in_group = max(largest_group, key=lambda r: r.quality_score)

        agreement = len(largest_group) / len(results)

        return FusionResult(
            strategy=FusionStrategy.MAJORITY_VOTE,
            output=best_in_group.output,
            confidence=best_in_group.confidence * (0.5 + 0.5 * agreement),
            num_rollouts_used=len(largest_group),
            rollout_ids=[r.rollout_id for r in largest_group],
            quality_score=best_in_group.quality_score,
            agreement_score=agreement,
            analysis={
                "num_groups": len(output_groups),
                "largest_group_size": len(largest_group),
                "agreement_rate": agreement,
            },
        )

    async def _fuse_last_k(
        self, results: List[RolloutResult], llm_fuser: Callable | None
    ) -> FusionResult:
        """Last-K 융합 (LLM 기반)."""
        if not llm_fuser:
            # LLM fuser 없으면 weighted_average fallback
            return self._fuse_weighted_average(results)

        # 융합 프롬프트 생성
        fusion_prompt = self._build_fusion_prompt(results)

        try:
            # LLM으로 융합
            fused_output = await llm_fuser(fusion_prompt)

            # 평균 confidence
            avg_confidence = sum(r.confidence for r in results) / len(results)
            avg_quality = sum(r.quality_score for r in results) / len(results)

            return FusionResult(
                strategy=FusionStrategy.LAST_K_FUSION,
                output=(
                    fused_output.get("content", "")
                    if isinstance(fused_output, dict)
                    else str(fused_output)
                ),
                confidence=avg_confidence * 1.1,  # 융합 보너스
                num_rollouts_used=len(results),
                rollout_ids=[r.rollout_id for r in results],
                quality_score=avg_quality,
                analysis={
                    "fusion_method": "llm_synthesis",
                    "input_rollouts": len(results),
                },
            )

        except Exception as e:
            logger.warning(f"LLM fusion failed: {e}, falling back to weighted_average")
            return self._fuse_weighted_average(results)

    def _fuse_ensemble(self, results: List[RolloutResult]) -> FusionResult:
        """앙상블 조합."""
        # 각 결과의 고유 인사이트 추출 및 조합
        best = max(results, key=lambda r: r.quality_score)

        # 다양성 점수 계산
        unique_contents = set()
        for r in results:
            # 첫 100자로 유사성 판단
            unique_contents.add(r.output[:100].lower())

        diversity = len(unique_contents) / len(results)

        # 앙상블 confidence
        ensemble_conf = sum(r.confidence for r in results) / len(results)
        ensemble_conf *= 0.8 + 0.2 * diversity  # 다양성 보너스

        return FusionResult(
            strategy=FusionStrategy.ENSEMBLE,
            output=best.output,
            confidence=ensemble_conf,
            num_rollouts_used=len(results),
            rollout_ids=[r.rollout_id for r in results],
            quality_score=best.quality_score,
            diversity_score=diversity,
            analysis={
                "ensemble_size": len(results),
                "diversity": diversity,
                "best_rollout": best.rollout_id,
            },
        )

    def _build_fusion_prompt(self, results: List[RolloutResult]) -> str:
        """융합 프롬프트 생성."""
        outputs = "\n\n".join(
            [
                f"--- Result {i + 1} (confidence: {r.confidence:.2f}) ---\n{r.output}"
                for i, r in enumerate(results)
            ]
        )

        return f"""You are a synthesis agent. Multiple AI agents have generated the following responses to the same query. Synthesize these into a single, comprehensive response that:

1. Combines the best insights from each response
2. Resolves any contradictions by favoring higher-confidence responses
3. Maintains coherence and completeness
4. Eliminates redundancy

## Responses to Synthesize:
{outputs}

## Your Synthesized Response:
"""


class TestTimeScaler:
    """High-level test-time scaling coordinator."""

    def __init__(
        self,
        num_rollouts: int = 3,
        fusion_strategy: FusionStrategy = FusionStrategy.LAST_K_FUSION,
        timeout_seconds: float = 120.0,
    ):
        self.executor = ParallelRolloutExecutor(
            num_rollouts=num_rollouts,
            timeout_seconds=timeout_seconds,
        )
        self.fusion_agent = FusionAgent(default_strategy=fusion_strategy)

    async def scale(
        self,
        prompt: str,
        executor_fn: Callable,
        llm_fuser: Callable | None = None,
        context: Dict[str, Any] | None = None,
    ) -> Tuple[FusionResult, List[RolloutResult]]:
        """Run parallel rollouts and fuse the results."""
        rollout_results = await self.executor.execute(
            prompt=prompt,
            executor_fn=executor_fn,
            context=context,
        )
        fusion_result = await self.fusion_agent.fuse(
            rollout_results,
            llm_fuser=llm_fuser,
        )
        return fusion_result, rollout_results


class MemoryAwareTestTimeScaler(TestTimeScaler):
    """메모리 인식 테스트 타임 스케일링.

    ReasoningBank의 핵심 혁신:
    - 실행 전: 관련 추론 메모리를 검색하여 프롬프트에 주입
    - 실행 후: 병렬 궤적(rollouts)을 대조하여 새로운 추론 메모리 학습
    """

    def __init__(
        self,
        num_rollouts: int = 3,
        fusion_strategy: FusionStrategy = FusionStrategy.LAST_K_FUSION,
        timeout_seconds: float = 120.0,
        domain: str = "general",
    ):
        super().__init__(num_rollouts, fusion_strategy, timeout_seconds)
        self.domain = domain

    async def scale_with_memory(
        self,
        prompt: str,
        executor_fn: Callable,
        llm_fuser: Callable | None = None,
        context: Dict[str, Any] | None = None,
    ) -> Tuple[FusionResult, List[RolloutResult]]:
        """메모리 인식 스케일링을 수행합니다."""
        from src.core.prompts.reasoning_memory_prompts import MEMORY_INJECTION_TEMPLATE
        from src.core.reasoning_memory import get_reasoning_memory_bank
        from src.core.reasoning_memory_inducer import (
            TrajectoryRecord,
            get_reasoning_memory_inducer,
        )

        memory_bank = get_reasoning_memory_bank()
        inducer = get_reasoning_memory_inducer()

        # 1. 관련된 과거 메모리 검색
        memories = await memory_bank.select_reasoning_memory(
            query=prompt, n=3, domain_filter=self.domain
        )

        # 2. 프롬프트 강화
        enhanced_prompt = prompt
        if memories:
            memory_text = "\n\n".join([f"### {m.title}\n{m.content}" for m in memories])
            injection = MEMORY_INJECTION_TEMPLATE.format(memories=memory_text)
            enhanced_prompt = f"{prompt}\n\n{injection}"
            logger.info(f"Injected {len(memories)} reasoning memories into prompt.")

        # 3. 기존 병렬 스케일링 실행
        fusion_result, rollout_results = await self.scale(
            prompt=enhanced_prompt, executor_fn=executor_fn, llm_fuser=llm_fuser, context=context
        )

        # 4. 결과로부터 새로운 메모리 학습 (비동기 백그라운드 태스크로 던질 수도 있음)
        try:
            trajectories = []
            for r in rollout_results:
                traj = TrajectoryRecord(query=prompt, domain=self.domain)
                # 실제 구현에서는 executor_fn이 남긴 상세 로그를 활용해야 함.
                # 여기서는 rollout_result의 output을 observation으로 간주
                traj.add_step(think="Rollout execution", action="Execution", observation=r.output)
                traj.final_result = r.output

                # Confidence로 대략적인 성공/실패 판단
                if r.status == RolloutStatus.COMPLETED:
                    traj.status = "success" if r.confidence >= 0.7 else "fail"
                else:
                    traj.status = "fail"

                trajectories.append(traj)

            # 병렬 궤적 대조로 메모리 추출
            new_memories = await inducer.induce_from_parallel(trajectories, prompt, self.domain)

            # 저장
            if new_memories:
                await memory_bank.store_batch(new_memories)
                logger.info(
                    f"Learned {len(new_memories)} new reasoning memories from test-time scaling."
                )

        except Exception as e:
            logger.error(f"Failed to induce memory after test-time scaling: {e}")

        return fusion_result, rollout_results


# Singleton instance
_test_time_scaler: TestTimeScaler | None = None
_memory_aware_scaler: MemoryAwareTestTimeScaler | None = None


def get_test_time_scaler(
    num_rollouts: int = 3,
    fusion_strategy: FusionStrategy = FusionStrategy.LAST_K_FUSION,
    use_memory: bool = False,
    domain: str = "general",
) -> TestTimeScaler:
    """TestTimeScaler 싱글톤 인스턴스 반환."""
    global _test_time_scaler
    global _memory_aware_scaler

    if use_memory:
        if _memory_aware_scaler is None:
            _memory_aware_scaler = MemoryAwareTestTimeScaler(
                num_rollouts=num_rollouts, fusion_strategy=fusion_strategy, domain=domain
            )
        return _memory_aware_scaler
    else:
        if _test_time_scaler is None:
            _test_time_scaler = TestTimeScaler(
                num_rollouts=num_rollouts, fusion_strategy=fusion_strategy
            )
        return _test_time_scaler
