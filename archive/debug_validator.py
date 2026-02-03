"""
디버깅 및 검증 시스템

확률적 실패 재현 시스템, 병렬 실행 검증 도구,
과거 시나리오 재실행 검증, 실패 패턴 분석 및 추적,
재현 가능한 테스트 케이스 생성, 검증 결과 리포트 및 통계
"""

import asyncio
import json
import logging
import time
import hashlib
import random
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """실패 유형."""
    DETERMINISTIC = "deterministic"    # 결정적 실패
    PROBABILISTIC = "probabilistic"   # 확률적 실패
    INTERMITTENT = "intermittent"     # 간헐적 실패
    TIMEOUT = "timeout"              # 타임아웃
    RESOURCE = "resource"            # 리소스 부족
    NETWORK = "network"              # 네트워크 문제
    UNKNOWN = "unknown"              # 알 수 없는 실패


class TestScenario(Enum):
    """테스트 시나리오."""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    STRESS_TEST = "stress_test"
    REGRESSION_TEST = "regression_test"
    PERFORMANCE_TEST = "performance_test"
    FAILURE_REPRODUCTION = "failure_reproduction"


@dataclass
class FailurePattern:
    """실패 패턴."""
    pattern_id: str
    failure_type: FailureType
    error_message: str
    stack_trace: Optional[str] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    frequency: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    affected_components: List[str] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)

    def update_frequency(self):
        """빈도 업데이트."""
        self.frequency += 1
        self.last_seen = time.time()

    def matches(self, error_msg: str, conditions: Dict[str, Any]) -> bool:
        """패턴 매칭."""
        # 에러 메시지 유사성 검사
        if self.error_message not in error_msg and error_msg not in self.error_message:
            return False

        # 조건 일치 검사
        for key, value in self.conditions.items():
            if key not in conditions or conditions[key] != value:
                return False

        return True


@dataclass
class TestCase:
    """테스트 케이스."""
    case_id: str
    name: str
    scenario: TestScenario
    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    retries: int = 3
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_run: Optional[float] = None
    success_count: int = 0
    failure_count: int = 0

    @property
    def success_rate(self) -> float:
        """성공률."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class TestResult:
    """테스트 결과."""
    test_case: TestCase
    success: bool
    execution_time: float
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    run_id: str = field(default_factory=lambda: f"run_{int(time.time())}_{random.randint(1000, 9999)}")


@dataclass
class ValidationReport:
    """검증 리포트."""
    report_id: str
    scenario: TestScenario
    start_time: float
    end_time: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[TestResult] = field(default_factory=list)
    failure_patterns: List[FailurePattern] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """총 실행 시간."""
        return self.end_time - self.start_time

    @property
    def success_rate(self) -> float:
        """전체 성공률."""
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0

    def add_result(self, result: TestResult):
        """결과 추가."""
        self.results.append(result)
        if result.success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

    def generate_summary(self) -> Dict[str, Any]:
        """요약 생성."""
        execution_times = [r.execution_time for r in self.results]

        return {
            'report_id': self.report_id,
            'scenario': self.scenario.value,
            'duration': self.duration,
            'success_rate': self.success_rate,
            'total_tests': self.total_tests,
            'passed': self.passed_tests,
            'failed': self.failed_tests,
            'avg_execution_time': statistics.mean(execution_times) if execution_times else 0,
            'median_execution_time': statistics.median(execution_times) if execution_times else 0,
            'failure_patterns_count': len(self.failure_patterns),
            'recommendations_count': len(self.recommendations)
        }


class ProbabilisticFailureSimulator:
    """확률적 실패 시뮬레이터."""

    def __init__(self, failure_rate: float = 0.1):
        """
        초기화.

        Args:
            failure_rate: 실패 확률 (0.0 ~ 1.0)
        """
        self.failure_rate = failure_rate
        self.failure_patterns = [
            "Connection timeout",
            "Memory allocation failed",
            "Disk I/O error",
            "Network unreachable",
            "Database connection lost",
            "API rate limit exceeded",
            "Invalid response format",
            "Authentication failed"
        ]

    def should_fail(self) -> bool:
        """실패 여부 결정."""
        return random.random() < self.failure_rate

    def generate_failure(self) -> Tuple[str, str]:
        """실패 생성."""
        error_msg = random.choice(self.failure_patterns)
        stack_trace = self._generate_stack_trace(error_msg)
        return error_msg, stack_trace

    def _generate_stack_trace(self, error_msg: str) -> str:
        """스택 트레이스 생성."""
        frames = [
            f"File \"src/core/{random.choice(['agent', 'tool', 'network', 'database'])}.py\", line {random.randint(10, 500)}, in {random.choice(['execute', 'process', 'handle', 'connect'])}",
            f"File \"src/utils/{random.choice(['helpers', 'validators', 'formatters'])}.py\", line {random.randint(10, 200)}, in {random.choice(['validate', 'format', 'parse'])}",
            f"File \"main.py\", line {random.randint(50, 200)}, in main"
        ]

        stack = f"Traceback (most recent call last):\n"
        for frame in frames:
            stack += f"  {frame}\n"
        stack += f"{type(Exception).__name__}: {error_msg}\n"

        return stack


class ParallelTestExecutor:
    """병렬 테스트 실행기."""

    def __init__(self, max_concurrent: int = 5):
        """
        초기화.

        Args:
            max_concurrent: 최대 동시 실행 수
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_test(
        self,
        test_case: TestCase,
        test_function: Callable,
        failure_simulator: Optional[ProbabilisticFailureSimulator] = None
    ) -> TestResult:
        """
        테스트 실행.

        Args:
            test_case: 테스트 케이스
            test_function: 테스트 함수
            failure_simulator: 실패 시뮬레이터

        Returns:
            테스트 결과
        """
        async with self.semaphore:
            start_time = time.time()

            try:
                # 타임아웃과 재시도 로직
                for attempt in range(test_case.retries + 1):
                    try:
                        # 실패 시뮬레이션 (디버깅용)
                        if failure_simulator and failure_simulator.should_fail():
                            error_msg, stack_trace = failure_simulator.generate_failure()
                            raise Exception(error_msg)

                        # 테스트 함수 실행
                        result = await asyncio.wait_for(
                            test_function(test_case.input_data),
                            timeout=test_case.timeout
                        )

                        execution_time = time.time() - start_time

                        # 결과 검증
                        success = self._validate_result(result, test_case.expected_output)

                        return TestResult(
                            test_case=test_case,
                            success=success,
                            execution_time=execution_time,
                            output_data=result
                        )

                    except asyncio.TimeoutError:
                        if attempt == test_case.retries:
                            execution_time = time.time() - start_time
                            return TestResult(
                                test_case=test_case,
                                success=False,
                                execution_time=execution_time,
                                error_message=f"Test timed out after {test_case.timeout}s"
                            )
                        continue

                    except Exception as e:
                        if attempt == test_case.retries:
                            execution_time = time.time() - start_time
                            return TestResult(
                                test_case=test_case,
                                success=False,
                                execution_time=execution_time,
                                error_message=str(e),
                                stack_trace=self._get_stack_trace(e)
                            )
                        continue

            finally:
                test_case.last_run = time.time()

    def _validate_result(self, actual: Any, expected: Any) -> bool:
        """결과 검증."""
        if expected is None:
            return True  # 기대값이 없으면 성공으로 간주

        # 간단한 검증 로직
        if isinstance(expected, dict) and isinstance(actual, dict):
            # 키 존재 검증
            for key in expected.keys():
                if key not in actual:
                    return False
            return True

        elif isinstance(expected, list) and isinstance(actual, list):
            return len(actual) == len(expected)

        else:
            return actual == expected

    def _get_stack_trace(self, exception: Exception) -> str:
        """스택 트레이스 가져오기."""
        import traceback
        return "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))


class HistoricalScenarioReplayer:
    """과거 시나리오 재실행기."""

    def __init__(self):
        """초기화."""
        self.scenarios: Dict[str, Dict[str, Any]] = {}
        self.execution_history: Dict[str, List[TestResult]] = defaultdict(list)

    def save_scenario(self, scenario_id: str, input_data: Dict[str, Any], context: Dict[str, Any]):
        """
        시나리오 저장.

        Args:
            scenario_id: 시나리오 ID
            input_data: 입력 데이터
            context: 실행 컨텍스트
        """
        self.scenarios[scenario_id] = {
            'input_data': input_data,
            'context': context,
            'created_at': time.time(),
            'execution_count': 0
        }

    async def replay_scenario(
        self,
        scenario_id: str,
        test_function: Callable,
        iterations: int = 10
    ) -> List[TestResult]:
        """
        시나리오 재실행.

        Args:
            scenario_id: 시나리오 ID
            test_function: 테스트 함수
            iterations: 반복 횟수

        Returns:
            테스트 결과들
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")

        scenario = self.scenarios[scenario_id]
        results = []

        for i in range(iterations):
            start_time = time.time()

            try:
                result = await test_function(scenario['input_data'])
                execution_time = time.time() - start_time

                test_result = TestResult(
                    test_case=TestCase(
                        case_id=f"replay_{scenario_id}_{i}",
                        name=f"Replay {scenario_id} #{i}",
                        scenario=TestScenario.REGRESSION_TEST,
                        input_data=scenario['input_data']
                    ),
                    success=True,
                    execution_time=execution_time,
                    output_data=result
                )

            except Exception as e:
                execution_time = time.time() - start_time
                test_result = TestResult(
                    test_case=TestCase(
                        case_id=f"replay_{scenario_id}_{i}",
                        name=f"Replay {scenario_id} #{i}",
                        scenario=TestScenario.REGRESSION_TEST,
                        input_data=scenario['input_data']
                    ),
                    success=False,
                    execution_time=execution_time,
                    error_message=str(e),
                    stack_trace=self._get_stack_trace(e)
                )

            results.append(test_result)
            scenario['execution_count'] += 1

        self.execution_history[scenario_id].extend(results)
        return results

    def _get_stack_trace(self, exception: Exception) -> str:
        """스택 트레이스 가져오기."""
        import traceback
        return "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))


class DebugValidator:
    """
    디버깅 및 검증 시스템.

    확률적 실패 재현, 병렬 실행 검증, 과거 시나리오 재실행,
    실패 패턴 분석, 재현 가능한 테스트 케이스 생성.
    """

    def __init__(self):
        """초기화."""
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.test_cases: Dict[str, TestCase] = {}
        self.test_executor = ParallelTestExecutor()
        self.failure_simulator = ProbabilisticFailureSimulator()
        self.scenario_replayer = HistoricalScenarioReplayer()

        # 통계
        self.stats = {
            'total_tests_run': 0,
            'total_failures': 0,
            'patterns_discovered': 0,
            'scenarios_recorded': 0
        }

        logger.info("DebugValidator initialized")

    async def run_failure_reproduction_test(
        self,
        test_function: Callable,
        input_data: Dict[str, Any],
        iterations: int = 100,
        failure_rate: float = 0.1
    ) -> ValidationReport:
        """
        실패 재현 테스트 실행.

        Args:
            test_function: 테스트 함수
            input_data: 입력 데이터
            iterations: 반복 횟수
            failure_rate: 실패 확률

        Returns:
            검증 리포트
        """
        logger.info(f"Starting failure reproduction test with {iterations} iterations")

        # 실패 시뮬레이터 설정
        simulator = ProbabilisticFailureSimulator(failure_rate)

        # 테스트 케이스 생성
        test_case = TestCase(
            case_id=f"failure_repro_{int(time.time())}",
            name="Failure Reproduction Test",
            scenario=TestScenario.FAILURE_REPRODUCTION,
            input_data=input_data,
            tags=["failure_reproduction", "probabilistic"]
        )

        start_time = time.time()
        report = ValidationReport(
            report_id=f"report_{int(start_time)}",
            scenario=TestScenario.FAILURE_REPRODUCTION,
            start_time=start_time,
            end_time=0,
            total_tests=iterations,
            passed_tests=0,
            failed_tests=0
        )

        # 병렬 실행
        tasks = []
        for i in range(iterations):
            task = self.test_executor.execute_test(
                TestCase(
                    case_id=f"{test_case.case_id}_{i}",
                    name=f"{test_case.name} #{i}",
                    scenario=test_case.scenario,
                    input_data=input_data
                ),
                test_function,
                simulator
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 처리
        for result in results:
            if isinstance(result, Exception):
                # 실행 실패
                error_result = TestResult(
                    test_case=test_case,
                    success=False,
                    execution_time=0.0,
                    error_message=str(result)
                )
                report.add_result(error_result)
            else:
                report.add_result(result)

                # 실패 패턴 분석
                if not result.success:
                    self._analyze_failure(result)

        report.end_time = time.time()

        # 리포트 생성
        self._generate_failure_report(report)

        logger.info(f"Failure reproduction test completed: {report.passed_tests}/{report.total_tests} passed")
        return report

    async def run_regression_test(
        self,
        test_cases: List[TestCase],
        test_function: Callable
    ) -> ValidationReport:
        """
        회귀 테스트 실행.

        Args:
            test_cases: 테스트 케이스들
            test_function: 테스트 함수

        Returns:
            검증 리포트
        """
        logger.info(f"Starting regression test with {len(test_cases)} test cases")

        start_time = time.time()
        report = ValidationReport(
            report_id=f"regression_{int(start_time)}",
            scenario=TestScenario.REGRESSION_TEST,
            start_time=start_time,
            end_time=0,
            total_tests=len(test_cases),
            passed_tests=0,
            failed_tests=0
        )

        # 테스트 실행
        tasks = [self.test_executor.execute_test(tc, test_function) for tc in test_cases]
        results = await asyncio.gather(*tasks)

        # 결과 처리
        for result in results:
            report.add_result(result)

            # 실패 패턴 분석
            if not result.success:
                self._analyze_failure(result)

        report.end_time = time.time()

        logger.info(f"Regression test completed: {report.passed_tests}/{report.total_tests} passed")
        return report

    async def run_stress_test(
        self,
        test_function: Callable,
        input_variations: List[Dict[str, Any]],
        concurrent_users: int = 10,
        duration: float = 60.0
    ) -> ValidationReport:
        """
        스트레스 테스트 실행.

        Args:
            test_function: 테스트 함수
            input_variations: 입력 데이터 변형들
            concurrent_users: 동시 사용자 수
            duration: 테스트 지속 시간

        Returns:
            검증 리포트
        """
        logger.info(f"Starting stress test: {concurrent_users} concurrent users for {duration}s")

        start_time = time.time()
        report = ValidationReport(
            report_id=f"stress_{int(start_time)}",
            scenario=TestScenario.STRESS_TEST,
            start_time=start_time,
            end_time=start_time + duration,
            total_tests=0,  # 동적으로 계산
            passed_tests=0,
            failed_tests=0
        )

        # 스트레스 테스트용 실행기
        stress_executor = ParallelTestExecutor(concurrent_users)

        async def run_until_timeout():
            test_count = 0
            while time.time() - start_time < duration:
                # 랜덤 입력 선택
                input_data = random.choice(input_variations)

                test_case = TestCase(
                    case_id=f"stress_{test_count}",
                    name=f"Stress Test #{test_count}",
                    scenario=TestScenario.STRESS_TEST,
                    input_data=input_data
                )

                result = await stress_executor.execute_test(test_case, test_function)
                report.add_result(result)
                report.total_tests += 1

                if not result.success:
                    self._analyze_failure(result)

                test_count += 1

        await run_until_timeout()

        # 성능 메트릭 계산
        execution_times = [r.execution_time for r in report.results]
        report.performance_metrics = {
            'avg_response_time': statistics.mean(execution_times) if execution_times else 0,
            'median_response_time': statistics.median(execution_times) if execution_times else 0,
            'min_response_time': min(execution_times) if execution_times else 0,
            'max_response_time': max(execution_times) if execution_times else 0,
            'requests_per_second': len(report.results) / duration,
            'error_rate': report.failed_tests / report.total_tests if report.total_tests > 0 else 0
        }

        logger.info(f"Stress test completed: {report.total_tests} requests, "
                   f"{report.performance_metrics['requests_per_second']:.2f} req/s")
        return report

    def _analyze_failure(self, result: TestResult):
        """실패 분석."""
        if not result.error_message:
            return

        # 패턴 매칭
        pattern_key = self._generate_pattern_key(result.error_message)

        if pattern_key in self.failure_patterns:
            pattern = self.failure_patterns[pattern_key]
            pattern.update_frequency()
        else:
            # 새 패턴 생성
            pattern = FailurePattern(
                pattern_id=pattern_key,
                failure_type=self._classify_failure(result),
                error_message=result.error_message,
                stack_trace=result.stack_trace,
                conditions=result.test_case.input_data
            )
            self.failure_patterns[pattern_key] = pattern
            self.stats['patterns_discovered'] += 1

        self.stats['total_failures'] += 1

    def _generate_pattern_key(self, error_message: str) -> str:
        """패턴 키 생성."""
        # 에러 메시지의 핵심 부분 추출
        key_parts = error_message.lower().split()[:5]  # 처음 5단어
        return hashlib.md5(" ".join(key_parts).encode()).hexdigest()[:16]

    def _classify_failure(self, result: TestResult) -> FailureType:
        """실패 유형 분류."""
        error_msg = result.error_message.lower()

        if "timeout" in error_msg:
            return FailureType.TIMEOUT
        elif "connection" in error_msg or "network" in error_msg:
            return FailureType.NETWORK
        elif "memory" in error_msg or "disk" in error_msg:
            return FailureType.RESOURCE
        elif result.execution_time > result.test_case.timeout:
            return FailureType.TIMEOUT
        else:
            return FailureType.UNKNOWN

    def _generate_failure_report(self, report: ValidationReport):
        """실패 리포트 생성."""
        # 실패 패턴 추가
        pattern_counts = Counter()
        for result in report.results:
            if not result.success and result.error_message:
                pattern_key = self._generate_pattern_key(result.error_message)
                pattern_counts[pattern_key] += 1

        # 가장 빈번한 패턴들 추가
        for pattern_key, count in pattern_counts.most_common(5):
            if pattern_key in self.failure_patterns:
                pattern = self.failure_patterns[pattern_key]
                pattern.frequency = count
                report.failure_patterns.append(pattern)

        # 추천사항 생성
        if report.failed_tests > 0:
            failure_rate = report.failed_tests / report.total_tests

            if failure_rate > 0.5:
                report.recommendations.append("높은 실패율 감지: 시스템 안정성 검토 필요")
            elif failure_rate > 0.2:
                report.recommendations.append("중간 수준 실패율: 에러 처리 개선 고려")
            else:
                report.recommendations.append("낮은 실패율: 간헐적 문제 모니터링 유지")

            # 패턴 기반 추천
            for pattern in report.failure_patterns:
                if pattern.failure_type == FailureType.TIMEOUT:
                    report.recommendations.append("타임아웃 문제 감지: 타임아웃 값 조정 또는 성능 최적화 고려")
                elif pattern.failure_type == FailureType.NETWORK:
                    report.recommendations.append("네트워크 문제 감지: 연결 안정성 및 재시도 로직 검토")

    def get_failure_patterns(self) -> List[Dict[str, Any]]:
        """실패 패턴 목록 반환."""
        return [
            {
                'pattern_id': p.pattern_id,
                'failure_type': p.failure_type.value,
                'error_message': p.error_message,
                'frequency': p.frequency,
                'first_seen': datetime.fromtimestamp(p.first_seen).isoformat(),
                'last_seen': datetime.fromtimestamp(p.last_seen).isoformat(),
                'affected_components': p.affected_components
            }
            for p in self.failure_patterns.values()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환."""
        return {
            **self.stats,
            'active_patterns': len(self.failure_patterns),
            'test_cases': len(self.test_cases),
            'recorded_scenarios': len(self.scenario_replayer.scenarios)
        }

    def create_reproducible_test_case(
        self,
        failure_pattern: FailurePattern,
        test_function: Callable
    ) -> TestCase:
        """
        재현 가능한 테스트 케이스 생성.

        Args:
            failure_pattern: 실패 패턴
            test_function: 테스트 함수

        Returns:
            테스트 케이스
        """
        case_id = f"repro_{failure_pattern.pattern_id}_{int(time.time())}"

        test_case = TestCase(
            case_id=case_id,
            name=f"Reproduce: {failure_pattern.error_message[:50]}...",
            scenario=TestScenario.FAILURE_REPRODUCTION,
            input_data=failure_pattern.conditions.copy(),
            timeout=60.0,  # 실패 재현을 위해 더 긴 타임아웃
            retries=5,
            tags=["reproducible", failure_pattern.failure_type.value]
        )

        self.test_cases[case_id] = test_case
        return test_case


# 전역 디버그 검증기 인스턴스
_debug_validator = None

def get_debug_validator() -> DebugValidator:
    """전역 디버그 검증기 인스턴스 반환."""
    global _debug_validator
    if _debug_validator is None:
        _debug_validator = DebugValidator()
    return _debug_validator

def set_debug_validator(validator: DebugValidator):
    """전역 디버그 검증기 설정."""
    global _debug_validator
    _debug_validator = validator
