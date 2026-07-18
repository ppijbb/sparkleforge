"""Per-model execution performance tracking.

Split out of the former monolithic llm_manager.py (issue #582).
"""

from typing import Any, Dict

from src.core.llm_manager.types import TaskType


class ModelPerformanceTracker:
    """모델 성능 추적기."""

    def __init__(self):
        self.performance_stats: Dict[str, Dict[str, Any]] = {}

    def record_execution(
        self,
        model_name: str,
        task_type: TaskType,
        execution_time: float,
        success: bool,
        quality_score: float = None,
    ):
        """실행 기록."""
        if model_name not in self.performance_stats:
            self.performance_stats[model_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time": 0.0,
                "avg_quality": 0.0,
                "task_performance": {},
            }

        stats = self.performance_stats[model_name]
        stats["total_executions"] += 1
        stats["total_time"] += execution_time

        if success:
            stats["successful_executions"] += 1

        if quality_score is not None:
            current_avg = stats["avg_quality"]
            total = stats["successful_executions"]
            stats["avg_quality"] = (current_avg * (total - 1) + quality_score) / total

        # 작업별 성능 추적
        task_key = task_type.value
        if task_key not in stats["task_performance"]:
            stats["task_performance"][task_key] = {
                "executions": 0,
                "successes": 0,
                "avg_time": 0.0,
                "avg_quality": 0.0,
            }

        task_stats = stats["task_performance"][task_key]
        task_stats["executions"] += 1
        if success:
            task_stats["successes"] += 1

        # 평균 시간 업데이트
        current_avg_time = task_stats["avg_time"]
        task_stats["avg_time"] = (
            current_avg_time * (task_stats["executions"] - 1) + execution_time
        ) / task_stats["executions"]

        # 평균 품질 업데이트
        if quality_score is not None and success:
            current_avg_quality = task_stats["avg_quality"]
            task_stats["avg_quality"] = (
                current_avg_quality * (task_stats["successes"] - 1) + quality_score
            ) / task_stats["successes"]

    def get_model_score(self, model_name: str, task_type: TaskType = None) -> float:
        """모델 점수 반환."""
        if model_name not in self.performance_stats:
            return 0.0

        stats = self.performance_stats[model_name]

        if task_type:
            task_key = task_type.value
            if task_key not in stats["task_performance"]:
                return 0.0

            task_stats = stats["task_performance"][task_key]
            if task_stats["executions"] == 0:
                return 0.0

            success_rate = task_stats["successes"] / task_stats["executions"]
            avg_quality = task_stats["avg_quality"]
            speed_score = 1.0 / (1.0 + task_stats["avg_time"])  # 시간이 짧을수록 높은 점수

            return success_rate * 0.4 + avg_quality * 0.4 + speed_score * 0.2
        else:
            # 전체 성능 점수
            if stats["total_executions"] == 0:
                return 0.0

            success_rate = stats["successful_executions"] / stats["total_executions"]
            avg_quality = stats["avg_quality"]
            avg_time = stats["total_time"] / stats["total_executions"]
            speed_score = 1.0 / (1.0 + avg_time)

            return success_rate * 0.4 + avg_quality * 0.4 + speed_score * 0.2


