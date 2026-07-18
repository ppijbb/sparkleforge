"""Quality/confidence scoring helpers for ResearchAgent.

Split out of the former monolithic research_agent.py (issue #582).

Note: this module intentionally contains two methods named
`_calculate_completeness_score` with different signatures (one taking
`data: List[Dict]`, one taking `synthesis_result: Dict`). This mirrors a
pre-existing bug in the original monolith -- Python keeps only the second
definition in the class namespace, silently shadowing the first, which is
therefore unreachable dead code. Preserved as-is here to keep before/after
refactor behavior identical; see the follow-up cleanup issue before removing
either one.
"""

from typing import Any, Dict, List

from src.utils.logger import setup_logger

logger = setup_logger("research_agent", log_level="INFO")


class QualityMetricsMixin:
    """Quality, relevance, confidence, and completeness scoring helpers."""

    def _calculate_data_quality(self, data: List[Dict[str, Any]]) -> float:
        """Calculate data quality score.

        Args:
            data: Data to evaluate

        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            if not data:
                return 0.0

            total_score = 0
            count = 0

            for item in data:
                if "data_points" in item:
                    for point in item["data_points"]:
                        reliability = point.get("reliability_score", 0.5)
                        total_score += reliability
                        count += 1

            return total_score / count if count > 0 else 0.5

        except Exception as e:
            logger.error(f"Data quality calculation failed: {e}")
            return 0.5


    def _calculate_relevance_score(self, data: Dict[str, Any], task: Dict[str, Any]) -> float:
        """Calculate relevance score for data.

        Args:
            data: Data to evaluate
            task: Research task

        Returns:
            Relevance score (0.0 to 1.0)
        """
        try:
            # Simple relevance calculation based on task description
            task_description = task.get("description", "").lower()
            data_content = str(data).lower()

            # Count keyword matches
            keywords = task_description.split()
            matches = sum(1 for keyword in keywords if keyword in data_content)

            return min(matches / len(keywords) if keywords else 0, 1.0)

        except Exception as e:
            logger.error(f"Relevance score calculation failed: {e}")
            return 0.5


    def _calculate_quality_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive quality metrics.

        Args:
            data: Data to evaluate

        Returns:
            Quality metrics dictionary
        """
        try:
            return {
                "completeness": self._calculate_completeness_score(data),
                "accuracy": self._calculate_accuracy_score(data),
                "relevance": self._calculate_relevance_score(data[0], {}) if data else 0.5,
                "timeliness": self._calculate_timeliness_score(data),
                "consistency": self._calculate_consistency_score(data),
            }

        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return {
                "completeness": 0.5,
                "accuracy": 0.5,
                "relevance": 0.5,
                "timeliness": 0.5,
                "consistency": 0.5,
            }


    def _calculate_completeness_score(self, data: List[Dict[str, Any]]) -> float:
        """Calculate completeness score."""
        return min(len(data) / 5, 1.0) if data else 0.0


    def _calculate_accuracy_score(self, data: List[Dict[str, Any]]) -> float:
        """Calculate accuracy score."""
        return self._calculate_data_quality(data)


    def _calculate_timeliness_score(self, data: List[Dict[str, Any]]) -> float:
        """Calculate timeliness score."""
        # Implement actual timeliness calculation
        raise NotImplementedError("_calculate_timeliness_score requires actual implementation")


    def _calculate_consistency_score(self, data: List[Dict[str, Any]]) -> float:
        """Calculate consistency score."""
        # Implement actual consistency calculation
        raise NotImplementedError("_calculate_consistency_score requires actual implementation")

    # Analysis, synthesis, and validation methods using LLM

    def _calculate_analysis_quality(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate analysis quality score based on result structure."""
        if not analysis_result:
            return 0.0

        score = 0.5  # Base score

        # Check for required keys
        if "findings" in analysis_result or "results" in analysis_result:
            score += 0.2
        if "data_points" in analysis_result or "patterns" in analysis_result:
            score += 0.2
        if "conclusions" in analysis_result:
            score += 0.1

        return min(score, 1.0)


    def _calculate_confidence_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate confidence score based on analysis quality."""
        base_quality = self._calculate_analysis_quality(analysis_result)
        return base_quality * 0.9  # Slightly conservative


    def _calculate_synthesis_quality(self, synthesis_result: Dict[str, Any]) -> float:
        """Calculate synthesis quality score."""
        if not synthesis_result:
            return 0.0

        score = 0.5
        if "integrated_findings" in synthesis_result or "synthesis" in synthesis_result:
            score += 0.3
        if "themes" in synthesis_result or "connections" in synthesis_result:
            score += 0.2

        return min(score, 1.0)


    def _calculate_completeness_score(self, synthesis_result: Dict[str, Any]) -> float:
        """Calculate completeness score."""
        return self._calculate_synthesis_quality(synthesis_result) * 0.9


    def _calculate_research_quality_from_data(
        self, collected_data: List[Dict[str, Any]], additional_research: Dict[str, Any]
    ) -> float:
        """Calculate research quality based on collected data."""
        try:
            quality_score = 0.0

            # Base score for collected data
            if collected_data:
                quality_score += 0.4

                # Bonus for content length
                total_content_length = sum(len(item.get("content", "")) for item in collected_data)
                if total_content_length > 1000:
                    quality_score += 0.2
                elif total_content_length > 500:
                    quality_score += 0.1

            # Bonus for additional sources
            additional_sources = additional_research.get("sources", [])
            if additional_sources:
                quality_score += 0.2

                # Bonus for diverse source types
                source_types = set(source.get("source_type", "") for source in additional_sources)
                if len(source_types) > 1:
                    quality_score += 0.1

            # Bonus for total sources
            total_sources = len(collected_data) + len(additional_sources)
            if total_sources > 5:
                quality_score += 0.1
            elif total_sources > 3:
                quality_score += 0.05

            return min(quality_score, 1.0)

        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.5


    def _calculate_research_quality(self, research_result: Dict[str, Any]) -> float:
        """Calculate research quality score."""
        return 0.8


    def _calculate_comprehensiveness_score(self, research_result: Dict[str, Any]) -> float:
        """Calculate comprehensiveness score."""
        return 0.75


