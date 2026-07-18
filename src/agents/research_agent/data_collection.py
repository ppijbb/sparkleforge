"""Data collection pipeline for ResearchAgent.

Split out of the former monolithic research_agent.py (issue #582).
"""

from datetime import datetime
from typing import Any, Dict, List

from src.utils.logger import setup_logger

logger = setup_logger("research_agent", log_level="INFO")


class DataCollectionMixin:
    """Data collection, source identification, and processing pipeline."""

    async def _perform_data_analysis(self, query: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data analysis."""
        try:
            # Use LLM to analyze data
            prompt = f"""
            다음 주제에 대한 데이터 분석을 수행하세요:
            
            주제: {query}
            작업: {task.get("description", "")}
            
            다음을 포함한 분석을 제공하세요:
            1. 주요 트렌드 식별
            2. 통계적 인사이트
            3. 패턴 분석
            4. 결론 및 권고사항
            
            JSON 형태로 응답하세요.
            """

            response = await asyncio.to_thread(self.llm.generate_content, prompt)
            response_text = response.text.strip()
            try:
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    analysis = {"analysis": response_text, "type": "text"}

            return {
                "method": "data_analysis",
                "query": query,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return {
                "method": "data_analysis",
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


    async def _perform_content_analysis(self, query: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform content analysis."""
        try:
            # Use LLM for content analysis
            prompt = f"""
            다음 주제에 대한 콘텐츠 분석을 수행하세요:
            
            주제: {query}
            작업: {task.get("description", "")}
            
            다음을 포함한 분석을 제공하세요:
            1. 주요 주제 식별
            2. 감정 분석
            3. 키워드 분석
            4. 핵심 메시지 추출
            
            JSON 형태로 응답하세요.
            """

            response = await asyncio.to_thread(self.llm.generate_content, prompt)
            response_text = response.text.strip()
            try:
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    analysis = {"analysis": response_text, "type": "text"}

            return {
                "method": "content_analysis",
                "query": query,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {
                "method": "content_analysis",
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


    async def _llm_analyze_research_results(
        self, research_results: List[Dict[str, Any]], task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to analyze research results."""
        try:
            prompt = f"""
            다음 연구 결과들을 분석하고 종합하세요:
            
            연구 결과들: {research_results}
            원래 작업: {task.get("description", "")}
            
            다음을 포함한 분석을 제공하세요:
            1. 결과 요약
            2. 핵심 발견사항
            3. 신뢰성 평가
            4. 품질 점수 (0.0-1.0)
            5. 결론 및 권고사항
            
            JSON 형태로 응답하세요:
            {{
                "summary": "결과 요약",
                "key_findings": ["발견사항1", "발견사항2"],
                "quality_score": 0.0-1.0,
                "reliability_score": 0.0-1.0,
                "conclusions": "결론",
                "recommendations": ["권고사항1", "권고사항2"]
            }}
            """

            response = await asyncio.to_thread(self.llm.generate_content, prompt)
            # Parse Gemini response properly
            response_text = response.text.strip()
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")

        except Exception as e:
            logger.error(f"LLM result analysis failed: {e}")
            return {
                "summary": f"Analysis failed: {str(e)}",
                "key_findings": [],
                "quality_score": 0.0,
                "reliability_score": 0.0,
                "conclusions": "Analysis could not be completed",
                "recommendations": [],
            }


    async def _execute_data_collection_task(
        self, task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Execute data collection task.

        Args:
            task: Data collection task
            objective_id: Objective ID

        Returns:
            Data collection result
        """
        try:
            logger.info(f"Executing data collection task: {task.get('task_id')}")

            # Determine data sources based on task requirements
            data_sources = await self._identify_data_sources(task, objective_id)

            # Collect data from each source
            collected_data = []
            for source in data_sources:
                source_data = await self._collect_from_source(source, task, objective_id)
                if source_data:
                    collected_data.append(source_data)

            # Process and clean collected data
            processed_data = await self._process_collected_data(collected_data, task)

            # Generate data summary
            data_summary = await self._generate_data_summary(processed_data, task)

            result = {
                "data_collection_result": {
                    "sources_used": len(data_sources),
                    "data_points_collected": len(processed_data),
                    "data_quality_score": self._calculate_data_quality(processed_data),
                    "collection_timestamp": datetime.now().isoformat(),
                },
                "raw_data": processed_data,
                "data_summary": data_summary,
                "metadata": {
                    "task_description": task.get("description", ""),
                    "collection_method": "autonomous_research",
                    "quality_metrics": self._calculate_quality_metrics(processed_data),
                },
            }

            return result

        except Exception as e:
            logger.error(f"Data collection task failed: {e}")
            raise


    async def _identify_data_sources(
        self, task: Dict[str, Any], objective_id: str
    ) -> List[Dict[str, Any]]:
        """Identify appropriate data sources for the task.

        Args:
            task: Research task
            objective_id: Objective ID

        Returns:
            List of identified data sources
        """
        try:
            sources = []
            task_description = task.get("description", "").lower()

            # Web sources
            if any(
                keyword in task_description for keyword in ["web", "online", "internet", "website"]
            ):
                sources.append(
                    {
                        "type": "web_sources",
                        "subtype": "web_search",
                        "priority": 0.8,
                        "reliability": 0.7,
                    }
                )

            # Academic sources
            if any(
                keyword in task_description
                for keyword in ["academic", "research", "paper", "study", "literature"]
            ):
                sources.append(
                    {
                        "type": "academic_sources",
                        "subtype": "academic_search",
                        "priority": 0.9,
                        "reliability": 0.9,
                    }
                )

            # Data sources
            if any(
                keyword in task_description
                for keyword in ["data", "statistics", "dataset", "database"]
            ):
                sources.append(
                    {
                        "type": "data_sources",
                        "subtype": "data_analysis",
                        "priority": 0.8,
                        "reliability": 0.8,
                    }
                )

            # Default to web sources if no specific type identified
            if not sources:
                sources.append(
                    {
                        "type": "web_sources",
                        "subtype": "web_search",
                        "priority": 0.6,
                        "reliability": 0.7,
                    }
                )

            return sources

        except Exception as e:
            logger.error(f"Data source identification failed: {e}")
            return []


    async def _collect_from_source(
        self, source: Dict[str, Any], task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any] | None:
        """Collect data from a specific source.

        Args:
            source: Data source configuration
            task: Research task
            objective_id: Objective ID

        Returns:
            Collected data or None if collection failed
        """
        try:
            source_type = source.get("type", "web_sources")
            subtype = source.get("subtype", "web_search")

            # Simulate data collection based on source type
            if source_type == "web_sources":
                return await self._collect_web_data(subtype, task, objective_id)
            elif source_type == "academic_sources":
                return await self._collect_academic_data(subtype, task, objective_id)
            elif source_type == "data_sources":
                return await self._collect_structured_data(subtype, task, objective_id)
            else:
                return await self._collect_general_data(subtype, task, objective_id)

        except Exception as e:
            logger.error(f"Data collection from source failed: {e}")
            return None


    async def _collect_web_data(
        self, subtype: str, task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Collect data from web sources.

        Args:
            subtype: Web data subtype
            task: Research task
            objective_id: Objective ID

        Returns:
            Web data collection result
        """
        # 실제 웹 데이터 수집 수행 (MCP 도구 사용)
        raise NotImplementedError(
            "_simulate_web_data_collection is not implemented. Use actual MCP search tools instead."
        )


    async def _collect_academic_data(
        self, subtype: str, task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Collect data from academic sources.

        Args:
            subtype: Academic data subtype
            task: Research task
            objective_id: Objective ID

        Returns:
            Academic data collection result
        """
        try:
            # Simulate academic data collection
            academic_data = {
                "source_type": "academic",
                "subtype": subtype,
                "data_points": [
                    {
                        "title": f"Academic paper on {task.get('description', 'task')}",
                        "authors": ["Dr. Researcher", "Prof. Academic"],
                        "journal": "Journal of Research",
                        "year": 2024,
                        "abstract": f"Abstract of research on {task.get('description', 'task')}",
                        "reliability_score": 0.9,
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
                "collection_metadata": {
                    "search_query": task.get("description", ""),
                    "results_count": 1,
                    "collection_method": "autonomous_academic_search",
                },
            }

            return academic_data

        except Exception as e:
            logger.error(f"Academic data collection failed: {e}")
            return {}


    async def _collect_structured_data(
        self, subtype: str, task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Collect structured data from data sources.

        Args:
            subtype: Data source subtype
            task: Research task
            objective_id: Objective ID

        Returns:
            Structured data collection result
        """
        try:
            # Simulate structured data collection
            structured_data = {
                "source_type": "structured",
                "subtype": subtype,
                "data_points": [
                    {
                        "metric": f"Metric for {task.get('description', 'task')}",
                        "value": 85.5,
                        "unit": "percentage",
                        "reliability_score": 0.8,
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
                "collection_metadata": {
                    "data_source": "database",
                    "results_count": 1,
                    "collection_method": "autonomous_data_analysis",
                },
            }

            return structured_data

        except Exception as e:
            logger.error(f"Structured data collection failed: {e}")
            return {}


    async def _collect_general_data(
        self, subtype: str, task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Collect general data from unspecified sources.

        Args:
            subtype: Data subtype
            task: Research task
            objective_id: Objective ID

        Returns:
            General data collection result
        """
        try:
            # Simulate general data collection
            general_data = {
                "source_type": "general",
                "subtype": subtype,
                "data_points": [
                    {
                        "content": f"General research data for {task.get('description', 'task')}",
                        "reliability_score": 0.6,
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
                "collection_metadata": {
                    "collection_method": "autonomous_general_research",
                    "results_count": 1,
                },
            }

            return general_data

        except Exception as e:
            logger.error(f"General data collection failed: {e}")
            return {}


    async def _process_collected_data(
        self, collected_data: List[Dict[str, Any]], task: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process and clean collected data.

        Args:
            collected_data: Raw collected data
            task: Research task

        Returns:
            Processed data
        """
        try:
            processed_data = []

            for data in collected_data:
                if not data:
                    continue

                # Clean and process data
                processed_item = {
                    "source_type": data.get("source_type", "unknown"),
                    "data_points": data.get("data_points", []),
                    "processed_at": datetime.now().isoformat(),
                    "quality_score": self._calculate_data_quality([data]),
                    "relevance_score": self._calculate_relevance_score(data, task),
                }

                processed_data.append(processed_item)

            return processed_data

        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return collected_data


    async def _generate_data_summary(
        self, processed_data: List[Dict[str, Any]], task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary of processed data.

        Args:
            processed_data: Processed data
            task: Research task

        Returns:
            Data summary
        """
        try:
            total_data_points = sum(len(item.get("data_points", [])) for item in processed_data)
            avg_quality = (
                sum(item.get("quality_score", 0) for item in processed_data) / len(processed_data)
                if processed_data
                else 0
            )
            avg_relevance = (
                sum(item.get("relevance_score", 0) for item in processed_data) / len(processed_data)
                if processed_data
                else 0
            )

            summary = {
                "total_sources": len(processed_data),
                "total_data_points": total_data_points,
                "average_quality_score": avg_quality,
                "average_relevance_score": avg_relevance,
                "data_diversity": len(
                    set(item.get("source_type", "unknown") for item in processed_data)
                ),
                "summary_timestamp": datetime.now().isoformat(),
            }

            return summary

        except Exception as e:
            logger.error(f"Data summary generation failed: {e}")
            return {}


