"""Analysis/synthesis/validation/general-research task pipelines for ResearchAgent.

Split out of the former monolithic research_agent.py (issue #582).

Note: this module intentionally contains two methods named
`_generate_research_summary` with different signatures (one taking
`query, search_results, collected_data`, one taking `research_result, task`).
This mirrors a pre-existing bug in the original monolith -- Python keeps only
the second definition in the class namespace, silently shadowing the first,
which is therefore unreachable dead code. Preserved as-is here to keep
before/after refactor behavior identical; see the follow-up cleanup issue
before removing either one.
"""

import json
from datetime import datetime
from typing import Any, Dict, List

from src.utils.logger import setup_logger

logger = setup_logger("research_agent", log_level="INFO")


class TaskPipelinesMixin:
    """Analysis, synthesis, validation, and general-research task pipelines."""

    async def _execute_analysis_task(
        self, task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Execute analysis task.

        Args:
            task: Analysis task
            objective_id: Objective ID

        Returns:
            Analysis result
        """
        try:
            logger.info(f"Executing analysis task: {task.get('task_id')}")

            # Determine analysis method based on task requirements
            analysis_method = await self._select_analysis_method(task, objective_id)

            # Perform analysis
            analysis_result = await self._perform_analysis(analysis_method, task, objective_id)

            # Generate insights
            insights = await self._generate_insights(analysis_result, task)

            # Create analysis report
            analysis_report = await self._create_analysis_report(analysis_result, insights, task)

            result = {
                "analysis_result": {
                    "method_used": analysis_method,
                    "analysis_quality": self._calculate_analysis_quality(analysis_result),
                    "insights_generated": len(insights),
                    "analysis_timestamp": datetime.now().isoformat(),
                },
                "analysis_data": analysis_result,
                "insights": insights,
                "analysis_report": analysis_report,
                "metadata": {
                    "task_description": task.get("description", ""),
                    "analysis_type": task.get("task_type", "analysis"),
                    "confidence_score": self._calculate_confidence_score(analysis_result),
                },
            }

            return result

        except Exception as e:
            logger.error(f"Analysis task failed: {e}")
            raise


    async def _execute_synthesis_task(
        self, task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Execute synthesis task.

        Args:
            task: Synthesis task
            objective_id: Objective ID

        Returns:
            Synthesis result
        """
        try:
            logger.info(f"Executing synthesis task: {task.get('task_id')}")

            # Gather data from previous tasks
            synthesis_data = await self._gather_synthesis_data(task, objective_id)

            # Perform synthesis
            synthesis_result = await self._perform_synthesis(synthesis_data, task)

            # Generate recommendations
            recommendations = await self._generate_recommendations(synthesis_result, task)

            # Create synthesis report
            synthesis_report = await self._create_synthesis_report(
                synthesis_result, recommendations, task
            )

            result = {
                "synthesis_result": {
                    "sources_synthesized": len(synthesis_data),
                    "synthesis_quality": self._calculate_synthesis_quality(synthesis_result),
                    "recommendations_generated": len(recommendations),
                    "synthesis_timestamp": datetime.now().isoformat(),
                },
                "synthesis_data": synthesis_result,
                "recommendations": recommendations,
                "synthesis_report": synthesis_report,
                "metadata": {
                    "task_description": task.get("description", ""),
                    "synthesis_type": task.get("task_type", "synthesis"),
                    "completeness_score": self._calculate_completeness_score(synthesis_result),
                },
            }

            return result

        except Exception as e:
            logger.error(f"Synthesis task failed: {e}")
            raise


    async def _execute_validation_task(
        self, task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Execute validation task.

        Args:
            task: Validation task
            objective_id: Objective ID

        Returns:
            Validation result
        """
        try:
            logger.info(f"Executing validation task: {task.get('task_id')}")

            # Gather data to validate
            validation_data = await self._gather_validation_data(task, objective_id)

            # Perform validation
            validation_result = await self._perform_validation(validation_data, task)

            # Generate validation report
            validation_report = await self._create_validation_report(validation_result, task)

            result = {
                "validation_result": {
                    "validation_score": validation_result.get("overall_score", 0.0),
                    "issues_found": len(validation_result.get("issues", [])),
                    "validation_timestamp": datetime.now().isoformat(),
                },
                "validation_data": validation_result,
                "validation_report": validation_report,
                "metadata": {
                    "task_description": task.get("description", ""),
                    "validation_type": task.get("task_type", "validation"),
                    "reliability_score": self._calculate_reliability_score(validation_result),
                },
            }

            return result

        except Exception as e:
            logger.error(f"Validation task failed: {e}")
            raise


    async def _execute_general_research_task(
        self, task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Execute general research task.

        Args:
            task: General research task
            objective_id: Objective ID

        Returns:
            General research result
        """
        try:
            logger.info(f"Executing general research task: {task.get('task_id')}")

            # Perform general research
            research_result = await self._perform_general_research(task, objective_id)

            # Generate research summary
            research_summary = await self._generate_research_summary(research_result, task)

            result = {
                "research_result": {
                    "research_scope": task.get("description", ""),
                    "research_quality": self._calculate_research_quality(research_result),
                    "research_timestamp": datetime.now().isoformat(),
                },
                "research_data": research_result,
                "research_summary": research_summary,
                "metadata": {
                    "task_description": task.get("description", ""),
                    "research_type": "general",
                    "comprehensiveness_score": self._calculate_comprehensiveness_score(
                        research_result
                    ),
                },
            }

            return result

        except Exception as e:
            logger.error(f"General research task failed: {e}")
            raise


    async def _select_analysis_method(self, task: Dict[str, Any], objective_id: str) -> str:
        """Select appropriate analysis method using LLM."""
        from src.core.llm_manager import TaskType, execute_llm_task

        prompt = f"""
        Analyze the following research task and select the most appropriate analysis method:
        
        Task: {task.get("description", "")}
        Task Type: {task.get("task_type", "analysis")}
        Objective ID: {objective_id}
        
        Select from: quantitative, qualitative, mixed, comparative, predictive
        
        Respond with only the method name.
        """

        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.ANALYSIS,
                system_message="You are an expert research analyst.",
            )
            method = result.content.strip().lower()
            return (
                method
                if method in ["quantitative", "qualitative", "mixed", "comparative", "predictive"]
                else "mixed"
            )
        except Exception as e:
            logger.error(f"Failed to select analysis method: {e}")
            return "mixed"


    async def _perform_analysis(
        self, method: str, task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Perform analysis using selected method with LLM."""
        from src.core.llm_manager import TaskType, execute_llm_task

        prompt = f"""
        Perform {method} analysis for the following research task:
        
        Task: {task.get("description", "")}
        Task Type: {task.get("task_type", "analysis")}
        Method: {method}
        
        Provide comprehensive analysis results in JSON format with keys: findings, data_points, patterns, conclusions.
        """

        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.DEEP_REASONING,
                system_message=f"You are an expert {method} research analyst.",
            )

            try:
                analysis_data = json.loads(result.content)
            except:
                analysis_data = {
                    "method": method,
                    "results": result.content,
                    "raw_response": result.content,
                }

            return analysis_data
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "method": method,
                "results": f"Analysis failed: {str(e)}",
                "error": str(e),
            }


    async def _generate_insights(
        self, analysis_result: Dict[str, Any], task: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from analysis using LLM."""
        from src.core.llm_manager import TaskType, execute_llm_task

        prompt = f"""
        Generate key insights from the following analysis results:
        
        Analysis: {json.dumps(analysis_result, ensure_ascii=False, indent=2)}
        Task: {task.get("description", "")}
        
        Provide 3-5 key insights as a JSON array of strings.
        """

        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.REASONING,
                system_message=self.config.prompts["search_execution"]["system_message"],
            )

            try:
                insights = json.loads(result.content)
                if isinstance(insights, list):
                    return insights
                else:
                    return [insights] if isinstance(insights, str) else [result.content]
            except:
                # Fallback: parse as newline-separated list
                insights = [line.strip() for line in result.content.split("\n") if line.strip()]
                return insights[:5]  # Max 5 insights
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return [f"Analysis completed for: {task.get('description', 'task')}"]


    async def _create_analysis_report(
        self, analysis_result: Dict[str, Any], insights: List[str], task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create analysis report using LLM."""
        from src.core.llm_manager import TaskType, execute_llm_task

        prompt = f"""
        Create a comprehensive analysis report:
        
        Analysis Result: {json.dumps(analysis_result, ensure_ascii=False, indent=2)}
        Key Insights: {json.dumps(insights, ensure_ascii=False)}
        Task: {task.get("description", "")}
        
        Provide a structured report in JSON format with sections: executive_summary, key_findings, detailed_analysis, conclusions.
        """

        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.GENERATION,
                system_message=self.config.prompts["content_analysis"]["system_message"],
            )

            try:
                report = json.loads(result.content)
            except:
                report = {"report": result.content, "insights": insights}

            return report
        except Exception as e:
            logger.error(f"Report creation failed: {e}")
            return {
                "report": f"Report generation failed: {str(e)}",
                "insights": insights,
            }


    async def _gather_synthesis_data(
        self, task: Dict[str, Any], objective_id: str
    ) -> List[Dict[str, Any]]:
        """Gather data from previous tasks for synthesis."""
        # Get data from shared memory or previous execution results
        try:
            from src.core.shared_memory import get_shared_memory

            memory = get_shared_memory()

            # Search for related data
            query = task.get("description", "")
            related_data = memory.search(query, limit=10)

            return related_data if related_data else []
        except Exception as e:
            logger.error(f"Failed to gather synthesis data: {e}")
            return []


    async def _perform_synthesis(
        self, data: List[Dict[str, Any]], task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform synthesis using LLM."""
        from src.core.llm_manager import TaskType, execute_llm_task

        prompt = f"""
        Synthesize the following data into a cohesive analysis:
        
        Data: {json.dumps(data, ensure_ascii=False, indent=2)}
        Task: {task.get("description", "")}
        
        Provide synthesis in JSON format with keys: integrated_findings, themes, connections, synthesis_summary.
        """

        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.SYNTHESIS,
                system_message=self.config.prompts["synthesis_report"]["system_message"],
            )

            try:
                synthesis = json.loads(result.content)
            except:
                synthesis = {"synthesis": result.content}

            return synthesis
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {"synthesis": f"Synthesis failed: {str(e)}", "error": str(e)}


    async def _generate_recommendations(
        self, synthesis_result: Dict[str, Any], task: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations using LLM."""
        from src.core.llm_manager import TaskType, execute_llm_task

        prompt = f"""
        Generate actionable recommendations based on:
        
        Synthesis: {json.dumps(synthesis_result, ensure_ascii=False, indent=2)}
        Task: {task.get("description", "")}
        
        Provide 3-5 recommendations as a JSON array of strings.
        """

        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.REASONING,
                system_message=self.config.prompts["recommendation_generation"]["system_message"],
            )

            try:
                recommendations = json.loads(result.content)
                if isinstance(recommendations, list):
                    return recommendations
                else:
                    return (
                        [recommendations] if isinstance(recommendations, str) else [result.content]
                    )
            except:
                recommendations = [
                    line.strip() for line in result.content.split("\n") if line.strip()
                ]
                return recommendations[:5]
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return [f"Complete task: {task.get('description', 'task')}"]


    async def _create_synthesis_report(
        self,
        synthesis_result: Dict[str, Any],
        recommendations: List[str],
        task: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create synthesis report using LLM."""
        from src.core.llm_manager import TaskType, execute_llm_task

        prompt = f"""
        Create a synthesis report:
        
        Synthesis: {json.dumps(synthesis_result, ensure_ascii=False, indent=2)}
        Recommendations: {json.dumps(recommendations, ensure_ascii=False)}
        Task: {task.get("description", "")}
        
        Provide report in JSON format with sections: summary, key_points, recommendations, next_steps.
        """

        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.GENERATION,
                system_message="You are an expert report writer.",
            )

            try:
                report = json.loads(result.content)
            except:
                report = {"report": result.content, "recommendations": recommendations}

            return report
        except Exception as e:
            logger.error(f"Synthesis report creation failed: {e}")
            return {
                "report": f"Report generation failed: {str(e)}",
                "recommendations": recommendations,
            }


    async def _gather_validation_data(
        self, task: Dict[str, Any], objective_id: str
    ) -> List[Dict[str, Any]]:
        """Gather data for validation."""
        try:
            from src.core.shared_memory import get_shared_memory

            memory = get_shared_memory()

            query = task.get("description", "")
            related_data = memory.search(query, limit=10)

            return related_data if related_data else []
        except Exception as e:
            logger.error(f"Failed to gather validation data: {e}")
            return []


    async def _perform_validation(
        self, data: List[Dict[str, Any]], task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform validation using LLM."""
        from src.core.llm_manager import TaskType, execute_llm_task

        prompt = f"""
        Validate the following research data:
        
        Data: {json.dumps(data, ensure_ascii=False, indent=2)}
        Task: {task.get("description", "")}
        
        Provide validation in JSON format with keys: overall_score (0-1), issues (array), strengths (array), validation_summary.
        """

        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.VERIFICATION,
                system_message=self.config.prompts["quality_validation"]["system_message"],
            )

            try:
                validation = json.loads(result.content)
                if "overall_score" not in validation:
                    validation["overall_score"] = 0.8  # Default score
            except:
                validation = {
                    "overall_score": 0.8,
                    "issues": [],
                    "summary": result.content,
                }

            return validation
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"overall_score": 0.5, "issues": [str(e)], "error": str(e)}


    async def _create_validation_report(
        self, validation_result: Dict[str, Any], task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create validation report using LLM."""
        from src.core.llm_manager import TaskType, execute_llm_task

        prompt = f"""
        Create a validation report:
        
        Validation Result: {json.dumps(validation_result, ensure_ascii=False, indent=2)}
        Task: {task.get("description", "")}
        
        Provide report in JSON format with sections: validation_score, issues_found, strengths, recommendations.
        """

        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.GENERATION,
                system_message=self.config.prompts["final_assessment"]["system_message"],
            )

            try:
                report = json.loads(result.content)
            except:
                report = {
                    "report": result.content,
                    "score": validation_result.get("overall_score", 0.0),
                }

            return report
        except Exception as e:
            logger.error(f"Validation report creation failed: {e}")
            return {
                "report": f"Report generation failed: {str(e)}",
                "score": validation_result.get("overall_score", 0.0),
            }


    async def _generate_research_summary(
        self,
        query: str,
        search_results: Dict[str, Any],
        collected_data: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive research summary using LLM."""
        try:
            if not self.llm:
                return {
                    "summary": "LLM not available",
                    "key_points": [],
                    "insights": [],
                }

            # Prepare data for LLM analysis
            data_summary = {
                "query": query,
                "search_success": (
                    search_results.get("success", False)
                    if isinstance(search_results, dict)
                    else False
                ),
                "results_count": (
                    len(search_results.get("results", []))
                    if isinstance(search_results, dict)
                    else 0
                ),
                "collected_data_count": len(collected_data) if collected_data else 0,
                "research_data": (
                    search_results.get("research_data", {})
                    if isinstance(search_results, dict)
                    else {}
                ),
            }

            prompt = f"""
            Based on the research query: "{query}"
            
            Research data summary:
            - Search successful: {data_summary["search_success"]}
            - Results found: {data_summary["results_count"]}
            - Data collected: {data_summary["collected_data_count"]}
            - Research data: {json.dumps(data_summary["research_data"], ensure_ascii=False, indent=2)}
            
            Generate a comprehensive research summary including:
            1. Executive summary of findings
            2. Key insights and trends
            3. Important statistics and data points
            4. Expert analysis and opinions
            5. Future implications and recommendations
            
            Format as JSON:
            {{
                "executive_summary": "comprehensive summary",
                "key_insights": ["insight1", "insight2", "insight3"],
                "statistics": {{"metric1": "value1", "metric2": "value2"}},
                "expert_analysis": "detailed analysis",
                "future_implications": ["implication1", "implication2"],
                "recommendations": ["recommendation1", "recommendation2"]
            }}
            """

            response_text = await self._call_llm_with_retry(prompt)

            # Try to parse JSON, with fallback
            try:
                summary_data = json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a basic structure
                summary_data = {
                    "executive_summary": f"Research summary for {query}",
                    "key_insights": [f"Key insight about {query}"],
                    "statistics": {"metric1": "value1"},
                    "expert_analysis": f"Expert analysis of {query}",
                    "future_implications": [f"Future implication of {query}"],
                    "recommendations": [f"Recommendation for {query}"],
                }

            return summary_data

        except Exception as e:
            logger.error(f"Research summary generation failed: {e}")
            return {
                "executive_summary": f"Research summary generation failed: {e}",
                "key_insights": [],
                "statistics": {},
                "expert_analysis": "Analysis unavailable",
                "future_implications": [],
                "recommendations": [],
            }


    async def _perform_general_research(
        self, task: Dict[str, Any], objective_id: str
    ) -> Dict[str, Any]:
        """Perform general research with real data collection."""
        try:
            if not task or not isinstance(task, dict):
                task = {}
            task_description = task.get("description", "")
            if not task_description:
                task_description = "general research"
            research_query = str(task_description).replace("Research task for objective: ", "")

            # 1. Perform web search
            search_results = await self._perform_web_search(research_query)

            # 2. Collect additional data from URLs
            collected_data = []
            if (
                isinstance(search_results, dict)
                and search_results.get("success")
                and search_results.get("results")
            ):
                results = search_results.get("results", [])
                if isinstance(results, list):
                    for result in results[:3]:  # Limit to top 3 results
                        if isinstance(result, dict):
                            try:
                                # For now, just use the snippet as content since we don't have URLs
                                content = result.get("snippet", "")
                                if content:
                                    collected_data.append(
                                        {
                                            "url": result.get("url", ""),
                                            "title": result.get("title", ""),
                                            "content": content,
                                            "source": result.get("source", "unknown"),
                                        }
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to process result: {e}")
                                continue

            # 3. Perform additional research using different methods
            additional_research = await self._perform_additional_research(research_query)
            if not isinstance(additional_research, dict):
                additional_research = {"sources": []}

            # 4. Generate comprehensive research summary
            research_summary = await self._generate_research_summary(
                {
                    "query": research_query,
                    "collected_data": collected_data,
                    "additional_research": additional_research,
                    "research_quality": 0.8,
                },
                task,
            )

            # Ensure research_summary is a dict
            if not isinstance(research_summary, dict):
                research_summary = {
                    "summary": "Research summary generation failed",
                    "error": "Invalid summary format",
                }

            # 5. Compile research results
            research_result = {
                "task_id": task.get("task_id", "unknown"),
                "agent": "researcher",
                "task_type": task.get("task_type", "general"),
                "result": {
                    "research_data": {
                        "query": research_query,
                        "web_search_results": search_results,
                        "collected_data": collected_data,
                        "additional_research": additional_research,
                        "research_summary": research_summary,
                        "total_sources": len(collected_data)
                        + len(additional_research.get("sources", [])),
                        "research_timestamp": datetime.now().isoformat(),
                    },
                    "sources": [
                        item.get("url", "")
                        for item in collected_data
                        if isinstance(item, dict) and item.get("url")
                    ],
                    "metadata": {
                        "research_method": "web_search_and_analysis",
                        "data_quality": "high" if len(collected_data) > 0 else "low",
                        "completeness": "partial" if len(collected_data) < 3 else "complete",
                    },
                },
            }

            return research_result

        except Exception as e:
            logger.error(f"General research failed: {e}")
            return {"research": f"Research failed: {str(e)}", "error": str(e)}


    async def _fetch_web_content(self, url: str) -> str:
        """Fetch content from a web URL using MCP tools."""
        try:
            # Use MCP tools for content fetching
            from src.core.mcp_integration import execute_tool

            result = await execute_tool("fetch", {"url": url})

            if result.get("success", False):
                content = result.get("data", {}).get("content", "")

                # Basic HTML tag removal
                import re

                text = re.sub(r"<[^>]+>", "", content)
                text = re.sub(r"\s+", " ", text)

                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = " ".join(chunk for chunk in chunks if chunk)

                # Limit content length
                return text[:5000] if len(text) > 5000 else text
            else:
                logger.error(f"MCP fetch failed for {url}: {result.get('error', 'Unknown error')}")
                return ""

        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return ""


    async def _perform_additional_research(self, query: str) -> Dict[str, Any]:
        """Perform additional research using different methods."""
        try:
            additional_sources = []

            # 1. Try academic sources
            try:
                academic_results = await self._search_academic_sources(query)
                additional_sources.extend(academic_results)
            except Exception as e:
                logger.warning(f"Academic search failed: {e}")

            # 2. Try news sources
            try:
                news_results = await self._search_news_sources(query)
                additional_sources.extend(news_results)
            except Exception as e:
                logger.warning(f"News search failed: {e}")

            # 3. Try social media sources
            try:
                social_results = await self._search_social_sources(query)
                additional_sources.extend(social_results)
            except Exception as e:
                logger.warning(f"Social media search failed: {e}")

            return {
                "sources": additional_sources,
                "total_additional_sources": len(additional_sources),
                "research_methods": ["academic", "news", "social_media"],
            }

        except Exception as e:
            logger.error(f"Additional research failed: {e}")
            return {
                "sources": [],
                "total_additional_sources": 0,
                "research_methods": [],
            }


    async def _search_academic_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search academic sources."""
        try:
            # Try Google Scholar
            scholar_query = f"site:scholar.google.com {query}"
            search_results = await self._perform_web_search(scholar_query)

            academic_sources = []
            if search_results.get("success") and search_results.get("results"):
                for result in search_results["results"][:2]:
                    academic_sources.append(
                        {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "snippet": result.get("snippet", ""),
                            "source_type": "academic",
                            "search_engine": "google_scholar",
                        }
                    )

            return academic_sources

        except Exception as e:
            logger.error(f"Academic search failed: {e}")
            return []


    async def _search_news_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search news sources."""
        try:
            # Try news-specific search
            news_query = f"{query} news"
            search_results = await self._perform_web_search(news_query)

            news_sources = []
            if search_results.get("success") and search_results.get("results"):
                for result in search_results["results"][:2]:
                    news_sources.append(
                        {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "snippet": result.get("snippet", ""),
                            "source_type": "news",
                            "search_engine": "general",
                        }
                    )

            return news_sources

        except Exception as e:
            logger.error(f"News search failed: {e}")
            return []


    async def _search_social_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search social media sources."""
        try:
            # Try social media search
            social_query = f"{query} site:twitter.com OR site:reddit.com OR site:linkedin.com"
            search_results = await self._perform_web_search(social_query)

            social_sources = []
            if search_results.get("success") and search_results.get("results"):
                for result in search_results["results"][:2]:
                    social_sources.append(
                        {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "snippet": result.get("snippet", ""),
                            "source_type": "social_media",
                            "search_engine": "general",
                        }
                    )

            return social_sources

        except Exception as e:
            logger.error(f"Social media search failed: {e}")
            return []


    async def _generate_research_summary(
        self, research_result: Dict[str, Any], task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate research summary from actual data."""
        try:
            # Ensure research_result is a dictionary
            if not isinstance(research_result, dict):
                research_result = {
                    "query": str(research_result),
                    "collected_data": [],
                    "additional_research": {},
                }

            # Extract key information from research results with type safety
            query = research_result.get("query", "")

            # Ensure collected_data is a list
            collected_data = research_result.get("collected_data", [])
            if not isinstance(collected_data, list):
                collected_data = []

            # Ensure additional_research is a dict
            additional_research = research_result.get("additional_research", {})
            if not isinstance(additional_research, dict):
                additional_research = {}

            # Generate summary based on collected data
            summary_parts = []

            if collected_data:
                summary_parts.append(
                    f"Found {len(collected_data)} primary sources with detailed content."
                )

                # Extract key insights from collected data
                key_insights = []
                for item in collected_data[:2]:  # Top 2 sources
                    if isinstance(item, dict):
                        title = item.get("title", "")
                        content = item.get("content", "")
                        if title and content:
                            # Extract first 200 characters as insight
                            insight = content[:200] + "..." if len(content) > 200 else content
                            key_insights.append(f"• {title}: {insight}")

                if key_insights:
                    summary_parts.append("Key insights:")
                    summary_parts.extend(key_insights)

            # Ensure additional_sources is a list
            additional_sources = additional_research.get("sources", [])
            if not isinstance(additional_sources, list):
                additional_sources = []

            if additional_sources:
                summary_parts.append(
                    f"Found {len(additional_sources)} additional sources from academic, news, and social media."
                )

            # Calculate research metrics
            total_sources = len(collected_data) + len(additional_sources)
            research_quality = research_result.get("research_quality", 0.5)

            summary_parts.append(f"Research quality: {research_quality:.1%}")
            summary_parts.append(f"Total sources: {total_sources}")

            summary_text = "\n".join(summary_parts)

            return {
                "summary": summary_text,
                "key_insights": key_insights if "key_insights" in locals() else [],
                "total_sources": total_sources,
                "research_quality": research_quality,
                "summary_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Research summary generation failed: {e}")
            return {"summary": f"Summary generation failed: {str(e)}", "error": str(e)}


