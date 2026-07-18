"""Browser automation methods for ResearchAgent.

Split out of the former monolithic research_agent.py (issue #582).
"""

from datetime import datetime
from typing import Any, Dict, List

from src.automation.browser_manager import BrowserManager
from src.utils.logger import setup_logger

logger = setup_logger("research_agent", log_level="INFO")


class BrowserAutomationMixin:
    """Browser-driven navigation, search, and interactive research."""

    async def browser_navigate_and_extract(self, url: str, extraction_goal: str) -> Dict[str, Any]:
        """Navigate to URL and extract content using enhanced browser manager.

        Args:
            url: URL to navigate to
            extraction_goal: Specific goal for content extraction

        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            # Initialize browser manager if not already done
            if self.browser_manager is None:
                self.browser_manager = BrowserManager()

            # Initialize browser if not already done
            if not self.browser_manager.browser_available:
                await self.browser_manager.initialize_browser()

            # Use browser manager for extraction
            result = await self.browser_manager.navigate_and_extract(url, extraction_goal, self.llm)

            logger.info(
                f"Content extraction completed for {url} using {result.get('method', 'unknown')} method"
            )
            return result

        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return {
                "success": False,
                "url": url,
                "extraction_goal": extraction_goal,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    # Fallback methods are now handled by BrowserManager


    async def browser_search_and_extract(
        self, query: str, extraction_goal: str, max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Perform web search and extract content from results using enhanced browser manager.

        Args:
            query: Search query
            extraction_goal: Goal for content extraction
            max_results: Maximum number of results to process

        Returns:
            List of extracted content from search results
        """
        try:
            # Initialize browser manager if not already done
            if self.browser_manager is None:
                self.browser_manager = BrowserManager()

            # Initialize browser if not already done
            if not self.browser_manager.browser_available:
                await self.browser_manager.initialize_browser()

            # Use browser manager for search and extract
            results = await self.browser_manager.search_and_extract(
                query, extraction_goal, max_results, self.llm
            )

            logger.info(
                f"Search and extract completed for query '{query}' with {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"Search and extract failed: {e}")
            return []


    async def browser_interactive_research(
        self, research_plan: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform interactive research using browser automation.

        Args:
            research_plan: List of research steps with actions and goals

        Returns:
            Dictionary containing research results
        """
        try:
            # Initialize browser manager if not already done
            if self.browser_manager is None:
                self.browser_manager = BrowserManager()

            # Initialize browser if not already done
            if not self.browser_manager.browser_available:
                await self.browser_manager.initialize_browser()

            research_results = {
                "plan": research_plan,
                "steps_completed": [],
                "data_collected": [],
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }

            for step in research_plan:
                step_result = await self._execute_research_step(step)
                research_results["steps_completed"].append(step_result)

                if step_result.get("data_collected"):
                    research_results["data_collected"].extend(step_result["data_collected"])

            return research_results

        except Exception as e:
            logger.error(f"Interactive research failed: {e}")
            return {
                "plan": research_plan,
                "steps_completed": [],
                "data_collected": [],
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


    async def cleanup_browser(self):
        """Clean up browser resources using browser manager."""
        try:
            if self.browser_manager is not None:
                await self.browser_manager.cleanup()
                logger.info("Browser resources cleaned up")

        except Exception as e:
            logger.error(f"Browser cleanup failed: {e}")

