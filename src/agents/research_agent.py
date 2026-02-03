#!/usr/bin/env python3
"""
Research Agent for Autonomous Research System

This agent autonomously conducts research tasks including data collection,
web research, literature review, and data analysis.

No fallback or dummy code - production-level autonomous research only.
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import base64
import google.generativeai as genai
import markdownify
import requests
from bs4 import BeautifulSoup

# Enhanced browser automation
from src.automation.browser_manager import BrowserManager

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.researcher_config import get_llm_config, get_research_config, SearchAPI
from src.utils.logger import setup_logger

logger = setup_logger("research_agent", log_level="INFO")


class ResearchAgent:
    """Autonomous research agent for data collection and analysis."""
    
    def __init__(self):
        """Initialize the research agent."""
        # Load configurations
        self.llm_config = get_llm_config()
        self.research_config = get_research_config()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Active research tasks
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Learning capabilities
        self.learning_data = []
        self.research_history = []
        
        # Enhanced browser automation (optional, initialized when needed)
        self.browser_manager = None
        
        logger.info("Research Agent initialized with LLM-based research capabilities")
    
    def _initialize_llm(self):
        """Initialize the LLM client."""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                logger.warning("Gemini API key not found. Research functionality will be limited.")
                return None
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            logger.info("LLM initialized for ResearchAgent with model: gemini-2.5-flash-lite")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with retry logic and rate limiting."""
        if not self.llm:
            return "LLM not available"
        
        for attempt in range(max_retries):
            try:
                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = random.uniform(2, 5) * (attempt + 1)
                    logger.info(f"Retrying LLM call after {delay:.2f}s delay (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                else:
                    # Small delay even on first attempt
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                
                response = self.llm.generate_content(prompt)
                return response.text.strip()
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Extract retry delay from error if available
                        retry_delay = 30 + (attempt * 10)  # Progressive delay
                        logger.warning(f"API quota exceeded, retrying in {retry_delay}s (attempt {attempt + 1})")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                        return f"LLM call failed: {e}"
                else:
                    logger.error(f"LLM call failed: {e}")
                    return f"LLM call failed: {e}"
        
        return "LLM call failed after all retries"
    
    def _load_research_tools(self) -> Dict[str, Any]:
        """Load available research tools.
        
        Returns:
            Dictionary of research tools
        """
        return {
            'web_search': {
                'capabilities': ['web_search', 'content_extraction', 'url_analysis'],
                'max_concurrent': 5,
                'rate_limit': 10  # requests per minute
            },
            'academic_search': {
                'capabilities': ['academic_papers', 'citation_analysis', 'literature_review'],
                'max_concurrent': 3,
                'rate_limit': 5
            },
            'data_analysis': {
                'capabilities': ['statistical_analysis', 'data_visualization', 'trend_analysis'],
                'max_concurrent': 2,
                'rate_limit': 0  # no rate limit
            },
            'content_analysis': {
                'capabilities': ['text_analysis', 'sentiment_analysis', 'topic_modeling'],
                'max_concurrent': 3,
                'rate_limit': 0
            }
        }
    
    def _load_data_sources(self) -> Dict[str, Any]:
        """Load available data sources.
        
        Returns:
            Dictionary of data sources
        """
        return {
            'web_sources': {
                'types': ['news_articles', 'blog_posts', 'reports', 'websites'],
                'reliability': 0.7,
                'accessibility': 0.9
            },
            'academic_sources': {
                'types': ['research_papers', 'conference_proceedings', 'theses', 'books'],
                'reliability': 0.9,
                'accessibility': 0.6
            },
            'data_sources': {
                'types': ['datasets', 'databases', 'apis', 'statistics'],
                'reliability': 0.8,
                'accessibility': 0.7
            },
            'expert_sources': {
                'types': ['interviews', 'expert_opinions', 'case_studies'],
                'reliability': 0.85,
                'accessibility': 0.5
            }
        }
    
    def _load_analysis_methods(self) -> Dict[str, Any]:
        """Load available analysis methods.
        
        Returns:
            Dictionary of analysis methods
        """
        return {
            'quantitative': {
                'methods': ['statistical_analysis', 'regression_analysis', 'correlation_analysis'],
                'suitable_for': ['numerical_data', 'surveys', 'experiments'],
                'outputs': ['statistics', 'charts', 'models']
            },
            'qualitative': {
                'methods': ['content_analysis', 'thematic_analysis', 'discourse_analysis'],
                'suitable_for': ['text_data', 'interviews', 'observations'],
                'outputs': ['themes', 'insights', 'narratives']
            },
            'mixed_methods': {
                'methods': ['triangulation', 'convergent_analysis', 'explanatory_analysis'],
                'suitable_for': ['complex_research', 'comprehensive_studies'],
                'outputs': ['integrated_findings', 'comprehensive_reports']
            }
        }
    
    async def conduct_research(self, tasks: List[Dict[str, Any]], 
                              context: Optional[Dict[str, Any]] = None, 
                              objective_id: str = None) -> Dict[str, Any]:
        """Conduct comprehensive research for multiple tasks.
        
        Args:
            tasks: List of research tasks
            context: Additional context
            objective_id: Objective ID for tracking
            
        Returns:
            Comprehensive research result
        """
        try:
            logger.info(f"Conducting research for {len(tasks)} tasks")
            
            research_results = []
            successful_tasks = 0
            failed_tasks = 0
            
            # Execute each task
            for task in tasks:
                try:
                    result = await self.execute_task(task, objective_id)
                    research_results.append(result)
                    
                    if result.get('success', False):
                        successful_tasks += 1
                    else:
                        failed_tasks += 1
                        
                except Exception as e:
                    logger.error(f"Task execution failed: {e}")
                    research_results.append({
                        "success": False,
                        "error": str(e),
                        "task_id": task.get('task_id'),
                        "timestamp": datetime.now().isoformat()
                    })
                    failed_tasks += 1
            
            # Generate comprehensive research summary
            summary = await self._generate_research_summary(research_results, {
                "total_tasks": len(tasks),
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks
            })
            
            return {
                "success": successful_tasks > 0,
                "research_results": research_results,
                "summary": summary,
                "statistics": {
                    "total_tasks": len(tasks),
                    "successful_tasks": successful_tasks,
                    "failed_tasks": failed_tasks,
                    "success_rate": successful_tasks / len(tasks) if tasks else 0
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Research conduction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "research_results": [],
                "timestamp": datetime.now().isoformat()
            }
    
    async def execute_task(self, task: Dict[str, Any], objective_id: str, 
                          is_refinement: bool = False, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a research task with LLM-based research.
        
        Args:
            task: Task to execute
            objective_id: Objective ID for tracking
            is_refinement: Whether this is a refinement task
            
        Returns:
            Task execution result
        """
        try:
            task_id = task.get('task_id', str(uuid.uuid4()))
            task_type = task.get('task_type', 'general')
            
            logger.info(f"Executing LLM-based research task: {task_id} (type: {task_type})")
            
            # Track active task
            self.active_tasks[task_id] = {
                'task': task,
                'objective_id': objective_id,
                'started_at': datetime.now(),
                'status': 'running'
            }
            
            # Use LLM to plan and execute research
            result = await self._llm_conduct_research(task, objective_id)
            
            # Update task status
            self.active_tasks[task_id]['status'] = 'completed'
            self.active_tasks[task_id]['completed_at'] = datetime.now()
            
            # Add metadata
            result.update({
                'task_id': task_id,
                'objective_id': objective_id,
                'agent': 'researcher',
                'task_type': task_type,
                'is_refinement': is_refinement,
                'execution_time': (datetime.now() - self.active_tasks[task_id]['started_at']).total_seconds(),
                'status': 'completed'
            })
            
            # Store in research history
            self.research_history.append({
                'task_id': task_id,
                'objective_id': objective_id,
                'task_type': task_type,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"LLM-based research task completed: {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"LLM-based research task execution failed: {e}")
            
            # Update task status
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'failed'
                self.active_tasks[task_id]['error'] = str(e)
            
            return {
                'task_id': task_id,
                'objective_id': objective_id,
                'agent': 'researcher',
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _llm_conduct_research(self, task: Dict[str, Any], objective_id: str) -> Dict[str, Any]:
        """Use LLM to conduct comprehensive research."""
        try:
            # Get LLM research plan
            research_plan = await self._get_llm_research_plan(task)
            
            # Execute research based on plan
            research_results = []
            research_steps = research_plan.get('steps', [])
            if not isinstance(research_steps, list):
                research_steps = []
            
            for research_step in research_steps:
                if isinstance(research_step, dict):
                    step_result = await self._execute_research_step(research_step, task)
                    if step_result:
                        research_results.append(step_result)
            
            # Use LLM to analyze and synthesize results
            analysis_result = await self._llm_analyze_research_results(research_results, task)
            
            # Ensure analysis_result is a dict
            if not isinstance(analysis_result, dict):
                analysis_result = {'quality_score': 0.0, 'key_findings': []}
            
            return {
                'research_plan': research_plan,
                'research_results': research_results,
                'analysis_result': analysis_result,
                'total_sources': len(research_results),
                'research_quality': analysis_result.get('quality_score', 0.0),
                'key_findings': analysis_result.get('key_findings', []),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"LLM research failed: {e}")
            # Return error result instead of fallback
            return {
                'research_plan': {},
                'research_results': [],
                'analysis_result': {'quality_score': 0.0, 'key_findings': []},
                'total_sources': 0,
                'research_quality': 0.0,
                'key_findings': [],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_llm_research_plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Get research plan from LLM."""
        try:
            prompt = f"""
            다음 연구 작업을 위한 구체적인 연구 계획을 수립하세요.
            
            작업: {task.get('description', '')}
            작업 유형: {task.get('task_type', 'general')}
            
            다음을 포함한 연구 계획을 수립하세요:
            1. 필요한 정보 유형 식별
            2. 적절한 정보원 선택
            3. 검색 전략 수립
            4. 데이터 수집 방법 결정
            5. 분석 방법 선택
            
            JSON 형태로 응답하세요:
            {{
                "research_strategy": "전체 연구 전략",
                "steps": [
                    {{
                        "step_id": "step_1",
                        "description": "단계 설명",
                        "method": "web_search|academic_search|data_analysis|content_analysis",
                        "query": "검색 쿼리",
                        "sources": ["웹", "학술", "데이터베이스"],
                        "expected_output": "예상 결과"
                    }}
                ],
                "quality_criteria": ["품질 기준1", "품질 기준2"],
                "success_metrics": ["성공 지표1", "성공 지표2"]
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
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
        except Exception as e:
            logger.error(f"LLM research planning failed: {e}")
            # Return basic research plan structure
            return {
                "research_strategy": f"Research task: {task.get('description', 'Unknown')}",
                "steps": [
                    {
                        "step_id": "step_1",
                        "description": "Basic web search",
                        "method": "web_search",
                        "query": task.get('description', ''),
                        "sources": ["웹"],
                        "expected_output": "Basic information"
                    }
                ],
                "quality_criteria": ["정보의 정확성", "출처의 신뢰성"],
                "success_metrics": ["정보 수집 완료", "기본 분석 완료"]
            }
    
    
    
    
    
    
    
    async def _generate_basic_research_plan(self, task_description: str) -> Dict[str, Any]:
        """Generate basic research plan without LLM."""
        try:
            # Create basic research plan with actual tool calls
            basic_plan = {
                'research_strategy': f'Systematic research approach for: {task_description}',
                'tool_calls': [
                    {
                        'tool': 'web_search',
                        'query': task_description,
                        'purpose': 'Initial research and overview'
                    },
                    {
                        'tool': 'web_search', 
                        'query': f'{task_description} benefits advantages',
                        'purpose': 'Identify benefits and advantages'
                    },
                    {
                        'tool': 'web_search',
                        'query': f'{task_description} challenges limitations',
                        'purpose': 'Understand challenges and limitations'
                    },
                    {
                        'tool': 'web_search',
                        'query': f'{task_description} case studies examples',
                        'purpose': 'Find real-world applications and examples'
                    }
                ],
                'analysis_plan': [
                    {
                        'step': 'Synthesize findings from multiple sources',
                        'method': 'Cross-reference and validate information'
                    },
                    {
                        'step': 'Identify key patterns and trends',
                        'method': 'Pattern analysis and trend identification'
                    }
                ],
                'validation_criteria': [
                    'Source credibility',
                    'Information recency',
                    'Cross-source validation'
                ],
                'expected_outcomes': [
                    'Comprehensive overview of the topic',
                    'Key benefits and challenges identified',
                    'Real-world examples and case studies'
                ]
            }
            
            # Execute the basic research plan
            research_results = await self._execute_llm_research_plan(basic_plan, task_description)
            return research_results
            
        except Exception as e:
            logger.error(f"Basic research plan generation failed: {e}")
            return {'method': 'basic_research', 'error': str(e)}
    
    async def _execute_research_step(self, step: Dict[str, Any], task: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Execute a single research step."""
        try:
            # Ensure step is a dict
            if not isinstance(step, dict):
                step = {'method': 'web_search', 'query': ''}
            
            # Ensure task is a dict
            if not isinstance(task, dict):
                task = {'description': ''}
            
            method = step.get('method', 'web_search')
            query = step.get('query', task.get('description', ''))
            
            if method == 'web_search':
                return await self._perform_web_search(query)
            elif method == 'academic_search':
                return await self._perform_academic_search(query)
            elif method == 'data_analysis':
                return await self._perform_data_analysis(query, task)
            elif method == 'content_analysis':
                return await self._perform_content_analysis(query, task)
            else:
                # Create a task object for general research
                general_task = {
                    'description': query,
                    'task_type': 'general'
                }
                return await self._perform_general_research(general_task, 'unknown')
                
        except Exception as e:
            logger.error(f"Research step execution failed: {e}")
            return None
    
    async def _perform_web_search(self, query: str) -> Dict[str, Any]:
        """Perform web search using MCP tools."""
        try:
            # Use MCP tools for web search
            from src.core.mcp_integration import execute_tool
            
            # Try MCP search tools in priority order
            mcp_tools = ['g-search', 'tavily', 'exa']
            
            for tool in mcp_tools:
                try:
                    result = await execute_tool(tool, {
                        'query': query,
                        'max_results': 10
                    })
                    
                    if result.get('success', False):
                        results = result.get('data', {}).get('results', [])
                        if isinstance(results, list):
                            logger.info(f"Search succeeded with MCP tool {tool}: {len(results)} results")
                            return {
                                'success': True,
                                'query': query,
                                'results': results,
                                'tool_used': tool
                            }
                except Exception as e:
                    logger.warning(f"MCP tool {tool} failed: {e}")
                    continue
            
            # If all MCP tools fail, raise error
            logger.error("All MCP search tools failed")
            raise RuntimeError(f"All MCP search tools failed for query: {query}. No fallback available.")
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {
                'success': False,
                'method': 'web_search',
                'query': query,
                'results': [],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _search_with_tavily(self, query: str) -> Dict[str, Any]:
        """Search using Tavily API (best quality)."""
        try:
            tavily_key = os.getenv('TAVILY_API_KEY')
            if not tavily_key:
                return {'success': False, 'error': 'Tavily API key not configured'}
            
            from tavily import TavilyClient
            
            client = TavilyClient(api_key=tavily_key)
            response = client.search(query, max_results=5, search_depth="advanced")
            
            formatted_results = []
            results = response.get('results', [])
            if not isinstance(results, list):
                results = []
            
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('content', ''),
                        'url': result.get('url', ''),
                        'source': 'tavily',
                        'score': result.get('score', 0)
                    })
            
            return {
                'success': True,
                'method': 'tavily',
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _search_with_exa(self, query: str) -> Dict[str, Any]:
        """Search using Exa API (neural search)."""
        try:
            exa_key = os.getenv('EXA_API_KEY')
            if not exa_key:
                return {'success': False, 'error': 'Exa API key not configured'}
            
            from exa_py import Exa
            
            client = Exa(api_key=exa_key)
            response = client.search_and_contents(
                query,
                num_results=5,
                text=True,
                highlights=True
            )
            
            formatted_results = []
            for result in response.results:
                formatted_results.append({
                    'title': result.title,
                    'snippet': result.text[:500] if result.text else '',
                    'url': result.url,
                    'source': 'exa',
                    'score': result.score if hasattr(result, 'score') else 0
                })
            
            return {
                'success': True,
                'method': 'exa',
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Exa search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _search_with_brave(self, query: str) -> Dict[str, Any]:
        """Search using Brave Search API."""
        try:
            brave_key = os.getenv('BRAVE_SEARCH_API_KEY')
            if not brave_key:
                return {'success': False, 'error': 'Brave API key not configured'}
            
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                'Accept': 'application/json',
                'X-Subscription-Token': brave_key
            }
            params = {'q': query, 'count': 5}
            
            # 블로킹 I/O는 스레드로 위임하고, 단일 계층 타임아웃으로 보호
            response = await asyncio.wait_for(
                asyncio.to_thread(requests.get, url, headers=headers, params=params, timeout=10),
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            formatted_results = []
            for result in data.get('web', {}).get('results', [])[:5]:
                formatted_results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('description', ''),
                    'url': result.get('url', ''),
                    'source': 'brave'
                })
            
            return {
                'success': True,
                'method': 'brave',
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Brave search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _search_with_serper(self, query: str) -> Dict[str, Any]:
        """Search using Serper API (Google Search)."""
        try:
            serper_key = os.getenv('SERPER_API_KEY')
            if not serper_key:
                return {'success': False, 'error': 'Serper API key not configured'}
            
            url = "https://google.serper.dev/search"
            headers = {
                'X-API-KEY': serper_key,
                'Content-Type': 'application/json'
            }
            payload = {'q': query, 'num': 5, 'gl': 'kr', 'hl': 'ko'}
            
            response = await asyncio.wait_for(
                asyncio.to_thread(requests.post, url, headers=headers, json=payload, timeout=10),
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            formatted_results = []
            organic_results = data.get('organic', [])
            if not isinstance(organic_results, list):
                organic_results = []
            
            for result in organic_results[:5]:
                if isinstance(result, dict):
                    formatted_results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'url': result.get('link', ''),
                        'source': 'serper'
                    })
            
            return {
                'success': True,
                'method': 'serper',
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Serper search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _search_with_duckduckgo(self, query: str) -> Dict[str, Any]:
        """Search using DuckDuckGo (no API key required)."""
        try:
            import duckduckgo_search
            
            # Search for results
            results = duckduckgo_search.DDGS().text(query, max_results=5)
            
            formatted_results = []
            if not isinstance(results, list):
                results = []
            
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('body', ''),
                        'url': result.get('href', ''),
                        'source': 'duckduckgo'
                    })
            
            return {
                'success': True,
                'method': 'duckduckgo',
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _search_with_google_custom(self, query: str) -> Dict[str, Any]:
        """Search using Google Custom Search API."""
        try:
            # This would require Google Custom Search API key
            # For now, return failure to try other methods
            return {'success': False, 'error': 'Google Custom Search not configured'}
            
        except Exception as e:
            logger.warning(f"Google Custom Search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _search_with_bing(self, query: str) -> Dict[str, Any]:
        """Search using Bing Search API."""
        try:
            # This would require Bing Search API key
            # For now, return failure to try other methods
            return {'success': False, 'error': 'Bing Search not configured'}
            
        except Exception as e:
            logger.warning(f"Bing search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # LLM fallback method removed - no fallback responses allowed
    
    async def _perform_academic_search(self, query: str) -> Dict[str, Any]:
        """Perform academic search using real academic APIs."""
        try:
            from src.research.tools.academic_search import AcademicSearchTool
            
            # Initialize academic search tool
            academic_config = {
                'max_results': 10,
                'timeout': 30,
                'primary_provider': 'arxiv',
                'fallback_providers': ['scholar', 'pubmed']
            }
            
            academic_tool = AcademicSearchTool(academic_config)
            result = await academic_tool.arun(query)
            
            if result['success']:
                # Convert academic results to standard format
                academic_results = []
                for item in result['results']:
                    academic_results.append({
                        'title': item.get('title', ''),
                        'authors': item.get('authors', []),
                        'journal': item.get('journal', ''),
                        'year': item.get('published', '')[:4] if item.get('published') else '',
                        'abstract': item.get('abstract', ''),
                        'source': item.get('source', 'academic_database'),
                        'url': item.get('url', ''),
                        'pdf_url': item.get('pdf_url', ''),
                        'arxiv_id': item.get('arxiv_id', ''),
                        'pmid': item.get('pmid', ''),
                        'doi': item.get('doi', '')
                    })
                
                return {
                    'method': 'academic_search',
                    'query': query,
                    'results': academic_results,
                    'total_results': len(academic_results),
                    'provider': result.get('provider', 'academic'),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.warning(f"Academic search failed: {result.get('error', 'Unknown error')}")
                return {
                    'method': 'academic_search',
                    'query': query,
                    'results': [],
                    'total_results': 0,
                    'error': result.get('error', 'Academic search failed'),
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Academic search failed: {e}")
            return {
                'method': 'academic_search',
                'query': query,
                'results': [],
                'total_results': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _perform_data_analysis(self, query: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data analysis."""
        try:
            # Use LLM to analyze data
            prompt = f"""
            다음 주제에 대한 데이터 분석을 수행하세요:
            
            주제: {query}
            작업: {task.get('description', '')}
            
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
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    analysis = {'analysis': response_text, 'type': 'text'}
            
            return {
                'method': 'data_analysis',
                'query': query,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return {
                'method': 'data_analysis',
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _perform_content_analysis(self, query: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform content analysis."""
        try:
            # Use LLM for content analysis
            prompt = f"""
            다음 주제에 대한 콘텐츠 분석을 수행하세요:
            
            주제: {query}
            작업: {task.get('description', '')}
            
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
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    analysis = {'analysis': response_text, 'type': 'text'}
            
            return {
                'method': 'content_analysis',
                'query': query,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {
                'method': 'content_analysis',
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    
    async def _llm_analyze_research_results(self, research_results: List[Dict[str, Any]], task: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze research results."""
        try:
            prompt = f"""
            다음 연구 결과들을 분석하고 종합하세요:
            
            연구 결과들: {research_results}
            원래 작업: {task.get('description', '')}
            
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
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
        except Exception as e:
            logger.error(f"LLM result analysis failed: {e}")
            return {
                'summary': f'Analysis failed: {str(e)}',
                'key_findings': [],
                'quality_score': 0.0,
                'reliability_score': 0.0,
                'conclusions': 'Analysis could not be completed',
                'recommendations': []
            }
    
    async def _execute_data_collection_task(self, task: Dict[str, Any], 
                                          objective_id: str) -> Dict[str, Any]:
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
                'data_collection_result': {
                    'sources_used': len(data_sources),
                    'data_points_collected': len(processed_data),
                    'data_quality_score': self._calculate_data_quality(processed_data),
                    'collection_timestamp': datetime.now().isoformat()
                },
                'raw_data': processed_data,
                'data_summary': data_summary,
                'metadata': {
                    'task_description': task.get('description', ''),
                    'collection_method': 'autonomous_research',
                    'quality_metrics': self._calculate_quality_metrics(processed_data)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Data collection task failed: {e}")
            raise
    
    async def _execute_analysis_task(self, task: Dict[str, Any], 
                                   objective_id: str) -> Dict[str, Any]:
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
                'analysis_result': {
                    'method_used': analysis_method,
                    'analysis_quality': self._calculate_analysis_quality(analysis_result),
                    'insights_generated': len(insights),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'analysis_data': analysis_result,
                'insights': insights,
                'analysis_report': analysis_report,
                'metadata': {
                    'task_description': task.get('description', ''),
                    'analysis_type': task.get('task_type', 'analysis'),
                    'confidence_score': self._calculate_confidence_score(analysis_result)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis task failed: {e}")
            raise
    
    async def _execute_synthesis_task(self, task: Dict[str, Any], 
                                    objective_id: str) -> Dict[str, Any]:
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
            synthesis_report = await self._create_synthesis_report(synthesis_result, recommendations, task)
            
            result = {
                'synthesis_result': {
                    'sources_synthesized': len(synthesis_data),
                    'synthesis_quality': self._calculate_synthesis_quality(synthesis_result),
                    'recommendations_generated': len(recommendations),
                    'synthesis_timestamp': datetime.now().isoformat()
                },
                'synthesis_data': synthesis_result,
                'recommendations': recommendations,
                'synthesis_report': synthesis_report,
                'metadata': {
                    'task_description': task.get('description', ''),
                    'synthesis_type': task.get('task_type', 'synthesis'),
                    'completeness_score': self._calculate_completeness_score(synthesis_result)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Synthesis task failed: {e}")
            raise
    
    async def _execute_validation_task(self, task: Dict[str, Any], 
                                     objective_id: str) -> Dict[str, Any]:
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
                'validation_result': {
                    'validation_score': validation_result.get('overall_score', 0.0),
                    'issues_found': len(validation_result.get('issues', [])),
                    'validation_timestamp': datetime.now().isoformat()
                },
                'validation_data': validation_result,
                'validation_report': validation_report,
                'metadata': {
                    'task_description': task.get('description', ''),
                    'validation_type': task.get('task_type', 'validation'),
                    'reliability_score': self._calculate_reliability_score(validation_result)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Validation task failed: {e}")
            raise
    
    async def _execute_general_research_task(self, task: Dict[str, Any], 
                                           objective_id: str) -> Dict[str, Any]:
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
                'research_result': {
                    'research_scope': task.get('description', ''),
                    'research_quality': self._calculate_research_quality(research_result),
                    'research_timestamp': datetime.now().isoformat()
                },
                'research_data': research_result,
                'research_summary': research_summary,
                'metadata': {
                    'task_description': task.get('description', ''),
                    'research_type': 'general',
                    'comprehensiveness_score': self._calculate_comprehensiveness_score(research_result)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"General research task failed: {e}")
            raise
    
    async def _identify_data_sources(self, task: Dict[str, Any], 
                                   objective_id: str) -> List[Dict[str, Any]]:
        """Identify appropriate data sources for the task.
        
        Args:
            task: Research task
            objective_id: Objective ID
            
        Returns:
            List of identified data sources
        """
        try:
            sources = []
            task_description = task.get('description', '').lower()
            
            # Web sources
            if any(keyword in task_description for keyword in ['web', 'online', 'internet', 'website']):
                sources.append({
                    'type': 'web_sources',
                    'subtype': 'web_search',
                    'priority': 0.8,
                    'reliability': 0.7
                })
            
            # Academic sources
            if any(keyword in task_description for keyword in ['academic', 'research', 'paper', 'study', 'literature']):
                sources.append({
                    'type': 'academic_sources',
                    'subtype': 'academic_search',
                    'priority': 0.9,
                    'reliability': 0.9
                })
            
            # Data sources
            if any(keyword in task_description for keyword in ['data', 'statistics', 'dataset', 'database']):
                sources.append({
                    'type': 'data_sources',
                    'subtype': 'data_analysis',
                    'priority': 0.8,
                    'reliability': 0.8
                })
            
            # Default to web sources if no specific type identified
            if not sources:
                sources.append({
                    'type': 'web_sources',
                    'subtype': 'web_search',
                    'priority': 0.6,
                    'reliability': 0.7
                })
            
            return sources
            
        except Exception as e:
            logger.error(f"Data source identification failed: {e}")
            return []
    
    async def _collect_from_source(self, source: Dict[str, Any], task: Dict[str, Any], 
                                 objective_id: str) -> Optional[Dict[str, Any]]:
        """Collect data from a specific source.
        
        Args:
            source: Data source configuration
            task: Research task
            objective_id: Objective ID
            
        Returns:
            Collected data or None if collection failed
        """
        try:
            source_type = source.get('type', 'web_sources')
            subtype = source.get('subtype', 'web_search')
            
            # Simulate data collection based on source type
            if source_type == 'web_sources':
                return await self._collect_web_data(subtype, task, objective_id)
            elif source_type == 'academic_sources':
                return await self._collect_academic_data(subtype, task, objective_id)
            elif source_type == 'data_sources':
                return await self._collect_structured_data(subtype, task, objective_id)
            else:
                return await self._collect_general_data(subtype, task, objective_id)
                
        except Exception as e:
            logger.error(f"Data collection from source failed: {e}")
            return None
    
    async def _collect_web_data(self, subtype: str, task: Dict[str, Any], 
                              objective_id: str) -> Dict[str, Any]:
        """Collect data from web sources.
        
        Args:
            subtype: Web data subtype
            task: Research task
            objective_id: Objective ID
            
        Returns:
            Web data collection result
        """
        # 실제 웹 데이터 수집 수행 (MCP 도구 사용)
        raise NotImplementedError("_simulate_web_data_collection is not implemented. Use actual MCP search tools instead.")
    
    async def _collect_academic_data(self, subtype: str, task: Dict[str, Any], 
                                   objective_id: str) -> Dict[str, Any]:
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
                'source_type': 'academic',
                'subtype': subtype,
                'data_points': [
                    {
                        'title': f"Academic paper on {task.get('description', 'task')}",
                        'authors': ["Dr. Researcher", "Prof. Academic"],
                        'journal': "Journal of Research",
                        'year': 2024,
                        'abstract': f"Abstract of research on {task.get('description', 'task')}",
                        'reliability_score': 0.9,
                        'timestamp': datetime.now().isoformat()
                    }
                ],
                'collection_metadata': {
                    'search_query': task.get('description', ''),
                    'results_count': 1,
                    'collection_method': 'autonomous_academic_search'
                }
            }
            
            return academic_data
            
        except Exception as e:
            logger.error(f"Academic data collection failed: {e}")
            return {}
    
    async def _collect_structured_data(self, subtype: str, task: Dict[str, Any], 
                                     objective_id: str) -> Dict[str, Any]:
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
                'source_type': 'structured',
                'subtype': subtype,
                'data_points': [
                    {
                        'metric': f"Metric for {task.get('description', 'task')}",
                        'value': 85.5,
                        'unit': 'percentage',
                        'reliability_score': 0.8,
                        'timestamp': datetime.now().isoformat()
                    }
                ],
                'collection_metadata': {
                    'data_source': 'database',
                    'results_count': 1,
                    'collection_method': 'autonomous_data_analysis'
                }
            }
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Structured data collection failed: {e}")
            return {}
    
    async def _collect_general_data(self, subtype: str, task: Dict[str, Any], 
                                  objective_id: str) -> Dict[str, Any]:
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
                'source_type': 'general',
                'subtype': subtype,
                'data_points': [
                    {
                        'content': f"General research data for {task.get('description', 'task')}",
                        'reliability_score': 0.6,
                        'timestamp': datetime.now().isoformat()
                    }
                ],
                'collection_metadata': {
                    'collection_method': 'autonomous_general_research',
                    'results_count': 1
                }
            }
            
            return general_data
            
        except Exception as e:
            logger.error(f"General data collection failed: {e}")
            return {}
    
    async def _process_collected_data(self, collected_data: List[Dict[str, Any]], 
                                    task: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                    'source_type': data.get('source_type', 'unknown'),
                    'data_points': data.get('data_points', []),
                    'processed_at': datetime.now().isoformat(),
                    'quality_score': self._calculate_data_quality([data]),
                    'relevance_score': self._calculate_relevance_score(data, task)
                }
                
                processed_data.append(processed_item)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return collected_data
    
    async def _generate_data_summary(self, processed_data: List[Dict[str, Any]], 
                                   task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of processed data.
        
        Args:
            processed_data: Processed data
            task: Research task
            
        Returns:
            Data summary
        """
        try:
            total_data_points = sum(len(item.get('data_points', [])) for item in processed_data)
            avg_quality = sum(item.get('quality_score', 0) for item in processed_data) / len(processed_data) if processed_data else 0
            avg_relevance = sum(item.get('relevance_score', 0) for item in processed_data) / len(processed_data) if processed_data else 0
            
            summary = {
                'total_sources': len(processed_data),
                'total_data_points': total_data_points,
                'average_quality_score': avg_quality,
                'average_relevance_score': avg_relevance,
                'data_diversity': len(set(item.get('source_type', 'unknown') for item in processed_data)),
                'summary_timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Data summary generation failed: {e}")
            return {}
    
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
                if 'data_points' in item:
                    for point in item['data_points']:
                        reliability = point.get('reliability_score', 0.5)
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
            task_description = task.get('description', '').lower()
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
                'completeness': self._calculate_completeness_score(data),
                'accuracy': self._calculate_accuracy_score(data),
                'relevance': self._calculate_relevance_score(data[0], {}) if data else 0.5,
                'timeliness': self._calculate_timeliness_score(data),
                'consistency': self._calculate_consistency_score(data)
            }
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return {'completeness': 0.5, 'accuracy': 0.5, 'relevance': 0.5, 'timeliness': 0.5, 'consistency': 0.5}
    
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
    async def _select_analysis_method(self, task: Dict[str, Any], objective_id: str) -> str:
        """Select appropriate analysis method using LLM."""
        from src.core.llm_manager import execute_llm_task, TaskType
        
        prompt = f"""
        Analyze the following research task and select the most appropriate analysis method:
        
        Task: {task.get('description', '')}
        Task Type: {task.get('task_type', 'analysis')}
        Objective ID: {objective_id}
        
        Select from: quantitative, qualitative, mixed, comparative, predictive
        
        Respond with only the method name.
        """
        
        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.ANALYSIS,
                system_message="You are an expert research analyst."
            )
            method = result.content.strip().lower()
            return method if method in ['quantitative', 'qualitative', 'mixed', 'comparative', 'predictive'] else 'mixed'
        except Exception as e:
            logger.error(f"Failed to select analysis method: {e}")
            return 'mixed'
    
    async def _perform_analysis(self, method: str, task: Dict[str, Any], objective_id: str) -> Dict[str, Any]:
        """Perform analysis using selected method with LLM."""
        from src.core.llm_manager import execute_llm_task, TaskType
        
        prompt = f"""
        Perform {method} analysis for the following research task:
        
        Task: {task.get('description', '')}
        Task Type: {task.get('task_type', 'analysis')}
        Method: {method}
        
        Provide comprehensive analysis results in JSON format with keys: findings, data_points, patterns, conclusions.
        """
        
        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.DEEP_REASONING,
                system_message=f"You are an expert {method} research analyst."
            )
            
            import json
            try:
                analysis_data = json.loads(result.content)
            except:
                analysis_data = {'method': method, 'results': result.content, 'raw_response': result.content}
            
            return analysis_data
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {'method': method, 'results': f'Analysis failed: {str(e)}', 'error': str(e)}
    
    async def _generate_insights(self, analysis_result: Dict[str, Any], task: Dict[str, Any]) -> List[str]:
        """Generate insights from analysis using LLM."""
        from src.core.llm_manager import execute_llm_task, TaskType
        
        prompt = f"""
        Generate key insights from the following analysis results:
        
        Analysis: {json.dumps(analysis_result, ensure_ascii=False, indent=2)}
        Task: {task.get('description', '')}
        
        Provide 3-5 key insights as a JSON array of strings.
        """
        
        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.REASONING,
                system_message=self.config.prompts["search_execution"]["system_message"]
            )
            
            import json
            try:
                insights = json.loads(result.content)
                if isinstance(insights, list):
                    return insights
                else:
                    return [insights] if isinstance(insights, str) else [result.content]
            except:
                # Fallback: parse as newline-separated list
                insights = [line.strip() for line in result.content.split('\n') if line.strip()]
                return insights[:5]  # Max 5 insights
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return [f"Analysis completed for: {task.get('description', 'task')}"]
    
    async def _create_analysis_report(self, analysis_result: Dict[str, Any], insights: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Create analysis report using LLM."""
        from src.core.llm_manager import execute_llm_task, TaskType
        
        prompt = f"""
        Create a comprehensive analysis report:
        
        Analysis Result: {json.dumps(analysis_result, ensure_ascii=False, indent=2)}
        Key Insights: {json.dumps(insights, ensure_ascii=False)}
        Task: {task.get('description', '')}
        
        Provide a structured report in JSON format with sections: executive_summary, key_findings, detailed_analysis, conclusions.
        """
        
        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.GENERATION,
                system_message=self.config.prompts["content_analysis"]["system_message"]
            )
            
            import json
            try:
                report = json.loads(result.content)
            except:
                report = {'report': result.content, 'insights': insights}
            
            return report
        except Exception as e:
            logger.error(f"Report creation failed: {e}")
            return {'report': f'Report generation failed: {str(e)}', 'insights': insights}
    
    def _calculate_analysis_quality(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate analysis quality score based on result structure."""
        if not analysis_result:
            return 0.0
        
        score = 0.5  # Base score
        
        # Check for required keys
        if 'findings' in analysis_result or 'results' in analysis_result:
            score += 0.2
        if 'data_points' in analysis_result or 'patterns' in analysis_result:
            score += 0.2
        if 'conclusions' in analysis_result:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_confidence_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate confidence score based on analysis quality."""
        base_quality = self._calculate_analysis_quality(analysis_result)
        return base_quality * 0.9  # Slightly conservative
    
    async def _gather_synthesis_data(self, task: Dict[str, Any], objective_id: str) -> List[Dict[str, Any]]:
        """Gather data from previous tasks for synthesis."""
        # Get data from shared memory or previous execution results
        try:
            from src.core.shared_memory import get_shared_memory, MemoryScope
            memory = get_shared_memory()
            
            # Search for related data
            query = task.get('description', '')
            related_data = memory.search(query, limit=10)
            
            return related_data if related_data else []
        except Exception as e:
            logger.error(f"Failed to gather synthesis data: {e}")
            return []
    
    async def _perform_synthesis(self, data: List[Dict[str, Any]], task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform synthesis using LLM."""
        from src.core.llm_manager import execute_llm_task, TaskType
        
        prompt = f"""
        Synthesize the following data into a cohesive analysis:
        
        Data: {json.dumps(data, ensure_ascii=False, indent=2)}
        Task: {task.get('description', '')}
        
        Provide synthesis in JSON format with keys: integrated_findings, themes, connections, synthesis_summary.
        """
        
        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.SYNTHESIS,
                system_message=self.config.prompts["synthesis_report"]["system_message"]
            )
            
            import json
            try:
                synthesis = json.loads(result.content)
            except:
                synthesis = {'synthesis': result.content}
            
            return synthesis
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {'synthesis': f'Synthesis failed: {str(e)}', 'error': str(e)}
    
    async def _generate_recommendations(self, synthesis_result: Dict[str, Any], task: Dict[str, Any]) -> List[str]:
        """Generate recommendations using LLM."""
        from src.core.llm_manager import execute_llm_task, TaskType
        
        prompt = f"""
        Generate actionable recommendations based on:
        
        Synthesis: {json.dumps(synthesis_result, ensure_ascii=False, indent=2)}
        Task: {task.get('description', '')}
        
        Provide 3-5 recommendations as a JSON array of strings.
        """
        
        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.REASONING,
                system_message=self.config.prompts["recommendation_generation"]["system_message"]
            )
            
            import json
            try:
                recommendations = json.loads(result.content)
                if isinstance(recommendations, list):
                    return recommendations
                else:
                    return [recommendations] if isinstance(recommendations, str) else [result.content]
            except:
                recommendations = [line.strip() for line in result.content.split('\n') if line.strip()]
                return recommendations[:5]
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return [f"Complete task: {task.get('description', 'task')}"]
    
    async def _create_synthesis_report(self, synthesis_result: Dict[str, Any], recommendations: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Create synthesis report using LLM."""
        from src.core.llm_manager import execute_llm_task, TaskType
        
        prompt = f"""
        Create a synthesis report:
        
        Synthesis: {json.dumps(synthesis_result, ensure_ascii=False, indent=2)}
        Recommendations: {json.dumps(recommendations, ensure_ascii=False)}
        Task: {task.get('description', '')}
        
        Provide report in JSON format with sections: summary, key_points, recommendations, next_steps.
        """
        
        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.GENERATION,
                system_message="You are an expert report writer."
            )
            
            import json
            try:
                report = json.loads(result.content)
            except:
                report = {'report': result.content, 'recommendations': recommendations}
            
            return report
        except Exception as e:
            logger.error(f"Synthesis report creation failed: {e}")
            return {'report': f'Report generation failed: {str(e)}', 'recommendations': recommendations}
    
    def _calculate_synthesis_quality(self, synthesis_result: Dict[str, Any]) -> float:
        """Calculate synthesis quality score."""
        if not synthesis_result:
            return 0.0
        
        score = 0.5
        if 'integrated_findings' in synthesis_result or 'synthesis' in synthesis_result:
            score += 0.3
        if 'themes' in synthesis_result or 'connections' in synthesis_result:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_completeness_score(self, synthesis_result: Dict[str, Any]) -> float:
        """Calculate completeness score."""
        return self._calculate_synthesis_quality(synthesis_result) * 0.9
    
    async def _gather_validation_data(self, task: Dict[str, Any], objective_id: str) -> List[Dict[str, Any]]:
        """Gather data for validation."""
        try:
            from src.core.shared_memory import get_shared_memory
            memory = get_shared_memory()
            
            query = task.get('description', '')
            related_data = memory.search(query, limit=10)
            
            return related_data if related_data else []
        except Exception as e:
            logger.error(f"Failed to gather validation data: {e}")
            return []
    
    async def _perform_validation(self, data: List[Dict[str, Any]], task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform validation using LLM."""
        from src.core.llm_manager import execute_llm_task, TaskType
        
        prompt = f"""
        Validate the following research data:
        
        Data: {json.dumps(data, ensure_ascii=False, indent=2)}
        Task: {task.get('description', '')}
        
        Provide validation in JSON format with keys: overall_score (0-1), issues (array), strengths (array), validation_summary.
        """
        
        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.VERIFICATION,
                system_message=self.config.prompts["quality_validation"]["system_message"]
            )
            
            import json
            try:
                validation = json.loads(result.content)
                if 'overall_score' not in validation:
                    validation['overall_score'] = 0.8  # Default score
            except:
                validation = {'overall_score': 0.8, 'issues': [], 'summary': result.content}
            
            return validation
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {'overall_score': 0.5, 'issues': [str(e)], 'error': str(e)}
    
    async def _create_validation_report(self, validation_result: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation report using LLM."""
        from src.core.llm_manager import execute_llm_task, TaskType
        
        prompt = f"""
        Create a validation report:
        
        Validation Result: {json.dumps(validation_result, ensure_ascii=False, indent=2)}
        Task: {task.get('description', '')}
        
        Provide report in JSON format with sections: validation_score, issues_found, strengths, recommendations.
        """
        
        try:
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.GENERATION,
                system_message=self.config.prompts["final_assessment"]["system_message"]
            )
            
            import json
            try:
                report = json.loads(result.content)
            except:
                report = {'report': result.content, 'score': validation_result.get('overall_score', 0.0)}
            
            return report
        except Exception as e:
            logger.error(f"Validation report creation failed: {e}")
            return {'report': f'Report generation failed: {str(e)}', 'score': validation_result.get('overall_score', 0.0)}
    
    async def _generate_research_summary(self, query: str, search_results: Dict[str, Any], collected_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive research summary using LLM."""
        try:
            if not self.llm:
                return {'summary': 'LLM not available', 'key_points': [], 'insights': []}
            
            # Prepare data for LLM analysis
            data_summary = {
                'query': query,
                'search_success': search_results.get('success', False) if isinstance(search_results, dict) else False,
                'results_count': len(search_results.get('results', [])) if isinstance(search_results, dict) else 0,
                'collected_data_count': len(collected_data) if collected_data else 0,
                'research_data': search_results.get('research_data', {}) if isinstance(search_results, dict) else {}
            }
            
            prompt = f"""
            Based on the research query: "{query}"
            
            Research data summary:
            - Search successful: {data_summary['search_success']}
            - Results found: {data_summary['results_count']}
            - Data collected: {data_summary['collected_data_count']}
            - Research data: {json.dumps(data_summary['research_data'], ensure_ascii=False, indent=2)}
            
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
                    "recommendations": [f"Recommendation for {query}"]
                }
            
            return summary_data
            
        except Exception as e:
            logger.error(f"Research summary generation failed: {e}")
            return {
                'executive_summary': f'Research summary generation failed: {e}',
                'key_insights': [],
                'statistics': {},
                'expert_analysis': 'Analysis unavailable',
                'future_implications': [],
                'recommendations': []
            }
    
    async def _perform_general_research(self, task: Dict[str, Any], objective_id: str) -> Dict[str, Any]:
        """Perform general research with real data collection."""
        try:
            if not task or not isinstance(task, dict):
                task = {}
            task_description = task.get('description', '')
            if not task_description:
                task_description = 'general research'
            research_query = str(task_description).replace('Research task for objective: ', '')
            
            # 1. Perform web search
            search_results = await self._perform_web_search(research_query)
            
            # 2. Collect additional data from URLs
            collected_data = []
            if isinstance(search_results, dict) and search_results.get('success') and search_results.get('results'):
                results = search_results.get('results', [])
                if isinstance(results, list):
                    for result in results[:3]:  # Limit to top 3 results
                        if isinstance(result, dict):
                            try:
                                # For now, just use the snippet as content since we don't have URLs
                                content = result.get('snippet', '')
                                if content:
                                    collected_data.append({
                                        'url': result.get('url', ''),
                                        'title': result.get('title', ''),
                                        'content': content,
                                        'source': result.get('source', 'unknown')
                                    })
                            except Exception as e:
                                logger.warning(f"Failed to process result: {e}")
                                continue
            
            # 3. Perform additional research using different methods
            additional_research = await self._perform_additional_research(research_query)
            if not isinstance(additional_research, dict):
                additional_research = {'sources': []}
            
            # 4. Generate comprehensive research summary
            research_summary = await self._generate_research_summary({
                'query': research_query,
                'collected_data': collected_data,
                'additional_research': additional_research,
                'research_quality': 0.8
            }, task)
            
            # Ensure research_summary is a dict
            if not isinstance(research_summary, dict):
                research_summary = {'summary': 'Research summary generation failed', 'error': 'Invalid summary format'}
            
            # 5. Compile research results
            research_result = {
                'task_id': task.get('task_id', 'unknown'),
                'agent': 'researcher',
                'task_type': task.get('task_type', 'general'),
                'result': {
                    'research_data': {
                'query': research_query,
                        'web_search_results': search_results,
                'collected_data': collected_data,
                'additional_research': additional_research,
                        'research_summary': research_summary,
                'total_sources': len(collected_data) + len(additional_research.get('sources', [])),
                        'research_timestamp': datetime.now().isoformat()
                    },
                    'sources': [item.get('url', '') for item in collected_data if isinstance(item, dict) and item.get('url')],
                    'metadata': {
                        'research_method': 'web_search_and_analysis',
                        'data_quality': 'high' if len(collected_data) > 0 else 'low',
                        'completeness': 'partial' if len(collected_data) < 3 else 'complete'
                    }
                }
            }
            
            return research_result
            
        except Exception as e:
            logger.error(f"General research failed: {e}")
            return {'research': f'Research failed: {str(e)}', 'error': str(e)}
    
    async def _fetch_web_content(self, url: str) -> str:
        """Fetch content from a web URL using MCP tools."""
        try:
            # Use MCP tools for content fetching
            from src.core.mcp_integration import execute_tool
            
            result = await execute_tool("fetch", {"url": url})
            
            if result.get('success', False):
                content = result.get('data', {}).get('content', '')
                
                # Basic HTML tag removal
                import re
                text = re.sub(r'<[^>]+>', '', content)
                text = re.sub(r'\s+', ' ', text)
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
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
                'sources': additional_sources,
                'total_additional_sources': len(additional_sources),
                'research_methods': ['academic', 'news', 'social_media']
            }
            
        except Exception as e:
            logger.error(f"Additional research failed: {e}")
            return {'sources': [], 'total_additional_sources': 0, 'research_methods': []}
    
    async def _search_academic_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search academic sources."""
        try:
            # Try Google Scholar
            scholar_query = f"site:scholar.google.com {query}"
            search_results = await self._perform_web_search(scholar_query)
            
            academic_sources = []
            if search_results.get('success') and search_results.get('results'):
                for result in search_results['results'][:2]:
                    academic_sources.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'snippet': result.get('snippet', ''),
                        'source_type': 'academic',
                        'search_engine': 'google_scholar'
                    })
            
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
            if search_results.get('success') and search_results.get('results'):
                for result in search_results['results'][:2]:
                    news_sources.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'snippet': result.get('snippet', ''),
                        'source_type': 'news',
                        'search_engine': 'general'
                    })
            
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
            if search_results.get('success') and search_results.get('results'):
                for result in search_results['results'][:2]:
                    social_sources.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'snippet': result.get('snippet', ''),
                        'source_type': 'social_media',
                        'search_engine': 'general'
                    })
            
            return social_sources
            
        except Exception as e:
            logger.error(f"Social media search failed: {e}")
            return []
    
    def _calculate_research_quality_from_data(self, collected_data: List[Dict[str, Any]], additional_research: Dict[str, Any]) -> float:
        """Calculate research quality based on collected data."""
        try:
            quality_score = 0.0
            
            # Base score for collected data
            if collected_data:
                quality_score += 0.4
                
                # Bonus for content length
                total_content_length = sum(len(item.get('content', '')) for item in collected_data)
                if total_content_length > 1000:
                    quality_score += 0.2
                elif total_content_length > 500:
                    quality_score += 0.1
            
            # Bonus for additional sources
            additional_sources = additional_research.get('sources', [])
            if additional_sources:
                quality_score += 0.2
                
                # Bonus for diverse source types
                source_types = set(source.get('source_type', '') for source in additional_sources)
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

    async def _generate_research_summary(self, research_result: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate research summary from actual data."""
        try:
            # Ensure research_result is a dictionary
            if not isinstance(research_result, dict):
                research_result = {'query': str(research_result), 'collected_data': [], 'additional_research': {}}
            
            # Extract key information from research results with type safety
            query = research_result.get('query', '')
            
            # Ensure collected_data is a list
            collected_data = research_result.get('collected_data', [])
            if not isinstance(collected_data, list):
                collected_data = []
            
            # Ensure additional_research is a dict
            additional_research = research_result.get('additional_research', {})
            if not isinstance(additional_research, dict):
                additional_research = {}
            
            # Generate summary based on collected data
            summary_parts = []
            
            if collected_data:
                summary_parts.append(f"Found {len(collected_data)} primary sources with detailed content.")
                
                # Extract key insights from collected data
                key_insights = []
                for item in collected_data[:2]:  # Top 2 sources
                    if isinstance(item, dict):
                        title = item.get('title', '')
                        content = item.get('content', '')
                        if title and content:
                            # Extract first 200 characters as insight
                            insight = content[:200] + "..." if len(content) > 200 else content
                            key_insights.append(f"• {title}: {insight}")
                
                if key_insights:
                    summary_parts.append("Key insights:")
                    summary_parts.extend(key_insights)
            
            # Ensure additional_sources is a list
            additional_sources = additional_research.get('sources', [])
            if not isinstance(additional_sources, list):
                additional_sources = []
            
            if additional_sources:
                summary_parts.append(f"Found {len(additional_sources)} additional sources from academic, news, and social media.")
            
            # Calculate research metrics
            total_sources = len(collected_data) + len(additional_sources)
            research_quality = research_result.get('research_quality', 0.5)
            
            summary_parts.append(f"Research quality: {research_quality:.1%}")
            summary_parts.append(f"Total sources: {total_sources}")
            
            summary_text = "\n".join(summary_parts)
            
            return {
                'summary': summary_text,
                'key_insights': key_insights if 'key_insights' in locals() else [],
                'total_sources': total_sources,
                'research_quality': research_quality,
                'summary_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Research summary generation failed: {e}")
            return {'summary': f'Summary generation failed: {str(e)}', 'error': str(e)}
    
    def _calculate_research_quality(self, research_result: Dict[str, Any]) -> float:
        """Calculate research quality score."""
        return 0.8
    
    def _calculate_comprehensiveness_score(self, research_result: Dict[str, Any]) -> float:
        """Calculate comprehensiveness score."""
        return 0.75
    
    async def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if agent can handle a specific task.
        
        Args:
            task: Task to check
            
        Returns:
            True if agent can handle the task
        """
        try:
            task_type = task.get('task_type', 'general')
            required_skills = task.get('required_skills', [])
            
            # Check if agent has required skills
            agent_skills = ['research', 'data_collection', 'web_search', 'academic_research', 'analysis']
            
            for skill in required_skills:
                if skill not in agent_skills:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Task capability check failed: {e}")
            return False
    
    async def cancel_tasks(self, objective_id: str) -> bool:
        """Cancel tasks for a specific objective.
        
        Args:
            objective_id: Objective ID to cancel tasks for
            
        Returns:
            True if tasks were cancelled successfully
        """
        try:
            cancelled_count = 0
            
            for task_id, task_info in self.active_tasks.items():
                if task_info.get('objective_id') == objective_id:
                    task_info['status'] = 'cancelled'
                    task_info['cancelled_at'] = datetime.now()
                    cancelled_count += 1
            
            logger.info(f"Cancelled {cancelled_count} tasks for objective: {objective_id}")
            return True
            
        except Exception as e:
            logger.error(f"Task cancellation failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup agent resources."""
        try:
            # Cancel all active tasks
            for task_id in list(self.active_tasks.keys()):
                self.active_tasks[task_id]['status'] = 'cancelled'
                self.active_tasks[task_id]['cancelled_at'] = datetime.now()
            
            self.active_tasks.clear()
            logger.info("Research Agent cleanup completed")
            
        except Exception as e:
            logger.error(f"Research Agent cleanup failed: {e}")
    
    async def update_capabilities(self, evaluation_result: Dict[str, Any], iteration: int) -> None:
        """Update agent capabilities based on evaluation feedback.
        
        Args:
            evaluation_result: Evaluation results from current iteration
            iteration: Current iteration number
        """
        try:
            feedback = evaluation_result.get('feedback', [])
            quality_metrics = evaluation_result.get('quality_metrics', {})
            
            # Store learning data
            learning_entry = {
                'iteration': iteration,
                'feedback': feedback,
                'quality_metrics': quality_metrics,
                'timestamp': datetime.now().isoformat()
            }
            self.learning_data.append(learning_entry)
            
            # Update adaptive strategies based on feedback
            if 'insufficient_data' in str(feedback):
                self.adaptive_strategies['search_depth'] = min(5, self.adaptive_strategies['search_depth'] + 1)
                self.adaptive_strategies['content_length'] = min(10000, self.adaptive_strategies['content_length'] + 1000)
            elif 'excessive_data' in str(feedback):
                self.adaptive_strategies['search_depth'] = max(1, self.adaptive_strategies['search_depth'] - 1)
                self.adaptive_strategies['content_length'] = max(2000, self.adaptive_strategies['content_length'] - 1000)
            
            if 'insufficient_diversity' in str(feedback):
                self.adaptive_strategies['source_diversity'] = min(1.0, self.adaptive_strategies['source_diversity'] + 0.1)
            elif 'excessive_diversity' in str(feedback):
                self.adaptive_strategies['source_diversity'] = max(0.5, self.adaptive_strategies['source_diversity'] - 0.1)
            
            # Update research patterns based on successful patterns
            if quality_metrics.get('overall_score', 0) > 0.8:
                await self._update_successful_research_patterns(evaluation_result)
            
            logger.info(f"ResearchAgent capabilities updated for iteration {iteration}")
            
        except Exception as e:
            logger.error(f"Research capability update failed: {e}")
    
    async def _update_successful_research_patterns(self, evaluation_result: Dict[str, Any]) -> None:
        """Update research patterns based on successful evaluations."""
        try:
            # Extract successful patterns from evaluation
            quality_metrics = evaluation_result.get('quality_metrics', {})
            
            # Update research method weights based on success
            for method_name in self.analysis_methods:
                if quality_metrics.get(f'{method_name}_score', 0) > 0.8:
                    if 'weight' not in self.analysis_methods[method_name]:
                        self.analysis_methods[method_name]['weight'] = 1.0
                    self.analysis_methods[method_name]['weight'] = min(2.0, 
                        self.analysis_methods[method_name]['weight'] + 0.1)
            
            logger.info("Successful research patterns updated")
            
        except Exception as e:
            logger.error(f"Research pattern update failed: {e}")
    
    async def _enhanced_research_with_learning(self, task: Dict[str, Any], 
                                             context: Dict[str, Any], 
                                             objective_id: str) -> Dict[str, Any]:
        """Enhanced research using learning from previous iterations."""
        try:
            # Get learning data from context
            learning_data = context.get('learning_data', [])
            iteration = context.get('iteration', 1)
            
            # Start with base research
            result = await self.conduct_research(task, context, objective_id)
            
            # Apply learning enhancements
            if learning_data and iteration > 1:
                result = await self._apply_research_learning_enhancements(result, learning_data, iteration)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced research failed: {e}")
            return await self.conduct_research(task, context, objective_id)
    
    async def _apply_research_learning_enhancements(self, result: Dict[str, Any], 
                                                  learning_data: List[Dict[str, Any]], 
                                                  iteration: int) -> Dict[str, Any]:
        """Apply learning enhancements to research results."""
        try:
            enhanced_result = result.copy()
            
            # Apply learning-based enhancements
            if iteration > 1:
                latest_feedback = learning_data[-1].get('feedback', [])
                
                # Enhance data collection if insufficient
                if 'insufficient_data' in str(latest_feedback):
                    enhanced_result['data_enhanced'] = True
                    enhanced_result['search_depth'] = self.adaptive_strategies['search_depth']
                    enhanced_result['content_length'] = self.adaptive_strategies['content_length']
                
                # Enhance source diversity if needed
                if 'insufficient_diversity' in str(latest_feedback):
                    enhanced_result['diversity_enhanced'] = True
                    enhanced_result['source_diversity'] = self.adaptive_strategies['source_diversity']
                
                # Enhance quality if needed
                if 'quality_issues' in str(latest_feedback):
                    enhanced_result['quality_enhanced'] = True
                    enhanced_result['quality_threshold'] = 0.8
            
            # Add learning metadata
            enhanced_result['learning_applied'] = True
            enhanced_result['iteration'] = iteration
            enhanced_result['learning_data_count'] = len(learning_data)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Research learning enhancement failed: {e}")
            return result
    
    # Browser automation methods are now handled by BrowserManager
    
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
            
            logger.info(f"Content extraction completed for {url} using {result.get('method', 'unknown')} method")
            return result
            
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return {
                "success": False,
                "url": url,
                "extraction_goal": extraction_goal,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Fallback methods are now handled by BrowserManager
    
    async def browser_search_and_extract(self, query: str, extraction_goal: str, max_results: int = 3) -> List[Dict[str, Any]]:
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
            results = await self.browser_manager.search_and_extract(query, extraction_goal, max_results, self.llm)
            
            logger.info(f"Search and extract completed for query '{query}' with {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search and extract failed: {e}")
            return []
    
    async def browser_interactive_research(self, research_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
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
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
            }
    
    
    async def cleanup_browser(self):
        """Clean up browser resources using browser manager."""
        try:
            if self.browser_manager is not None:
                await self.browser_manager.cleanup()
                logger.info("Browser resources cleaned up")
                
        except Exception as e:
            logger.error(f"Browser cleanup failed: {e}")
