"""
Advanced Academic Search Tool with MCP Integration and Production-Grade Reliability
Implements 8 Core Innovations: Universal MCP Hub, Production-Grade Reliability, Multi-Model Orchestration
"""

import logging
import asyncio
import os
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

# Import core modules for 8 innovations
from .base_tool import BaseResearchTool, ToolResult, SearchResult
from ...core.reliability import execute_with_reliability, CircuitBreaker
from ...core.llm_manager import execute_llm_task, TaskType
from src.core.mcp_integration import get_best_tool_for_task, execute_tool


@dataclass
class ToolResponse:
    """Tool execution response."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ToolCategory(Enum):
    """Tool categories."""
    SEARCH = "search"
    DATA = "data"
    CODE = "code"
    ACADEMIC = "academic"
    BUSINESS = "business"
    UTILITY = "utility"

logger = logging.getLogger(__name__)


class AcademicProvider(Enum):
    """Available academic search providers with MCP priority."""
    ARXIV_MCP = "arxiv_mcp"
    SCHOLAR_MCP = "scholar_mcp"
    PUBMED_MCP = "pubmed_mcp"
    IEEE_MCP = "ieee_mcp"


@dataclass
class AcademicResult:
    """Standardized academic search result."""
    title: str
    authors: List[str]
    abstract: str
    published: str
    source: str
    url: str
    pdf_url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    pmid: str = ""
    categories: List[str] = None
    confidence_score: float = 1.0
    verification_status: str = "unverified"
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []


class AcademicSearchTool(BaseResearchTool):
    """Advanced academic search tool with MCP integration and production-grade reliability."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize academic search tool with 8 innovations."""
        super().__init__(config)
        self.max_results = config.get('max_results', 10)
        self.timeout = config.get('timeout', 30)
        self.enable_mcp_priority = config.get('enable_mcp_priority', True)
        self.enable_verification = config.get('enable_verification', True)
        self.enable_confidence_scoring = config.get('enable_confidence_scoring', True)
        
        # MCP-only provider priority
        self.primary_provider = config.get('primary_provider', AcademicProvider.ARXIV_MCP)
        self.alternative_providers = config.get('alternative_providers', [
            AcademicProvider.SCHOLAR_MCP,
            AcademicProvider.PUBMED_MCP,
            AcademicProvider.IEEE_MCP
        ])
    
        # Circuit breaker for reliability
        from ...core.reliability import CircuitBreakerConfig
        self.circuit_breaker = CircuitBreaker(
            name="academic_search",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60
            )
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self, query: str, **kwargs) -> ToolResponse:
        """Run academic search synchronously with production-grade reliability."""
        try:
            return asyncio.run(self.arun(query, **kwargs))
        except Exception as e:
            self.logger.error(f"Academic search failed: {e}")
            return ToolResponse(
                success=False,
                data=[],
                error=str(e),
                metadata={'query': query, 'timestamp': datetime.now().isoformat()}
            )
    
    async def arun(self, query: str, **kwargs) -> ToolResponse:
        """Run academic search with MCP priority and production-grade reliability."""
        return await execute_with_reliability(
            self._execute_academic_search,
            query=query,
            **kwargs
        )
    
    async def _execute_academic_search(self, query: str, **kwargs) -> ToolResponse:
        """Execute academic search with MCP-first approach and continuous verification."""
        try:
            # Step 1: Try MCP tools first (Universal MCP Hub - Innovation 6)
            if self.enable_mcp_priority:
                mcp_result = await self._try_mcp_search(query)
                if mcp_result.success and mcp_result.data:
                    # Apply continuous verification (Innovation 4)
                    if self.enable_verification:
                        verified_result = await self._verify_results(mcp_result.data, query)
                        return verified_result
                    return mcp_result
            
            # Step 2: Try alternative MCP providers
            api_result = await self._try_alternative_mcp_search(query)
            if api_result.success and api_result.data:
                # Apply continuous verification
                if self.enable_verification:
                    verified_result = await self._verify_results(api_result.data, query)
                    return verified_result
                return api_result
            
            # Step 3: All methods failed
            self.logger.warning("All academic search methods failed")
            return ToolResponse(
                success=False,
                data=[],
                error="All academic search providers failed",
                metadata={'query': query, 'timestamp': datetime.now().isoformat()}
            )
                
        except Exception as e:
            self.logger.error(f"Academic search execution failed: {e}")
            return ToolResponse(
                success=False,
                data=[],
                error=str(e),
                metadata={'query': query, 'timestamp': datetime.now().isoformat()}
            )
    
    async def _try_mcp_search(self, query: str) -> ToolResponse:
        """Try MCP tools first (Universal MCP Hub - Innovation 6)."""
        try:
            # Get best MCP tool for academic search
            best_tool = get_best_tool_for_task("academic_search")
            if not best_tool:
                return ToolResponse(success=False, data=[], error="No MCP academic tools available")
            
            # Execute MCP tool
            mcp_result = await execute_tool(best_tool, {"query": query, "max_results": self.max_results})
            
            if mcp_result.get('success', False):
                # Convert MCP result to AcademicResult format
                academic_results = self._convert_to_academic_results(mcp_result.get('data', []), best_tool)
                return ToolResponse(
                    success=True,
                    data=academic_results,
                    metadata={
                        'provider': best_tool,
                        'method': 'mcp',
                        'query': query,
                        'timestamp': datetime.now().isoformat()
                    }
                )
            else:
                return ToolResponse(success=False, data=[], error=mcp_result.get('error', 'MCP tool failed'))
                
        except Exception as e:
            self.logger.warning(f"MCP academic search failed: {e}")
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _try_alternative_mcp_search(self, query: str) -> ToolResponse:
        """Try alternative MCP providers."""
        all_results = []
        
        # Try primary provider first
        primary_result = await self._search_with_provider(self.primary_provider, query)
        if primary_result.success:
            all_results.extend(primary_result.data)
        
        # Try alternative MCP providers if needed
        if len(all_results) < self.max_results:
            for provider in self.alternative_providers:
                if len(all_results) >= self.max_results:
                    break
                    
                try:
                    result = await self._search_with_provider(provider, query)
                    if result.success:
                        all_results.extend(result.data)
                except Exception as e:
                    self.logger.warning(f"Provider {provider.value} failed: {e}")
                    continue
            
        if all_results:
            return ToolResponse(
                success=True,
                data=all_results[:self.max_results],
                metadata={
                    'provider': 'multi_mcp',
                    'method': 'mcp',
                    'query': query,
                    'total_found': len(all_results),
                    'timestamp': datetime.now().isoformat()
                }
            )
        else:
            return ToolResponse(success=False, data=[], error="All MCP providers failed")
    
    async def _search_with_provider(self, provider: AcademicProvider, query: str) -> ToolResponse:
        """Search with specific MCP provider using circuit breaker pattern."""
        try:
            if provider == AcademicProvider.ARXIV_MCP:
                return await self._arxiv_mcp_search(query)
            elif provider == AcademicProvider.SCHOLAR_MCP:
                return await self._scholar_mcp_search(query)
            elif provider == AcademicProvider.PUBMED_MCP:
                return await self._pubmed_mcp_search(query)
            elif provider == AcademicProvider.IEEE_MCP:
                return await self._ieee_mcp_search(query)
            else:
                return ToolResponse(success=False, data=[], error=f"Unknown provider: {provider}")
        except Exception as e:
            self.logger.error(f"Provider {provider.value} search failed: {e}")
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _verify_results(self, results: List[AcademicResult], query: str) -> ToolResponse:
        """Apply continuous verification to search results (Innovation 4)."""
        try:
            if not self.enable_verification or not results:
                return ToolResponse(success=True, data=results)
            
            # Use Multi-Model Orchestration for verification (Innovation 3)
            verification_prompt = f"""
            Verify the following academic search results for query: "{query}"
            
            Results to verify:
            {[f"- {r.title} ({r.source})" for r in results[:5]]}
            
            Please provide:
            1. Confidence score for each result (0.0-1.0)
            2. Verification status (verified/needs_review/unverified)
            3. Any potential issues or concerns
            """
            
            verification_result = await execute_llm_task(
                task_type=TaskType.VERIFICATION,
                prompt=verification_prompt,
                use_ensemble=True
            )
            
            # Apply verification results to academic results
            verified_results = []
            for i, result in enumerate(results):
                if i < len(verification_result.get('confidence_scores', [])):
                    result.confidence_score = verification_result['confidence_scores'][i]
                    result.verification_status = verification_result.get('verification_status', ['unverified'])[i]
                verified_results.append(result)
            
            return ToolResponse(
                success=True,
                data=verified_results,
                metadata={
                    'verification_applied': True,
                    'average_confidence': sum(r.confidence_score for r in verified_results) / len(verified_results),
                    'timestamp': datetime.now().isoformat()
                }
            )
                
        except Exception as e:
            self.logger.warning(f"Verification failed: {e}, returning unverified results")
            return ToolResponse(success=True, data=results)
    
    def _convert_to_academic_results(self, mcp_data: List[Dict], tool_name: str) -> List[AcademicResult]:
        """Convert MCP tool results to standardized AcademicResult format."""
        results = []
        for item in mcp_data:
            try:
                result = AcademicResult(
                    title=item.get('title', ''),
                    authors=item.get('authors', []),
                    abstract=item.get('abstract', ''),
                    published=item.get('published', ''),
                    source=tool_name,
                    url=item.get('url', ''),
                    pdf_url=item.get('pdf_url', ''),
                    doi=item.get('doi', ''),
                    arxiv_id=item.get('arxiv_id', ''),
                    pmid=item.get('pmid', ''),
                    categories=item.get('categories', []),
                    confidence_score=item.get('confidence_score', 1.0),
                    verification_status=item.get('verification_status', 'unverified')
                )
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to convert MCP result: {e}")
                continue
        
        return results
    
    # MCP Search Methods (Innovation 6: Universal MCP Hub)
    async def _arxiv_mcp_search(self, query: str) -> ToolResponse:
        """Search ArXiv using MCP tools."""
        try:
            mcp_result = await execute_tool("arxiv", {"query": query, "max_results": self.max_results})
            if mcp_result.get('success', False):
                academic_results = self._convert_to_academic_results(mcp_result.get('data', []), "arxiv_mcp")
                return ToolResponse(success=True, data=academic_results)
            else:
                return ToolResponse(success=False, data=[], error=mcp_result.get('error', 'ArXiv MCP failed'))
        except Exception as e:
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _scholar_mcp_search(self, query: str) -> ToolResponse:
        """Search Google Scholar using MCP tools."""
        try:
            mcp_result = await execute_tool("scholar", {"query": query, "max_results": self.max_results})
            if mcp_result.get('success', False):
                academic_results = self._convert_to_academic_results(mcp_result.get('data', []), "scholar_mcp")
                return ToolResponse(success=True, data=academic_results)
            else:
                return ToolResponse(success=False, data=[], error=mcp_result.get('error', 'Scholar MCP failed'))
        except Exception as e:
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _pubmed_mcp_search(self, query: str) -> ToolResponse:
        """Search PubMed using MCP tools."""
        try:
            mcp_result = await execute_tool("pubmed", {"query": query, "max_results": self.max_results})
            if mcp_result.get('success', False):
                academic_results = self._convert_to_academic_results(mcp_result.get('data', []), "pubmed_mcp")
                return ToolResponse(success=True, data=academic_results)
            else:
                return ToolResponse(success=False, data=[], error=mcp_result.get('error', 'PubMed MCP failed'))
        except Exception as e:
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _ieee_mcp_search(self, query: str) -> ToolResponse:
        """Search IEEE using MCP tools."""
        try:
            mcp_result = await execute_tool("ieee", {"query": query, "max_results": self.max_results})
            if mcp_result.get('success', False):
                academic_results = self._convert_to_academic_results(mcp_result.get('data', []), "ieee_mcp")
                return ToolResponse(success=True, data=academic_results)
            else:
                return ToolResponse(success=False, data=[], error=mcp_result.get('error', 'IEEE MCP failed'))
        except Exception as e:
            return ToolResponse(success=False, data=[], error=str(e))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        return {
            'total_searches': getattr(self, '_total_searches', 0),
            'successful_searches': getattr(self, '_successful_searches', 0),
            'mcp_usage_ratio': getattr(self, '_mcp_usage_ratio', 0.0),
            'average_response_time': getattr(self, '_average_response_time', 0.0),
            'verification_success_rate': getattr(self, '_verification_success_rate', 0.0)
        }
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """검색 작업 수행 (Universal MCP Hub 통합)."""
        try:
            result = await self.arun(query)
            if result.success:
                # Convert AcademicResult to SearchResult
                search_results = []
                for item in result.data:
                    search_result = SearchResult(
                        title=item.title,
                        url=item.url,
                        snippet=item.abstract,
                        source=item.source,
                        published_date=datetime.strptime(item.published, '%Y-%m-%d') if item.published else None,
                        relevance_score=item.confidence_score,
                        confidence_score=item.confidence_score,
                        mcp_tool_used=item.source,
                        verification_score=1.0 if item.verification_status == 'verified' else 0.0
                    )
                    search_results.append(search_result)
                return search_results
            else:
                return []
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    async def get_content(self, url: str) -> Optional[str]:
        """URL에서 콘텐츠 가져오기."""
        try:
            # Use MCP fetch tool
            from src.core.mcp_integration import execute_tool
            result = await execute_tool("fetch", {"url": url})
            if result.get('success', False):
                return result.get('data', {}).get('content', '')
            else:
                self.logger.error(f"Failed to fetch content from {url}: {result.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to fetch content from {url}: {e}")
            return None