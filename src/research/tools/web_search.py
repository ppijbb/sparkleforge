"""
Advanced Web Search Tool (v2.0 - 8대 혁신 통합)

Universal MCP Hub, Production-Grade Reliability, Multi-Model Orchestration을
통합한 고도화된 웹 검색 도구.
"""

import logging
import asyncio
import os
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime

from src.research.tools.base_tool import BaseResearchTool, SearchResult, ToolType
from src.core.mcp_integration import execute_tool, get_best_tool_for_task, ToolCategory
from src.core.reliability import execute_with_reliability

logger = logging.getLogger(__name__)


class SearchProvider(Enum):
    """사용 가능한 검색 제공자."""
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    GOOGLE = "google"
    BING = "bing"
    EXA = "exa"
    MCP_GSEARCH = "mcp_gsearch"
    MCP_TAVILY = "mcp_tavily"
    MCP_EXA = "mcp_exa"


class WebSearchTool(BaseResearchTool):
    """8대 혁신을 통합한 고도화된 웹 검색 도구."""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화."""
        super().__init__(config)
        self.tool_type = ToolType.SEARCH
        
        # 검색 제공자 설정 (MCP 전용)
        self.primary_provider = SearchProvider(config.get('primary_provider', 'mcp_gsearch'))
        
        # MCP 도구 매핑
        self.mcp_tool_mapping = {
            'web_search': 'g-search',
            'tavily_search': 'tavily',
            'exa_search': 'exa',
            'duckduckgo_search': 'duckduckgo'
        }
        
        logger.info(f"WebSearchTool initialized with primary provider: {self.primary_provider.value}")
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """웹 검색 실행 (Universal MCP Hub + Production-Grade Reliability)."""
        max_results = kwargs.get('max_results', self.max_results)
        search_depth = kwargs.get('search_depth', 'basic')
        include_answer = kwargs.get('include_answer', True)
        
        # Universal MCP Hub를 통한 검색
        search_result = await self.execute_with_mcp(
            operation='web_search',
            parameters={
                'query': query,
                'max_results': max_results,
                'search_depth': search_depth,
                'include_answer': include_answer
            },
            alternative_tools=['tavily', 'exa', 'g-search']
        )
        
        return self._convert_to_search_results(search_result['data'], query)
    
    async def get_content(self, url: str) -> Optional[str]:
        """URL에서 콘텐츠 가져오기 (Universal MCP Hub)."""
        # MCP fetch 도구 사용
        fetch_result = await self.execute_with_mcp(
            operation='data_fetch',
            parameters={'url': url},
            alternative_tools=['fetch', 'filesystem']
        )
        
        return fetch_result['data'].get('content', '')
    
    
    
    def _convert_to_search_results(self, data: Dict[str, Any], query: str) -> List[SearchResult]:
        """데이터를 SearchResult 객체로 변환."""
        results = []
        search_results = data.get('results', [])
        provider = data.get('provider', 'unknown')
        
        for item in search_results:
            # 관련성 점수 계산
            title = item.get('title', '')
            snippet = item.get('content', item.get('snippet', item.get('body', '')))
            relevance_score = self._calculate_relevance_score(query, title, snippet)
            
            # 신뢰도 점수 계산
            confidence_score = self._calculate_confidence_score({
                'source': provider,
                'score': item.get('score', 0.8),
                'metadata': item.get('metadata', {})
            })
            
            result = SearchResult(
                title=title,
                url=item.get('url', item.get('href', '')),
                snippet=snippet,
                source=provider,
                relevance_score=relevance_score,
                confidence_score=confidence_score,
                mcp_tool_used=provider if provider.startswith('mcp_') else None,
                execution_time=0.0,
                metadata={
                    'original_score': item.get('score', 0.8),
                    'provider': provider,
                    'search_timestamp': datetime.now().isoformat()
                }
            )
            
            results.append(result)
        
        return results
    
    async def search_with_streaming(
        self,
        query: str,
        callback: callable = None,
        **kwargs
    ) -> List[SearchResult]:
        """스트리밍 검색 (Streaming Pipeline - 혁신 5)."""
        logger.info(f"Starting streaming search for: {query}")
        
        # 즉시 초기 응답
        if callback:
            await callback({
                'status': 'started',
                'query': query,
                'timestamp': datetime.now().isoformat()
            })
        
        # 검색 실행
        results = await self.search(query, **kwargs)
        
        # 부분 결과 스트리밍
        if callback:
            for i, result in enumerate(results):
                await callback({
                    'status': 'partial',
                    'result': result,
                    'index': i,
                    'total': len(results),
                    'timestamp': datetime.now().isoformat()
                })
        
        # 최종 결과
        if callback:
            await callback({
                'status': 'completed',
                'query': query,
                'total_results': len(results),
                'timestamp': datetime.now().isoformat()
            })
        
        return results
    
    async def search_with_verification(
        self,
        query: str,
        **kwargs
    ) -> List[SearchResult]:
        """검증이 포함된 검색 (Continuous Verification - 혁신 4)."""
        # 기본 검색 실행
        results = await self.search(query, **kwargs)
        
        # 검증 실행
        verified_results = []
        for result in results:
            # 신뢰도 기반 검증
            if result.confidence_score >= 0.7:
                verified_results.append(result)
            else:
                # 낮은 신뢰도 결과는 추가 검증
                verification_score = await self._verify_result(result)
                if verification_score >= 0.6:
                    result.verification_score = verification_score
                    verified_results.append(result)
        
        return verified_results
    
    async def _verify_result(self, result: SearchResult) -> float:
        """결과 검증."""
        # 실제 구현에서는 더 정교한 검증 로직 사용
        # 여기서는 간단한 신뢰도 점수 반환
        return result.confidence_score
    
    def get_search_providers(self) -> List[str]:
        """사용 가능한 검색 제공자 반환."""
        providers = []
        
        # MCP 제공자만 사용
        if self.mcp_config.enabled:
            providers.extend(['mcp_gsearch', 'mcp_tavily', 'mcp_exa'])
        
        return list(set(providers))
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환."""
        stats = self.get_performance_stats()
        return {
            'tool_name': self.name,
            'tool_type': self.tool_type.value,
            'status': self.status.value,
            'success_rate': f"{stats['success_rate']:.2%}",
            'average_response_time': f"{stats['average_response_time']:.2f}s",
            'total_requests': stats['total_requests'],
            'available_providers': self.get_search_providers(),
            'mcp_enabled': self.mcp_config.enabled,
            'last_used': stats['last_used']
        }


# Global web search tool instance
def create_web_search_tool(config: Dict[str, Any] = None) -> WebSearchTool:
    """웹 검색 도구 생성."""
    if config is None:
        config = {
            'name': 'web_search',
            'enabled': True,
            'timeout': 30,
            'max_results': 10,
            'primary_provider': 'mcp_gsearch',
            'alternative_tools': ['mcp_tavily', 'mcp_exa']
        }
    
    return WebSearchTool(config)


# Convenience functions
async def search_web(query: str, max_results: int = 10, **kwargs) -> List[SearchResult]:
    """웹 검색 실행."""
    tool = create_web_search_tool()
    return await tool.search_with_reliability(query, max_results=max_results, **kwargs)


async def search_web_streaming(query: str, callback: callable = None, **kwargs) -> List[SearchResult]:
    """스트리밍 웹 검색 실행."""
    tool = create_web_search_tool()
    return await tool.search_with_streaming(query, callback, **kwargs)


async def search_web_with_verification(query: str, **kwargs) -> List[SearchResult]:
    """검증이 포함된 웹 검색 실행."""
    tool = create_web_search_tool()
    return await tool.search_with_verification(query, **kwargs)