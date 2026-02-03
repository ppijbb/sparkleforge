#!/usr/bin/env python3
"""
Source Credibility Validator for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade 출처 신뢰도 평가 시스템.
URL 도메인 신뢰도, 게시일 최신성, 저자/기관 권위, 인용 횟수 등을
종합적으로 분석하여 신뢰도 점수를 제공합니다.

2025년 10월 최신 기술 스택:
- newspaper3k for web content extraction
- scholarly for academic citation data
- MCP tools for external verification
- Production-grade reliability patterns
"""

import asyncio
import logging
import re
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import time

# External libraries for content analysis
import newspaper
from newspaper import Article
import scholarly
import requests
from bs4 import BeautifulSoup

# MCP integration
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.mcp_integration import execute_tool
from src.core.reliability import execute_with_reliability
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class DomainType(Enum):
    """도메인 타입 분류."""
    ACADEMIC = "academic"
    GOVERNMENT = "government"
    NEWS = "news"
    BLOG = "blog"
    SOCIAL = "social"
    COMMERCIAL = "commercial"
    UNKNOWN = "unknown"


class AuthorityLevel(Enum):
    """권위 수준."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class SourceCredibility:
    """출처 신뢰도 평가 결과."""
    url: str
    domain_trust: float  # 0.0-1.0
    recency_score: float  # 0.0-1.0
    authority_score: float  # 0.0-1.0
    citation_count: int
    overall_score: float  # 0.0-1.0
    domain_type: DomainType
    authority_level: AuthorityLevel
    publication_date: Optional[datetime] = None
    author: Optional[str] = None
    institution: Optional[str] = None
    verification_status: str = "unverified"
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SourceValidator:
    """
    Production-grade 출처 신뢰도 평가 시스템.
    
    Features:
    - URL 도메인 신뢰도 평가 (학술, 정부, 언론사 등)
    - 게시일 최신성 체크
    - 저자/기관 신뢰도 평가
    - 인용 횟수 및 영향력 지표 수집
    - MCP 도구를 통한 외부 검증
    """
    
    def __init__(self):
        """소스 검증기 초기화."""
        self.domain_trust_db = self._initialize_domain_trust_db()
        self.academic_domains = self._initialize_academic_domains()
        self.news_domains = self._initialize_news_domains()
        self.government_domains = self._initialize_government_domains()
        
        # 신뢰도 가중치
        self.weights = {
            'domain_trust': 0.3,
            'recency': 0.2,
            'authority': 0.3,
            'citations': 0.2
        }
        
        logger.info("SourceValidator initialized with production-grade reliability")
    
    def _initialize_domain_trust_db(self) -> Dict[str, float]:
        """도메인 신뢰도 데이터베이스 초기화."""
        return {
            # 학술 기관 (높은 신뢰도)
            'edu': 0.95,
            'ac.uk': 0.95,
            'ac.jp': 0.95,
            'ac.kr': 0.95,
            'ac.cn': 0.95,
            
            # 정부 기관
            'gov': 0.90,
            'gov.uk': 0.90,
            'gov.jp': 0.90,
            'gov.kr': 0.90,
            
            # 국제 기관
            'un.org': 0.85,
            'who.int': 0.85,
            'worldbank.org': 0.85,
            'imf.org': 0.85,
            
            # 신뢰할 수 있는 언론사
            'reuters.com': 0.80,
            'bbc.com': 0.80,
            'ap.org': 0.80,
            'npr.org': 0.80,
            'wsj.com': 0.75,
            'nytimes.com': 0.75,
            'washingtonpost.com': 0.75,
            'theguardian.com': 0.75,
            
            # 전문 기관
            'nature.com': 0.90,
            'science.org': 0.90,
            'cell.com': 0.90,
            'nejm.org': 0.90,
            'jamanetwork.com': 0.90,
            
            # 일반적인 도메인
            'com': 0.50,
            'org': 0.60,
            'net': 0.45,
            'info': 0.40
        }
    
    def _initialize_academic_domains(self) -> List[str]:
        """학술 도메인 목록 초기화."""
        return [
            'arxiv.org', 'scholar.google.com', 'researchgate.net',
            'academia.edu', 'jstor.org', 'springer.com', 'ieee.org',
            'acm.org', 'pubmed.ncbi.nlm.nih.gov', 'scholar.google.co.kr'
        ]
    
    def _initialize_news_domains(self) -> List[str]:
        """뉴스 도메인 목록 초기화."""
        return [
            'reuters.com', 'bbc.com', 'ap.org', 'npr.org',
            'wsj.com', 'nytimes.com', 'washingtonpost.com',
            'theguardian.com', 'cnn.com', 'foxnews.com',
            'bloomberg.com', 'forbes.com', 'techcrunch.com'
        ]
    
    def _initialize_government_domains(self) -> List[str]:
        """정부 도메인 목록 초기화."""
        return [
            'whitehouse.gov', 'cdc.gov', 'nih.gov', 'nasa.gov',
            'nsf.gov', 'energy.gov', 'defense.gov', 'state.gov',
            'treasury.gov', 'commerce.gov'
        ]
    
    async def validate_source(self, url: str, content: Optional[str] = None) -> SourceCredibility:
        """
        출처를 종합적으로 검증합니다.
        
        Args:
            url: 검증할 URL
            content: 웹 콘텐츠 (선택사항)
            
        Returns:
            SourceCredibility: 검증 결과
        """
        try:
            logger.info(f"Validating source: {url}")
            
            # 도메인 신뢰도 평가
            domain_trust, domain_type = await self._evaluate_domain_trust(url)
            
            # 최신성 평가
            recency_score, publication_date = await self._evaluate_recency(url, content)
            
            # 권위 평가
            authority_score, authority_level, author, institution = await self._evaluate_authority(url, content)
            
            # 인용 횟수 평가
            citation_count = await self._evaluate_citations(url, content)
            
            # 종합 신뢰도 점수 계산
            overall_score = self._calculate_overall_score(
                domain_trust, recency_score, authority_score, citation_count
            )
            
            # 검증 상태 결정
            verification_status = self._determine_verification_status(overall_score)
            
            # 신뢰도 생성
            credibility = SourceCredibility(
                url=url,
                domain_trust=domain_trust,
                recency_score=recency_score,
                authority_score=authority_score,
                citation_count=citation_count,
                overall_score=overall_score,
                domain_type=domain_type,
                authority_level=authority_level,
                publication_date=publication_date,
                author=author,
                institution=institution,
                verification_status=verification_status,
                confidence=self._calculate_confidence(overall_score, domain_trust, authority_score),
                metadata={
                    'validation_timestamp': datetime.now(timezone.utc).isoformat(),
                    'domain_analysis': self._analyze_domain(url),
                    'content_analysis': self._analyze_content(content) if content else None
                }
            )
            
            logger.info(f"Source validation completed: {url} - Score: {overall_score:.3f}")
            return credibility
            
        except Exception as e:
            logger.error(f"Source validation failed for {url}: {e}")
            return self._create_failed_credibility(url, str(e))
    
    async def _evaluate_domain_trust(self, url: str) -> Tuple[float, DomainType]:
        """도메인 신뢰도를 평가합니다."""
        try:
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # 직접 매칭
            if domain in self.domain_trust_db:
                trust_score = self.domain_trust_db[domain]
                domain_type = self._classify_domain_type(domain)
                return trust_score, domain_type
            
            # TLD 기반 평가
            tld = domain.split('.')[-1]
            if tld in self.domain_trust_db:
                base_trust = self.domain_trust_db[tld]
                # 서브도메인에 따른 조정
                if 'www.' in domain:
                    base_trust *= 0.95
                if len(domain.split('.')) > 2:
                    base_trust *= 0.9
                return base_trust, self._classify_domain_type(domain)
            
            # 패턴 기반 평가
            trust_score = self._pattern_based_trust_evaluation(domain)
            domain_type = self._classify_domain_type(domain)
            
            return trust_score, domain_type
            
        except Exception as e:
            logger.warning(f"Domain trust evaluation failed for {url}: {e}")
            return 0.3, DomainType.UNKNOWN
    
    def _classify_domain_type(self, domain: str) -> DomainType:
        """도메인 타입을 분류합니다."""
        domain_lower = domain.lower()
        
        if any(academic in domain_lower for academic in self.academic_domains):
            return DomainType.ACADEMIC
        elif any(gov in domain_lower for gov in self.government_domains):
            return DomainType.GOVERNMENT
        elif any(news in domain_lower for news in self.news_domains):
            return DomainType.NEWS
        elif 'blog' in domain_lower or 'medium.com' in domain_lower:
            return DomainType.BLOG
        elif any(social in domain_lower for social in ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com']):
            return DomainType.SOCIAL
        elif domain_lower.endswith('.com'):
            return DomainType.COMMERCIAL
        else:
            return DomainType.UNKNOWN
    
    def _pattern_based_trust_evaluation(self, domain: str) -> float:
        """패턴 기반 신뢰도 평가."""
        domain_lower = domain.lower()
        
        # 학술 관련 키워드
        academic_keywords = ['research', 'university', 'institute', 'academy', 'college', 'school']
        if any(keyword in domain_lower for keyword in academic_keywords):
            return 0.8
        
        # 정부 관련 키워드
        gov_keywords = ['government', 'ministry', 'department', 'agency', 'bureau']
        if any(keyword in domain_lower for keyword in gov_keywords):
            return 0.85
        
        # 뉴스 관련 키워드
        news_keywords = ['news', 'times', 'post', 'journal', 'herald', 'tribune']
        if any(keyword in domain_lower for keyword in news_keywords):
            return 0.7
        
        # 기본 신뢰도
        return 0.5
    
    async def _evaluate_recency(self, url: str, content: Optional[str] = None) -> Tuple[float, Optional[datetime]]:
        """최신성을 평가합니다."""
        try:
            # MCP 도구를 사용하여 웹 콘텐츠 가져오기
            if not content:
                content_result = await execute_tool("fetch", {"url": url})
                if content_result.get('success', False):
                    content = content_result.get('data', {}).get('content', '')
            
            if not content:
                return 0.5, None
            
            # 게시일 추출
            publication_date = await self._extract_publication_date(url, content)
            
            if not publication_date:
                return 0.5, None
            
            # 최신성 점수 계산 (최근일수록 높은 점수)
            now = datetime.now(timezone.utc)
            days_old = (now - publication_date).days
            
            if days_old <= 1:
                recency_score = 1.0
            elif days_old <= 7:
                recency_score = 0.9
            elif days_old <= 30:
                recency_score = 0.8
            elif days_old <= 90:
                recency_score = 0.6
            elif days_old <= 365:
                recency_score = 0.4
            else:
                recency_score = 0.2
            
            return recency_score, publication_date
            
        except Exception as e:
            logger.warning(f"Recency evaluation failed for {url}: {e}")
            return 0.5, None
    
    async def _extract_publication_date(self, url: str, content: str) -> Optional[datetime]:
        """게시일을 추출합니다."""
        try:
            # newspaper3k를 사용한 메타데이터 추출
            article = Article(url)
            article.set_html(content)
            article.parse()
            
            if article.publish_date:
                return article.publish_date.replace(tzinfo=timezone.utc)
            
            # HTML에서 직접 추출
            soup = BeautifulSoup(content, 'html.parser')
            
            # 다양한 메타 태그에서 날짜 추출
            date_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="pubdate"]',
                'meta[name="date"]',
                'meta[property="og:published_time"]',
                'time[datetime]'
            ]
            
            for selector in date_selectors:
                element = soup.select_one(selector)
                if element:
                    date_str = element.get('content') or element.get('datetime')
                    if date_str:
                        try:
                            # ISO 형식 파싱
                            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        except:
                            continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Publication date extraction failed for {url}: {e}")
            return None
    
    async def _evaluate_authority(self, url: str, content: Optional[str] = None) -> Tuple[float, AuthorityLevel, Optional[str], Optional[str]]:
        """권위를 평가합니다."""
        try:
            # MCP 도구를 사용하여 콘텐츠 가져오기
            if not content:
                content_result = await execute_tool("fetch", {"url": url})
                if content_result.get('success', False):
                    content = content_result.get('data', {}).get('content', '')
            
            if not content:
                return 0.5, AuthorityLevel.UNKNOWN, None, None
            
            # 저자 및 기관 추출
            author, institution = await self._extract_author_info(url, content)
            
            # 권위 점수 계산
            authority_score = 0.5  # 기본 점수
            
            # 저자 정보가 있는 경우
            if author:
                authority_score += 0.2
                # 저자 이름에서 학위나 직책 확인
                if any(title in author.lower() for title in ['dr.', 'prof.', 'professor', 'phd', 'md']):
                    authority_score += 0.2
            
            # 기관 정보가 있는 경우
            if institution:
                authority_score += 0.2
                # 신뢰할 수 있는 기관인지 확인
                if any(trusted_inst in institution.lower() for trusted_inst in 
                      ['university', 'institute', 'hospital', 'government', 'research']):
                    authority_score += 0.1
            
            # 도메인 기반 권위 조정
            domain_authority = self._get_domain_authority(url)
            authority_score = (authority_score + domain_authority) / 2
            
            # 권위 수준 결정
            if authority_score >= 0.8:
                authority_level = AuthorityLevel.HIGH
            elif authority_score >= 0.6:
                authority_level = AuthorityLevel.MEDIUM
            else:
                authority_level = AuthorityLevel.LOW
            
            return min(1.0, authority_score), authority_level, author, institution
            
        except Exception as e:
            logger.warning(f"Authority evaluation failed for {url}: {e}")
            return 0.5, AuthorityLevel.UNKNOWN, None, None
    
    async def _extract_author_info(self, url: str, content: str) -> Tuple[Optional[str], Optional[str]]:
        """저자 및 기관 정보를 추출합니다."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # 저자 추출
            author = None
            author_selectors = [
                'meta[name="author"]',
                'meta[property="article:author"]',
                '.author',
                '.byline',
                '[rel="author"]'
            ]
            
            for selector in author_selectors:
                element = soup.select_one(selector)
                if element:
                    author = element.get('content') or element.get_text(strip=True)
                    if author:
                        break
            
            # 기관 추출
            institution = None
            institution_selectors = [
                'meta[name="institution"]',
                '.institution',
                '.affiliation',
                '.organization'
            ]
            
            for selector in institution_selectors:
                element = soup.select_one(selector)
                if element:
                    institution = element.get('content') or element.get_text(strip=True)
                    if institution:
                        break
            
            return author, institution
            
        except Exception as e:
            logger.warning(f"Author info extraction failed for {url}: {e}")
            return None, None
    
    def _get_domain_authority(self, url: str) -> float:
        """도메인 기반 권위 점수를 반환합니다."""
        domain = urllib.parse.urlparse(url).netloc.lower()
        
        # 학술 도메인
        if any(academic in domain for academic in self.academic_domains):
            return 0.9
        
        # 정부 도메인
        if any(gov in domain for gov in self.government_domains):
            return 0.85
        
        # 신뢰할 수 있는 언론사
        if any(news in domain for news in self.news_domains):
            return 0.7
        
        return 0.5
    
    async def _evaluate_citations(self, url: str, content: Optional[str] = None) -> int:
        """인용 횟수를 평가합니다."""
        try:
            # MCP 도구를 사용하여 학술 검색
            search_result = await execute_tool("scholar", {
                "query": url,
                "max_results": 5
            })
            
            if search_result.get('success', False):
                citations = search_result.get('data', [])
                if citations:
                    # 첫 번째 결과의 인용 횟수 반환
                    return citations[0].get('cited_by', 0)
            
            return 0
            
        except Exception as e:
            logger.warning(f"Citation evaluation failed for {url}: {e}")
            return 0
    
    def _calculate_overall_score(self, domain_trust: float, recency: float, 
                               authority: float, citations: int) -> float:
        """종합 신뢰도 점수를 계산합니다."""
        # 인용 횟수를 0-1 스케일로 변환
        citation_score = min(1.0, citations / 100)  # 100회 이상은 1.0
        
        # 가중 평균 계산
        overall_score = (
            self.weights['domain_trust'] * domain_trust +
            self.weights['recency'] * recency +
            self.weights['authority'] * authority +
            self.weights['citations'] * citation_score
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _determine_verification_status(self, overall_score: float) -> str:
        """검증 상태를 결정합니다."""
        if overall_score >= 0.8:
            return "verified"
        elif overall_score >= 0.6:
            return "partially_verified"
        else:
            return "unverified"
    
    def _calculate_confidence(self, overall_score: float, domain_trust: float, 
                            authority_score: float) -> float:
        """신뢰도를 계산합니다."""
        # 점수 일관성 기반 신뢰도
        scores = [overall_score, domain_trust, authority_score]
        variance = sum((score - sum(scores)/len(scores))**2 for score in scores) / len(scores)
        consistency = 1.0 - min(1.0, variance)
        
        # 기본 신뢰도
        base_confidence = overall_score * 0.8 + consistency * 0.2
        
        return min(1.0, max(0.0, base_confidence))
    
    def _analyze_domain(self, url: str) -> Dict[str, Any]:
        """도메인 분석 정보를 반환합니다."""
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        
        return {
            'domain': domain,
            'tld': domain.split('.')[-1] if '.' in domain else '',
            'subdomain_count': len(domain.split('.')) - 1,
            'is_https': parsed.scheme == 'https',
            'path_depth': len([p for p in parsed.path.split('/') if p])
        }
    
    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """콘텐츠 분석 정보를 반환합니다."""
        if not content:
            return {}
        
        soup = BeautifulSoup(content, 'html.parser')
        
        return {
            'word_count': len(content.split()),
            'has_meta_description': bool(soup.find('meta', attrs={'name': 'description'})),
            'has_meta_keywords': bool(soup.find('meta', attrs={'name': 'keywords'})),
            'has_structured_data': bool(soup.find('script', type='application/ld+json')),
            'heading_count': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
        }
    
    def _create_failed_credibility(self, url: str, error: str) -> SourceCredibility:
        """실패한 검증 결과를 생성합니다."""
        return SourceCredibility(
            url=url,
            domain_trust=0.0,
            recency_score=0.0,
            authority_score=0.0,
            citation_count=0,
            overall_score=0.0,
            domain_type=DomainType.UNKNOWN,
            authority_level=AuthorityLevel.UNKNOWN,
            verification_status="failed",
            confidence=0.0,
            metadata={'error': error}
        )
    
    async def batch_validate(self, urls: List[str]) -> List[SourceCredibility]:
        """여러 URL을 배치로 검증합니다."""
        tasks = [self.validate_source(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외 처리
        validated_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch validation failed for {urls[i]}: {result}")
                validated_results.append(self._create_failed_credibility(urls[i], str(result)))
            else:
                validated_results.append(result)
        
        return validated_results
