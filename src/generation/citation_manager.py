#!/usr/bin/env python3
"""
Citation Manager for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade 인용 관리 시스템.
표준 인용 형식 생성, 인용 번호 자동 할당,
References 섹션 자동 생성, 본문 내 인라인 인용 삽입을 제공합니다.

2025년 10월 최신 기술 스택:
- MCP tools for source data collection
- Production-grade citation formatting
- Multiple citation styles support
- Automatic reference management
"""

import re
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import uuid

# MCP integration
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.mcp_integration import execute_tool
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class CitationStyle(Enum):
    """인용 스타일."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    IEEE = "ieee"
    NATURE = "nature"


@dataclass
class Source:
    """출처 정보."""
    id: str
    title: str
    authors: List[str]
    publication_date: Optional[datetime]
    url: Optional[str]
    doi: Optional[str]
    journal: Optional[str]
    volume: Optional[str]
    issue: Optional[str]
    pages: Optional[str]
    publisher: Optional[str]
    place: Optional[str]
    source_type: str  # 'journal', 'book', 'website', 'report', etc.
    credibility_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Citation:
    """인용 정보."""
    citation_id: str
    source_id: str
    in_text_citation: str
    reference_entry: str
    page_number: Optional[str] = None
    context: Optional[str] = None


class CitationManager:
    """
    Production-grade 인용 관리 시스템 (9대 혁신 - Citation Management 강화).
    
    Features:
    - 표준 인용 형식 생성 (APA, MLA, Chicago 등)
    - 인용 번호 자동 할당
    - References 섹션 자동 생성
    - 본문 내 인라인 인용 삽입
    - 체계적인 ID 생성 (PLAN-XX, CIT-X-XX)
    - ref_number 매핑 및 중복 제거
    - Thread-safe async 메서드 (병렬 실행 지원)
    """
    
    def __init__(self, default_style: CitationStyle = CitationStyle.APA, research_id: Optional[str] = None):
        """인용 매니저 초기화."""
        self.default_style = default_style
        self.sources: Dict[str, Source] = {}
        self.citations: Dict[str, Citation] = {}
        self.citation_counter = 0
        
        # 9대 혁신: 체계적인 ID 생성 시스템
        self.research_id = research_id or f"research_{int(datetime.now().timestamp())}"
        self._plan_counter = 0  # PLAN-XX 형식 (Planning 단계)
        self._block_counters: Dict[str, int] = {}  # CIT-X-XX 형식 (Research 단계, X=block 번호)
        self._ref_number_map: Dict[str, int] = {}  # citation_id -> ref_number (1-based) 매핑
        self._lock = asyncio.Lock()  # Thread-safe 병렬 실행 지원
        
        # 인용 스타일별 포맷터
        self.formatters = {
            CitationStyle.APA: self._format_apa,
            CitationStyle.MLA: self._format_mla,
            CitationStyle.CHICAGO: self._format_chicago,
            CitationStyle.HARVARD: self._format_harvard,
            CitationStyle.IEEE: self._format_ieee,
            CitationStyle.NATURE: self._format_nature
        }
        
        logger.info(f"CitationManager initialized with {default_style.value} style (research_id: {self.research_id})")
    
    def add_citation(
        self,
        source: Source,
        style: Optional[CitationStyle] = None,
        page_number: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        인용을 추가합니다.
        
        Args:
            source: 출처 정보
            style: 인용 스타일 (기본값 사용)
            page_number: 페이지 번호
            context: 인용 컨텍스트
            
        Returns:
            str: 인용 ID
        """
        if style is None:
            style = self.default_style
        
        # 출처 저장
        self.sources[source.id] = source
        
        # 인용 ID 생성
        citation_id = f"cite_{self.citation_counter + 1}"
        self.citation_counter += 1
        
        # 인용 형식 생성
        in_text_citation = self._generate_inline_citation(source, style, page_number)
        reference_entry = self._generate_reference_entry(source, style)
        
        # 인용 저장
        citation = Citation(
            citation_id=citation_id,
            source_id=source.id,
            in_text_citation=in_text_citation,
            reference_entry=reference_entry,
            page_number=page_number,
            context=context
        )
        
        self.citations[citation_id] = citation
        
        logger.info(f"Citation added: {citation_id} for source {source.id}")
        return citation_id
    
    def generate_inline_citation(self, citation_id: str) -> str:
        """인라인 인용을 생성합니다."""
        if citation_id not in self.citations:
            logger.warning(f"Citation not found: {citation_id}")
            return f"[{citation_id}]"
        
        return self.citations[citation_id].in_text_citation
    
    def generate_references_section(self, style: Optional[CitationStyle] = None) -> str:
        """References 섹션을 생성합니다."""
        if style is None:
            style = self.default_style
        
        if not self.sources:
            return "No references available."
        
        # 출처를 알파벳 순으로 정렬
        sorted_sources = sorted(
            self.sources.values(),
            key=lambda s: self._get_sort_key(s)
        )
        
        # References 섹션 생성
        references = [f"## References\n"]
        
        for i, source in enumerate(sorted_sources, 1):
            reference_entry = self._generate_reference_entry(source, style)
            references.append(f"{i}. {reference_entry}\n")
        
        return "".join(references)
    
    def validate_citation_completeness(self) -> Dict[str, Any]:
        """인용 완성도를 검증합니다."""
        validation_results = {
            'total_sources': len(self.sources),
            'total_citations': len(self.citations),
            'missing_fields': [],
            'incomplete_sources': [],
            'validation_score': 0.0
        }
        
        # 필수 필드 검증
        required_fields = ['title', 'authors', 'source_type']
        
        for source_id, source in self.sources.items():
            missing = []
            for field in required_fields:
                if not getattr(source, field, None):
                    missing.append(field)
            
            if missing:
                validation_results['incomplete_sources'].append({
                    'source_id': source_id,
                    'missing_fields': missing
                })
        
        # 검증 점수 계산
        total_fields = len(self.sources) * len(required_fields)
        missing_fields = sum(
            len(incomplete['missing_fields'])
            for incomplete in validation_results['incomplete_sources']
        )
        
        validation_results['validation_score'] = max(0.0, 1.0 - (missing_fields / total_fields))
        
        return validation_results
    
    def _generate_inline_citation(
        self,
        source: Source,
        style: CitationStyle,
        page_number: Optional[str] = None
    ) -> str:
        """인라인 인용을 생성합니다."""
        formatter = self.formatters.get(style, self._format_apa)
        return formatter(source, page_number, inline=True)
    
    def _generate_reference_entry(
        self,
        source: Source,
        style: CitationStyle
    ) -> str:
        """참조 항목을 생성합니다."""
        formatter = self.formatters.get(style, self._format_apa)
        return formatter(source, inline=False)
    
    def _format_apa(
        self,
        source: Source,
        page_number: Optional[str] = None,
        inline: bool = False
    ) -> str:
        """APA 형식으로 포맷합니다."""
        if inline:
            # 인라인 인용: (Author, Year)
            authors = self._format_authors_apa(source.authors)
            year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
            
            citation = f"({authors}, {year}"
            if page_number:
                citation += f", p. {page_number}"
            citation += ")"
            
            return citation
        else:
            # 참조 항목
            authors = self._format_authors_apa(source.authors)
            year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
            title = source.title
            
            if source.source_type == 'journal':
                journal = source.journal or 'Unknown Journal'
                volume = source.volume or ''
                issue = f"({source.issue})" if source.issue else ''
                pages = f", {source.pages}" if source.pages else ''
                doi = f" https://doi.org/{source.doi}" if source.doi else ''
                
                return f"{authors} ({year}). {title}. {journal}, {volume}{issue}{pages}.{doi}"
            
            elif source.source_type == 'book':
                publisher = source.publisher or 'Unknown Publisher'
                place = source.place or 'Unknown Place'
                
                return f"{authors} ({year}). {title}. {place}: {publisher}."
            
            elif source.source_type == 'website':
                url = source.url or ''
                access_date = datetime.now(timezone.utc).strftime('%B %d, %Y')
                
                return f"{authors} ({year}). {title}. Retrieved {access_date}, from {url}"
            
            else:
                return f"{authors} ({year}). {title}."
    
    def _format_mla(
        self,
        source: Source,
        page_number: Optional[str] = None,
        inline: bool = False
    ) -> str:
        """MLA 형식으로 포맷합니다."""
        if inline:
            # 인라인 인용: (Author Page)
            authors = self._format_authors_mla(source.authors)
            citation = f"({authors}"
            if page_number:
                citation += f" {page_number}"
            citation += ")"
            
            return citation
        else:
            # 참조 항목
            authors = self._format_authors_mla(source.authors)
            title = source.title
            
            if source.source_type == 'journal':
                journal = source.journal or 'Unknown Journal'
                volume = source.volume or ''
                issue = source.issue or ''
                year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
                pages = source.pages or ''
                url = source.url or ''
                
                return f'"{title}." {journal}, vol. {volume}, no. {issue}, {year}, pp. {pages}. {url}'
            
            elif source.source_type == 'book':
                publisher = source.publisher or 'Unknown Publisher'
                year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
                
                return f"{title}. {publisher}, {year}."
            
            else:
                return f"{title}."
    
    def _format_chicago(
        self,
        source: Source,
        page_number: Optional[str] = None,
        inline: bool = False
    ) -> str:
        """Chicago 형식으로 포맷합니다."""
        if inline:
            # 인라인 인용: (Author Year, Page)
            authors = self._format_authors_chicago(source.authors)
            year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
            
            citation = f"({authors} {year}"
            if page_number:
                citation += f", {page_number}"
            citation += ")"
            
            return citation
        else:
            # 참조 항목
            authors = self._format_authors_chicago(source.authors)
            title = source.title
            
            if source.source_type == 'journal':
                journal = source.journal or 'Unknown Journal'
                volume = source.volume or ''
                issue = source.issue or ''
                year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
                pages = source.pages or ''
                url = source.url or ''
                
                return f"{authors}. \"{title}.\" {journal} {volume}, no. {issue} ({year}): {pages}. {url}"
            
            elif source.source_type == 'book':
                publisher = source.publisher or 'Unknown Publisher'
                place = source.place or 'Unknown Place'
                year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
                
                return f"{authors}. {title}. {place}: {publisher}, {year}."
            
            else:
                return f"{authors}. \"{title}.\""
    
    def _format_harvard(
        self,
        source: Source,
        page_number: Optional[str] = None,
        inline: bool = False
    ) -> str:
        """Harvard 형식으로 포맷합니다."""
        if inline:
            # 인라인 인용: (Author Year: Page)
            authors = self._format_authors_harvard(source.authors)
            year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
            
            citation = f"({authors} {year}"
            if page_number:
                citation += f": {page_number}"
            citation += ")"
            
            return citation
        else:
            # 참조 항목
            authors = self._format_authors_harvard(source.authors)
            year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
            title = source.title
            
            if source.source_type == 'journal':
                journal = source.journal or 'Unknown Journal'
                volume = source.volume or ''
                issue = source.issue or ''
                pages = source.pages or ''
                
                return f"{authors} {year}, '{title}', {journal}, vol. {volume}, no. {issue}, pp. {pages}."
            
            elif source.source_type == 'book':
                publisher = source.publisher or 'Unknown Publisher'
                place = source.place or 'Unknown Place'
                
                return f"{authors} {year}, {title}, {publisher}, {place}."
            
            else:
                return f"{authors} {year}, '{title}'."
    
    def _format_ieee(
        self,
        source: Source,
        page_number: Optional[str] = None,
        inline: bool = False
    ) -> str:
        """IEEE 형식으로 포맷합니다."""
        if inline:
            # 인라인 인용: [1]
            return f"[{self.citation_counter}]"
        else:
            # 참조 항목
            authors = self._format_authors_ieee(source.authors)
            title = source.title
            
            if source.source_type == 'journal':
                journal = source.journal or 'Unknown Journal'
                volume = source.volume or ''
                issue = source.issue or ''
                year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
                pages = source.pages or ''
                
                return f"{authors}, \"{title},\" {journal}, vol. {volume}, no. {issue}, pp. {pages}, {year}."
            
            elif source.source_type == 'book':
                publisher = source.publisher or 'Unknown Publisher'
                place = source.place or 'Unknown Place'
                year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
                
                return f"{authors}, {title}. {place}: {publisher}, {year}."
            
            else:
                return f"{authors}, \"{title}.\""
    
    def _format_nature(
        self,
        source: Source,
        page_number: Optional[str] = None,
        inline: bool = False
    ) -> str:
        """Nature 형식으로 포맷합니다."""
        if inline:
            # 인라인 인용: (Author, Year)
            authors = self._format_authors_nature(source.authors)
            year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
            
            citation = f"({authors}, {year}"
            if page_number:
                citation += f", p. {page_number}"
            citation += ")"
            
            return citation
        else:
            # 참조 항목
            authors = self._format_authors_nature(source.authors)
            title = source.title
            
            if source.source_type == 'journal':
                journal = source.journal or 'Unknown Journal'
                volume = source.volume or ''
                year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
                pages = source.pages or ''
                doi = f" https://doi.org/{source.doi}" if source.doi else ''
                
                return f"{authors} {title}. {journal} {volume}, {pages} ({year}).{doi}"
            
            elif source.source_type == 'book':
                publisher = source.publisher or 'Unknown Publisher'
                year = source.publication_date.strftime('%Y') if source.publication_date else 'n.d.'
                
                return f"{authors} {title}. {publisher}, {year}."
            
            else:
                return f"{authors} {title}."
    
    def _format_authors_apa(self, authors: List[str]) -> str:
        """APA 형식으로 저자를 포맷합니다."""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        else:
            return f"{', '.join(authors[:-1])}, & {authors[-1]}"
    
    def _format_authors_mla(self, authors: List[str]) -> str:
        """MLA 형식으로 저자를 포맷합니다."""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} and {authors[1]}"
        else:
            return f"{', '.join(authors[:-1])}, and {authors[-1]}"
    
    def _format_authors_chicago(self, authors: List[str]) -> str:
        """Chicago 형식으로 저자를 포맷합니다."""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} and {authors[1]}"
        else:
            return f"{', '.join(authors[:-1])}, and {authors[-1]}"
    
    def _format_authors_harvard(self, authors: List[str]) -> str:
        """Harvard 형식으로 저자를 포맷합니다."""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} and {authors[1]}"
        else:
            return f"{', '.join(authors[:-1])}, and {authors[-1]}"
    
    def _format_authors_ieee(self, authors: List[str]) -> str:
        """IEEE 형식으로 저자를 포맷합니다."""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) <= 6:
            return ", ".join(authors)
        else:
            return f"{', '.join(authors[:3])} et al."
    
    def _format_authors_nature(self, authors: List[str]) -> str:
        """Nature 형식으로 저자를 포맷합니다."""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) <= 5:
            return ", ".join(authors)
        else:
            return f"{', '.join(authors[:3])} et al."
    
    def _get_sort_key(self, source: Source) -> str:
        """정렬 키를 반환합니다."""
        if source.authors:
            return source.authors[0].lower()
        return source.title.lower()
    
    def export_citations(self, format: str = "json") -> str:
        """인용을 내보냅니다."""
        if format == "json":
            data = {
                'sources': {sid: {
                    'id': source.id,
                    'title': source.title,
                    'authors': source.authors,
                    'publication_date': source.publication_date.isoformat() if source.publication_date else None,
                    'url': source.url,
                    'doi': source.doi,
                    'journal': source.journal,
                    'source_type': source.source_type,
                    'credibility_score': source.credibility_score
                } for sid, source in self.sources.items()},
                'citations': {cid: {
                    'citation_id': citation.citation_id,
                    'source_id': citation.source_id,
                    'in_text_citation': citation.in_text_citation,
                    'reference_entry': citation.reference_entry,
                    'page_number': citation.page_number,
                    'context': citation.context
                } for cid, citation in self.citations.items()}
            }
            return json.dumps(data, ensure_ascii=False, indent=2)
        
        elif format == "bibtex":
            return self._export_bibtex()
        
        else:
            return self.generate_references_section()
    
    def _export_bibtex(self) -> str:
        """BibTeX 형식으로 내보냅니다."""
        bibtex_entries = []
        
        for source in self.sources.values():
            if source.source_type == 'journal':
                entry_type = '@article'
            elif source.source_type == 'book':
                entry_type = '@book'
            else:
                entry_type = '@misc'
            
            bibtex = f"{entry_type}{{{source.id},\n"
            bibtex += f"  title = {{{source.title}}},\n"
            
            if source.authors:
                bibtex += f"  author = {{{' and '.join(source.authors)}}},\n"
            
            if source.publication_date:
                bibtex += f"  year = {{{source.publication_date.year}}},\n"
            
            if source.journal:
                bibtex += f"  journal = {{{source.journal}}},\n"
            
            if source.volume:
                bibtex += f"  volume = {{{source.volume}}},\n"
            
            if source.pages:
                bibtex += f"  pages = {{{source.pages}}},\n"
            
            if source.doi:
                bibtex += f"  doi = {{{source.doi}}},\n"
            
            if source.url:
                bibtex += f"  url = {{{source.url}}},\n"
            
            bibtex += "}\n"
            bibtex_entries.append(bibtex)
        
        return "\n".join(bibtex_entries)
    
    # ========== 9대 혁신: 체계적인 ID 생성 시스템 ==========
    
    def generate_plan_citation_id(self) -> str:
        """
        Planning 단계용 citation ID 생성 (PLAN-XX 형식).
        
        Returns:
            Citation ID in PLAN-XX format
        """
        self._plan_counter += 1
        return f"PLAN-{self._plan_counter:02d}"
    
    def generate_research_citation_id(self, block_id: str = "") -> str:
        """
        Research 단계용 citation ID 생성 (CIT-X-XX 형식).
        
        Args:
            block_id: Block ID (e.g., "block_3") - optional
        
        Returns:
            Citation ID in CIT-X-XX format
        """
        # Extract block number from block_id
        block_num = 0
        if block_id:
            try:
                if "_" in block_id:
                    block_num = int(block_id.split("_")[1])
            except (ValueError, IndexError):
                block_num = 0
        
        # Increment counter for this block
        block_key = str(block_num)
        if block_key not in self._block_counters:
            self._block_counters[block_key] = 0
        self._block_counters[block_key] += 1
        
        return f"CIT-{block_num}-{self._block_counters[block_key]:02d}"
    
    def get_next_citation_id(self, stage: str = "research", block_id: str = "") -> str:
        """
        다음 사용 가능한 citation ID 반환.
        
        Args:
            stage: "planning" or "research"
            block_id: Block ID (required for research stage)
        
        Returns:
            Next available citation ID
        """
        if stage == "planning":
            return self.generate_plan_citation_id()
        return self.generate_research_citation_id(block_id)
    
    def citation_exists(self, citation_id: str) -> bool:
        """
        Citation ID 존재 여부 확인.
        
        Args:
            citation_id: Citation ID to check
        
        Returns:
            True if citation exists, False otherwise
        """
        return citation_id in self.citations
    
    # ========== ref_number 매핑 시스템 ==========
    
    def _get_citation_dedup_key(self, citation: Citation, source: Optional[Source] = None) -> str:
        """
        Citation 중복 제거를 위한 고유 키 생성.
        
        Paper citation의 경우 title + first author로 중복 제거.
        다른 citation 타입은 각각 고유한 ref_number를 받음.
        
        Args:
            citation: Citation 객체
            source: Source 객체 (optional)
        
        Returns:
            Unique string key for deduplication
        """
        if source and source.source_type == 'journal' and source.title:
            # Paper citation: title + first author로 중복 제거
            title = source.title.lower().strip()
            authors = source.authors[0].lower().strip() if source.authors else ""
            if title:
                return f"paper:{title}|{authors}"
        
        # 다른 타입: 각 citation은 고유한 ref_number
        return f"unique:{citation.citation_id}"
    
    def _extract_citation_sort_key(self, citation_id: str) -> Tuple[int, int, int]:
        """
        Citation ID에서 정렬 키 추출.
        
        Args:
            citation_id: Citation ID (e.g., "PLAN-01", "CIT-1-02")
        
        Returns:
            Tuple for sorting (stage, block_num, seq_num)
        """
        try:
            if citation_id.startswith("PLAN-"):
                # PLAN-XX 형식: 처음에 배치
                num = int(citation_id.replace("PLAN-", ""))
                return (0, 0, num)
            elif citation_id.startswith("CIT-"):
                # CIT-X-XX 형식
                parts = citation_id.replace("CIT-", "").split("-")
                if len(parts) == 2:
                    return (1, int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            pass
        return (999, 999, 999)
    
    def build_ref_number_map(self) -> Dict[str, int]:
        """
        citation_id → ref_number 매핑 구축 (중복 제거 포함).
        
        Returns:
            Dictionary mapping citation_id to reference number (1-based)
        """
        if not self.citations:
            self._ref_number_map = {}
            return self._ref_number_map
        
        # 모든 citation ID를 숫자 부분으로 정렬
        sorted_citation_ids = sorted(self.citations.keys(), key=self._extract_citation_sort_key)
        
        # 중복 제거 키와 할당된 ref_number 추적
        seen_keys: Dict[str, int] = {}
        ref_idx = 0
        ref_map: Dict[str, int] = {}
        
        for citation_id in sorted_citation_ids:
            citation = self.citations.get(citation_id)
            if not citation:
                continue
            
            source = self.sources.get(citation.source_id)
            dedup_key = self._get_citation_dedup_key(citation, source)
            
            if dedup_key in seen_keys:
                # 중복: 기존 ref_number 사용
                ref_map[citation_id] = seen_keys[dedup_key]
            else:
                # 새로운 고유 citation
                ref_idx += 1
                seen_keys[dedup_key] = ref_idx
                ref_map[citation_id] = ref_idx
        
        self._ref_number_map = ref_map
        return ref_map
    
    def get_ref_number(self, citation_id: str) -> int:
        """
        Citation ID에 대한 ref_number 반환.
        매핑이 아직 구축되지 않았다면 먼저 구축.
        
        Args:
            citation_id: Citation ID
        
        Returns:
            Reference number (1-based), or 0 if not found
        """
        if not self._ref_number_map:
            self.build_ref_number_map()
        return self._ref_number_map.get(citation_id, 0)
    
    def get_ref_number_map(self) -> Dict[str, int]:
        """
        전체 ref_number 매핑 반환.
        매핑이 아직 구축되지 않았다면 먼저 구축.
        
        Returns:
            Dictionary mapping citation_id to reference number
        """
        if not self._ref_number_map:
            self.build_ref_number_map()
        return self._ref_number_map.copy()
    
    # ========== Thread-safe async 메서드 (병렬 실행 지원) ==========
    
    async def generate_plan_citation_id_async(self) -> str:
        """
        Thread-safe async version of generate_plan_citation_id.
        
        Returns:
            Citation ID in PLAN-XX format
        """
        async with self._lock:
            return self.generate_plan_citation_id()
    
    async def generate_research_citation_id_async(self, block_id: str = "") -> str:
        """
        Thread-safe async version of generate_research_citation_id.
        
        Args:
            block_id: Block ID (e.g., "block_3")
        
        Returns:
            Citation ID in CIT-X-XX format
        """
        async with self._lock:
            return self.generate_research_citation_id(block_id)
    
    async def get_next_citation_id_async(self, stage: str = "research", block_id: str = "") -> str:
        """
        Thread-safe async version of get_next_citation_id.
        
        Args:
            stage: "planning" or "research"
            block_id: Block ID (required for research stage)
        
        Returns:
            Next available citation ID
        """
        async with self._lock:
            return self.get_next_citation_id(stage, block_id)
    
    async def add_citation_async(
        self,
        source: Source,
        style: Optional[CitationStyle] = None,
        page_number: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Thread-safe async version of add_citation.
        
        Args:
            source: 출처 정보
            style: 인용 스타일 (기본값 사용)
            page_number: 페이지 번호
            context: 인용 컨텍스트
        
        Returns:
            str: 인용 ID
        """
        async with self._lock:
            return self.add_citation(source, style, page_number, context)
