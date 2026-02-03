"""
Document Organizer Agent (완전 자동형 SparkleForge)

문서 구조 분석, 자동 분류 및 정리, 중복 제거, 인덱스 생성, 문서 포맷 통일 기능 제공.
"""

import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocumentInfo:
    """문서 정보."""
    path: Path
    content: str
    file_type: str
    size: int
    hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class DocumentIndex:
    """문서 인덱스."""
    root_path: Path
    documents: Dict[str, DocumentInfo]
    categories: Dict[str, List[str]]
    duplicates: Dict[str, List[str]]
    index_file: Optional[Path] = None


class DocumentOrganizerAgent:
    """문서 정리 자동화 에이전트."""
    
    SUPPORTED_TYPES = {
        '.md': 'markdown',
        '.txt': 'text',
        '.pdf': 'pdf',
        '.docx': 'word',
        '.html': 'html',
        '.rst': 'restructuredtext',
        '.org': 'org-mode'
    }
    
    CATEGORY_KEYWORDS = {
        'readme': ['readme', 'getting started', 'introduction'],
        'docs': ['documentation', 'docs', 'guide', 'tutorial'],
        'api': ['api', 'endpoint', 'reference', 'interface'],
        'config': ['config', 'configuration', 'settings', '.env'],
        'test': ['test', 'spec', 'fixture', 'mock'],
        'script': ['script', 'tool', 'utility', 'cli'],
        'changelog': ['changelog', 'history', 'release notes'],
        'license': ['license', 'copyright', 'legal']
    }
    
    def __init__(self, root_path: Optional[Path] = None):
        """
        초기화.
        
        Args:
            root_path: 문서 루트 경로 (None이면 현재 디렉토리)
        """
        self.root_path = root_path or Path.cwd()
        self.root_path = self.root_path.resolve()
        self.index: Optional[DocumentIndex] = None
    
    async def analyze_documents(self, include_patterns: Optional[List[str]] = None,
                                exclude_patterns: Optional[List[str]] = None) -> DocumentIndex:
        """
        문서 구조 분석.
        
        Args:
            include_patterns: 포함할 파일 패턴
            exclude_patterns: 제외할 파일/디렉토리 패턴
        
        Returns:
            DocumentIndex
        """
        logger.info(f"Analyzing documents in: {self.root_path}")
        
        if include_patterns is None:
            include_patterns = ['*.md', '*.txt', '*.rst', '*.org']
        
        if exclude_patterns is None:
            exclude_patterns = [
                '.git', 'node_modules', '.venv', 'venv', '__pycache__',
                '.pytest_cache', '.mypy_cache', 'build', 'dist'
            ]
        
        documents: Dict[str, DocumentInfo] = {}
        categories: Dict[str, List[str]] = defaultdict(list)
        
        # 문서 파일 탐색
        for pattern in include_patterns:
            for file_path in self.root_path.rglob(pattern):
                # 제외 패턴 확인
                if any(exclude in str(file_path) for exclude in exclude_patterns):
                    continue
                
                try:
                    doc_info = await self._analyze_document(file_path)
                    if doc_info:
                        rel_path = str(file_path.relative_to(self.root_path))
                        documents[rel_path] = doc_info
                        
                        # 카테고리 분류
                        category = self._classify_document(doc_info)
                        doc_info.category = category
                        categories[category].append(rel_path)
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # 중복 검사
        duplicates = await self._find_duplicates(documents)
        
        self.index = DocumentIndex(
            root_path=self.root_path,
            documents=documents,
            categories=dict(categories),
            duplicates=duplicates
        )
        
        logger.info(f"Document analysis complete: {len(documents)} documents, {len(categories)} categories")
        return self.index
    
    async def _analyze_document(self, file_path: Path) -> Optional[DocumentInfo]:
        """문서 분석."""
        try:
            if file_path.suffix.lower() not in self.SUPPORTED_TYPES:
                return None
            
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            file_type = self.SUPPORTED_TYPES[file_path.suffix.lower()]
            size = len(content)
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # 메타데이터 추출
            metadata = await self._extract_metadata(file_path, content, file_type)
            
            # 태그 추출
            tags = self._extract_tags(content, file_type)
            
            return DocumentInfo(
                path=file_path,
                content=content,
                file_type=file_type,
                size=size,
                hash=content_hash,
                metadata=metadata,
                tags=tags
            )
        except Exception as e:
            logger.error(f"Failed to analyze document {file_path}: {e}")
            return None
    
    async def _extract_metadata(self, file_path: Path, content: str, file_type: str) -> Dict[str, Any]:
        """메타데이터 추출."""
        metadata = {
            'file_name': file_path.name,
            'file_type': file_type,
            'size': len(content),
            'lines': len(content.split('\n')),
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        # Markdown 프론트매터 추출
        if file_type == 'markdown':
            frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if frontmatter_match:
                try:
                    import yaml
                    frontmatter = yaml.safe_load(frontmatter_match.group(1))
                    metadata.update(frontmatter)
                except Exception:
                    pass
        
        # 제목 추출
        if file_type == 'markdown':
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if title_match:
                metadata['title'] = title_match.group(1).strip()
        
        return metadata
    
    def _extract_tags(self, content: str, file_type: str) -> List[str]:
        """태그 추출."""
        tags = []
        content_lower = content.lower()
        
        # 카테고리 키워드 기반 태그
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(category)
        
        # 기술 스택 태그
        tech_keywords = {
            'python': ['python', 'pip', 'pytest', 'django', 'flask'],
            'javascript': ['javascript', 'node', 'npm', 'react', 'vue'],
            'docker': ['docker', 'dockerfile', 'container'],
            'kubernetes': ['kubernetes', 'k8s', 'helm'],
            'aws': ['aws', 's3', 'lambda', 'ec2'],
            'azure': ['azure', 'blob', 'functions'],
            'gcp': ['gcp', 'gcs', 'cloud functions']
        }
        
        for tech, keywords in tech_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(tech)
        
        return list(set(tags))
    
    def _classify_document(self, doc_info: DocumentInfo) -> str:
        """문서 분류."""
        path_lower = str(doc_info.path).lower()
        content_lower = doc_info.content.lower()
        
        # 경로 기반 분류
        if 'readme' in path_lower:
            return 'readme'
        elif 'docs' in path_lower or 'documentation' in path_lower:
            return 'docs'
        elif 'api' in path_lower:
            return 'api'
        elif 'test' in path_lower or 'spec' in path_lower:
            return 'test'
        elif 'config' in path_lower or 'settings' in path_lower:
            return 'config'
        elif 'changelog' in path_lower or 'history' in path_lower:
            return 'changelog'
        elif 'license' in path_lower:
            return 'license'
        
        # 내용 기반 분류
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(keyword in content_lower for keyword in keywords):
                return category
        
        return 'other'
    
    async def _find_duplicates(self, documents: Dict[str, DocumentInfo]) -> Dict[str, List[str]]:
        """중복 문서 찾기."""
        hash_to_files: Dict[str, List[str]] = defaultdict(list)
        
        for rel_path, doc_info in documents.items():
            hash_to_files[doc_info.hash].append(rel_path)
        
        # 해시가 2개 이상인 경우만 중복으로 간주
        duplicates = {
            hash_val: files
            for hash_val, files in hash_to_files.items()
            if len(files) > 1
        }
        
        return duplicates
    
    async def organize_documents(self, target_structure: Optional[Dict[str, str]] = None,
                                 dry_run: bool = True) -> Dict[str, Any]:
        """
        문서 자동 정리.
        
        Args:
            target_structure: 목표 디렉토리 구조 (카테고리 -> 디렉토리 매핑)
            dry_run: 실제 이동 없이 시뮬레이션만 수행
        
        Returns:
            정리 결과
        """
        if not self.index:
            await self.analyze_documents()
        
        if target_structure is None:
            target_structure = {
                'readme': 'docs/readme',
                'docs': 'docs',
                'api': 'docs/api',
                'test': 'docs/test',
                'config': 'docs/config',
                'changelog': 'docs/changelog',
                'license': 'docs/license',
                'other': 'docs/other'
            }
        
        moves: List[Dict[str, str]] = []
        creates: List[str] = []
        
        # 디렉토리 생성
        for category_dir in set(target_structure.values()):
            target_path = self.root_path / category_dir
            if not target_path.exists():
                creates.append(category_dir)
                if not dry_run:
                    target_path.mkdir(parents=True, exist_ok=True)
        
        # 문서 이동
        for rel_path, doc_info in self.index.documents.items():
            category = doc_info.category or 'other'
            target_dir = target_structure.get(category, 'docs/other')
            
            source_path = self.root_path / rel_path
            target_path = self.root_path / target_dir / source_path.name
            
            # 이미 올바른 위치에 있으면 스킵
            if source_path.parent == target_path.parent:
                continue
            
            moves.append({
                'from': rel_path,
                'to': str(target_path.relative_to(self.root_path)),
                'category': category
            })
            
            if not dry_run:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.rename(target_path)
        
        return {
            'dry_run': dry_run,
            'directories_created': creates,
            'files_moved': moves,
            'total_moves': len(moves)
        }
    
    async def remove_duplicates(self, keep_strategy: str = 'newest', dry_run: bool = True) -> Dict[str, Any]:
        """
        중복 문서 제거.
        
        Args:
            keep_strategy: 유지 전략 ('newest', 'oldest', 'shortest_path')
            dry_run: 실제 삭제 없이 시뮬레이션만 수행
        
        Returns:
            제거 결과
        """
        if not self.index:
            await self.analyze_documents()
        
        to_remove: List[str] = []
        
        for hash_val, file_paths in self.index.duplicates.items():
            if keep_strategy == 'newest':
                # 가장 최근 수정된 파일 유지
                file_times = [
                    (fp, (self.root_path / fp).stat().st_mtime)
                    for fp in file_paths
                ]
                file_times.sort(key=lambda x: x[1], reverse=True)
                keep_file = file_times[0][0]
            elif keep_strategy == 'oldest':
                # 가장 오래된 파일 유지
                file_times = [
                    (fp, (self.root_path / fp).stat().st_mtime)
                    for fp in file_paths
                ]
                file_times.sort(key=lambda x: x[1])
                keep_file = file_times[0][0]
            elif keep_strategy == 'shortest_path':
                # 가장 짧은 경로 유지
                keep_file = min(file_paths, key=len)
            else:
                keep_file = file_paths[0]
            
            # 유지할 파일 제외하고 나머지 삭제
            for file_path in file_paths:
                if file_path != keep_file:
                    to_remove.append(file_path)
        
        if not dry_run:
            for rel_path in to_remove:
                file_path = self.root_path / rel_path
                if file_path.exists():
                    file_path.unlink()
        
        return {
            'dry_run': dry_run,
            'files_removed': to_remove,
            'total_removed': len(to_remove),
            'duplicate_groups': len(self.index.duplicates)
        }
    
    async def generate_index(self, output_path: Optional[Path] = None,
                            format: str = 'markdown') -> Path:
        """
        문서 인덱스 생성.
        
        Args:
            output_path: 출력 파일 경로 (None이면 자동 생성)
            format: 인덱스 형식 ('markdown', 'json')
        
        Returns:
            생성된 인덱스 파일 경로
        """
        if not self.index:
            await self.analyze_documents()
        
        if output_path is None:
            output_path = self.root_path / 'DOCUMENT_INDEX.md'
        
        if format == 'markdown':
            content = self._generate_markdown_index()
            output_path.write_text(content, encoding='utf-8')
        elif format == 'json':
            content = json.dumps({
                'root_path': str(self.index.root_path),
                'documents': {
                    path: {
                        'file_type': doc.file_type,
                        'size': doc.size,
                        'category': doc.category,
                        'tags': doc.tags,
                        'metadata': doc.metadata
                    }
                    for path, doc in self.index.documents.items()
                },
                'categories': self.index.categories,
                'duplicates': self.index.duplicates
            }, indent=2, ensure_ascii=False)
            output_path.write_text(content, encoding='utf-8')
        
        self.index.index_file = output_path
        logger.info(f"Generated index: {output_path}")
        return output_path
    
    def _generate_markdown_index(self) -> str:
        """Markdown 인덱스 생성."""
        lines = [
            '# Document Index',
            '',
            f'Generated: {datetime.now().isoformat()}',
            '',
            f'Total documents: {len(self.index.documents)}',
            f'Categories: {len(self.index.categories)}',
            f'Duplicate groups: {len(self.index.duplicates)}',
            '',
            '## Categories',
            ''
        ]
        
        # 카테고리별 문서 목록
        for category, file_paths in sorted(self.index.categories.items()):
            lines.append(f'### {category.title()} ({len(file_paths)} documents)')
            lines.append('')
            for file_path in sorted(file_paths):
                doc_info = self.index.documents[file_path]
                title = doc_info.metadata.get('title', file_path)
                lines.append(f'- [{title}]({file_path})')
            lines.append('')
        
        # 중복 문서 섹션
        if self.index.duplicates:
            lines.extend([
                '## Duplicate Documents',
                '',
                'The following documents have identical content:',
                ''
            ])
            
            for hash_val, file_paths in self.index.duplicates.items():
                lines.append(f'### Group: {hash_val[:8]}')
                for file_path in file_paths:
                    lines.append(f'- {file_path}')
                lines.append('')
        
        return '\n'.join(lines)
    
    async def unify_format(self, target_format: str = 'markdown', dry_run: bool = True) -> Dict[str, Any]:
        """
        문서 포맷 통일.
        
        Args:
            target_format: 목표 포맷 ('markdown', 'html', 'rst')
            dry_run: 실제 변환 없이 시뮬레이션만 수행
        
        Returns:
            변환 결과
        """
        if not self.index:
            await self.analyze_documents()
        
        conversions: List[Dict[str, str]] = []
        
        for rel_path, doc_info in self.index.documents.items():
            if doc_info.file_type == target_format:
                continue
            
            source_path = self.root_path / rel_path
            target_path = source_path.with_suffix(f'.{target_format}')
            
            conversions.append({
                'from': rel_path,
                'to': str(target_path.relative_to(self.root_path)),
                'from_format': doc_info.file_type,
                'to_format': target_format
            })
            
            if not dry_run:
                # 실제 변환은 LLM 또는 변환 라이브러리 사용
                converted_content = await self._convert_format(
                    doc_info.content,
                    doc_info.file_type,
                    target_format
                )
                target_path.write_text(converted_content, encoding='utf-8')
                source_path.unlink()  # 원본 삭제
        
        return {
            'dry_run': dry_run,
            'conversions': conversions,
            'total_conversions': len(conversions)
        }
    
    async def _convert_format(self, content: str, from_format: str, to_format: str) -> str:
        """포맷 변환 (LLM 사용)."""
        from src.core.mcp_integration import get_mcp_hub
        
        mcp_hub = get_mcp_hub()
        
        prompt = f"""Convert the following document from {from_format} to {to_format} format.

Preserve all content, structure, and formatting as much as possible.

Original document ({from_format}):
```
{content}
```

Converted document ({to_format}):"""

        response = await mcp_hub.call_llm_async(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )
        
        converted = response.get("content", "").strip()
        
        # 코드 블록 추출
        code_match = re.search(r'```(?:\w+)?\n(.*?)```', converted, re.DOTALL)
        if code_match:
            converted = code_match.group(1).strip()
        
        return converted

