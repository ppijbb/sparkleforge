"""
Context Loader (완전 자동형 SparkleForge)

SPARKLEFORGE.md 파일 읽기, 프로젝트별 컨텍스트 제공,
계층적 컨텍스트 (프로젝트 루트 → 하위 디렉토리), 컨텍스트 캐싱 기능 제공.
gemini-cli의 GEMINI.md 패턴을 참고하여 구현.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ContextFile:
    """컨텍스트 파일 정보."""
    path: Path
    content: str
    level: int  # 0 = 루트, 1 = 1단계 하위, etc.
    hash: str


class ContextLoader:
    """컨텍스트 파일 로더."""
    
    CONTEXT_FILENAMES = [
        'SPARKLEFORGE.md',
        'SPARKLEFORGE.txt',
        '.sparkleforge.md',
        '.sparkleforge.txt'
    ]
    
    def __init__(self, root_path: Optional[Path] = None):
        """
        초기화.
        
        Args:
            root_path: 프로젝트 루트 경로 (None이면 현재 디렉토리)
        """
        self.root_path = root_path or Path.cwd()
        self.root_path = self.root_path.resolve()
        
        self._cache: Dict[str, ContextFile] = {}
        self._cache_timestamp: Dict[str, float] = {}
        self.cache_ttl = 3600  # 1시간
    
    async def load_context(
        self,
        working_dir: Optional[Path] = None,
        include_parents: bool = True
    ) -> str:
        """
        컨텍스트 로드 (계층적).
        
        Args:
            working_dir: 작업 디렉토리 (None이면 현재 디렉토리)
            include_parents: 부모 디렉토리의 컨텍스트도 포함할지 여부
        
        Returns:
            통합된 컨텍스트 문자열
        """
        if working_dir is None:
            working_dir = Path.cwd()
        else:
            working_dir = Path(working_dir).resolve()
        
        context_files = []
        
        if include_parents:
            # 루트까지 모든 디렉토리에서 컨텍스트 파일 찾기
            current = working_dir
            level = 0
            
            while current != self.root_path.parent and level < 10:  # 최대 10단계
                for filename in self.CONTEXT_FILENAMES:
                    context_path = current / filename
                    if context_path.exists():
                        context_file = await self._load_context_file(context_path, level)
                        if context_file:
                            context_files.append(context_file)
                            break  # 한 디렉토리당 하나의 파일만
                
                if current == self.root_path:
                    break
                current = current.parent
                level += 1
        else:
            # 현재 디렉토리만
            for filename in self.CONTEXT_FILENAMES:
                context_path = working_dir / filename
                if context_path.exists():
                    context_file = await self._load_context_file(context_path, 0)
                    if context_file:
                        context_files.append(context_file)
                        break
        
        # 레벨 순서대로 정렬 (루트가 먼저)
        context_files.sort(key=lambda x: x.level)
        
        # 컨텍스트 통합
        if not context_files:
            return ""
        
        parts = []
        for i, ctx_file in enumerate(context_files):
            if i > 0:
                parts.append(f"\n\n--- Context from {ctx_file.path.parent.relative_to(self.root_path)} ---\n")
            parts.append(ctx_file.content)
        
        return "\n".join(parts)
    
    async def _load_context_file(self, path: Path, level: int) -> Optional[ContextFile]:
        """컨텍스트 파일 로드 (캐싱 포함)."""
        cache_key = str(path)
        
        # 캐시 확인
        if cache_key in self._cache:
            cache_time = self._cache_timestamp.get(cache_key, 0)
            file_time = path.stat().st_mtime
            
            if file_time <= cache_time:
                # 캐시 유효
                cached = self._cache[cache_key]
                return ContextFile(
                    path=cached.path,
                    content=cached.content,
                    level=level,
                    hash=cached.hash
                )
        
        # 파일 읽기
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            context_file = ContextFile(
                path=path,
                content=content,
                level=level,
                hash=content_hash
            )
            
            # 캐시 저장
            self._cache[cache_key] = context_file
            self._cache_timestamp[cache_key] = datetime.now().timestamp()
            
            return context_file
        except Exception as e:
            logger.warning(f"Failed to load context file {path}: {e}")
            return None
    
    async def find_context_files(self, root: Optional[Path] = None) -> List[Path]:
        """
        모든 컨텍스트 파일 찾기.
        
        Args:
            root: 검색 루트 (None이면 프로젝트 루트)
        
        Returns:
            컨텍스트 파일 경로 리스트
        """
        if root is None:
            root = self.root_path
        
        context_files = []
        
        for context_file in root.rglob("*"):
            if context_file.name in self.CONTEXT_FILENAMES:
                context_files.append(context_file)
        
        return context_files
    
    def clear_cache(self):
        """캐시 초기화."""
        self._cache.clear()
        self._cache_timestamp.clear()
        logger.info("Context cache cleared")
    
    async def create_context_template(self, path: Optional[Path] = None) -> Path:
        """
        컨텍스트 파일 템플릿 생성.
        
        Args:
            path: 생성할 파일 경로 (None이면 SPARKLEFORGE.md)
        
        Returns:
            생성된 파일 경로
        """
        if path is None:
            path = self.root_path / "SPARKLEFORGE.md"
        
        template = """# SparkleForge Context

This file provides context for SparkleForge operations in this project.

## Project Overview

Describe your project here.

## Key Conventions

- Code style: ...
- Architecture: ...
- Testing: ...

## Important Files

- `main.py`: Entry point
- `src/`: Source code

## Special Instructions

Any special instructions for SparkleForge when working on this project.

## Examples

Example queries or tasks that work well with this project.
"""
        
        path.write_text(template, encoding='utf-8')
        logger.info(f"Created context template: {path}")
        return path

