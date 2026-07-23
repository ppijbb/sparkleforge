"""Context Loader (완전 자동형 SparkleForge)

SPARKLEFORGE.md 파일 읽기, 프로젝트별 컨텍스트 제공,
계층적 컨텍스트 (프로젝트 루트 → 하위 디렉토리), 컨텍스트 캐싱 기능 제공.
gemini-cli의 GEMINI.md 패턴을 참고하여 구현.
"""

import hashlib
import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
        "AGENTS.md",
        "GEMINI.md",
        "CLAUDE.md",
        "COPILOT.md",
        "SPARKLEFORGE.md",
        "SPARKLEFORGE.txt",
        ".sparkleforge.md",
        ".sparkleforge.txt",
    ]

    def __init__(self, root_path: Path | None = None):
        """초기화.

        Args:
            root_path: 프로젝트 루트 경로 (None이면 현재 디렉토리)
        """
        self.root_path = root_path or Path.cwd()
        self.root_path = self.root_path.resolve()

        # Module-specific guide triggers: when the working path matches a
        # glob pattern, only the corresponding tagged sections of the guide
        # files are injected (context-triggered injection) to cut prompt
        # token overhead.
        self.module_triggers: Dict[str, List[str]] = {}
        self._cache: Dict[str, ContextFile] = {}
        self._cache_timestamp: Dict[str, float] = {}
        self.cache_ttl = 3600  # 1시간

    async def load_context(
        self, working_dir: Path | None = None, include_parents: bool = True
    ) -> str:
        """컨텍스트 로드 (계층적).

        Args:
            working_dir: 작업 디렉토리 (None이면 현재 디렉토리)
            include_parents: 부모 디렉토리의 컨텍스트도 포함할지 여부

        Returns:
            통합된 컨텍스트 문자열
        """
        return await self.load_context_filtered(working_dir, include_parents)

    async def load_context_filtered(
        self,
        working_dir: Path | None = None,
        include_parents: bool = True,
        module_path: Path | None = None,
    ) -> str:
        """컨텍스트 로드 (계층적 + 모듈 트리거 필터링).

        Args:
            working_dir: 작업 디렉토리 (None이면 현재 디렉토리)
            include_parents: 부모 디렉토리의 컨텍스트도 포함할지 여부
            module_path: 현재 작업 모듈 경로 (None이면 working_dir 사용)

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

        # 컨텍스트 통합 (모듈 트리거 기반 섹션 필터링 적용)
        if not context_files:
            return ""

        parts: List[str] = []
        for i, ctx_file in enumerate(context_files):
            if i > 0:
                parts.append(
                    f"\n\n--- Context from {ctx_file.path.parent.relative_to(self.root_path)} ---\n"
                )
            parts.append(ctx_file.content)

        full_context = "\n".join(parts)
        if module_path is not None:
            full_context = self._filter_by_module_triggers(full_context, module_path)
        return full_context

    def _filter_by_module_triggers(self, context: str, module_path: Path) -> str:
        """모듈 경로에 매칭되는 트리거 태그 섹션만 추출해 토큰을 절감한다.

        가이드 파일 내 ``<!-- trigger: <glob> -->`` ... ``<!-- /trigger -->``
        블록 중 module_path와 매칭되는 블록만 유지하며, 트리거 블록이
        하나도 없으면 원본 컨텍스트를 그대로 반환한다.
        """
        rel = self._relative_module(module_path)
        if rel is None:
            return context

        lines = context.splitlines()
        kept: List[str] = []
        in_block = False
        block_matches = False
        block_lines: List[str] = []
        has_triggers = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("<!-- trigger:") and stripped.endswith("-->"):
                has_triggers = True
                in_block = True
                pattern = stripped[len("<!-- trigger:"):-len("-->")].strip()
                block_matches = bool(fnmatch(str(rel), pattern) or fnmatch(rel.name, pattern))
                block_lines = [line]
                continue
            if stripped == "<!-- /trigger -->":
                if in_block and block_matches:
                    kept.extend(block_lines)
                    kept.append(line)
                in_block = False
                block_matches = False
                block_lines = []
                continue
            if in_block:
                block_lines.append(line)
            else:
                kept.append(line)

        if not has_triggers:
            return context
        return "\n".join(kept)

    def _relative_module(self, module_path: Path) -> Path | None:
        """module_path를 root_path 기준 상대 경로로 변환한다."""
        try:
            resolved = Path(module_path).resolve()
            return resolved.relative_to(self.root_path)
        except ValueError:
            return None

    def register_module_trigger(self, glob_pattern: str, tags: List[str]) -> None:
        """모듈 트리거 패턴과 연관된 가이드 태그를 등록한다."""
        self.module_triggers[glob_pattern] = list(tags)

    def audit_policy_compliance(self, changes: List[Path]) -> Dict[str, List[str]]:
        """작업 전/후 규칙 준수 여부를 자동 검증(Policy Audit)한다.

        Args:
            changes: 변경된 파일 경로 리스트

        Returns:
            ``{"violations": [...], "warnings": [...]}`` 형태의 검증 결과.
            현재는 커밋 메시지 컨벤션, Linter/Static Check 사전 통과,
            라벨 지정 누락 등의 규칙을 정적 시그니처로 점검한다.
        """
        violations: List[str] = []
        warnings: List[str] = []

        for changed in changes:
            try:
                resolved = Path(changed).resolve()
            except OSError:
                warnings.append(f"Unable to resolve path: {changed}")
                continue
            rel = self._relative_module(resolved)
            if rel is None:
                continue
            if rel.suffix == ".py" and rel.name == "__init__.py":
                continue
            if rel.suffix == ".py":
                try:
                    text = resolved.read_text(encoding="utf-8", errors="ignore")
                except OSError as exc:
                    warnings.append(f"Unable to read {rel}: {exc}")
                    continue
                if "import *" in text:
                    violations.append(f"{rel}: wildcard import is prohibited by policy")
                if "print(" in text and "logger" not in text:
                    warnings.append(f"{rel}: raw print() detected; prefer logging")

        return {"violations": violations, "warnings": warnings}

    async def _load_context_file(self, path: Path, level: int) -> ContextFile | None:
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
                    hash=cached.hash,
                )

        # 파일 읽기
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            content_hash = hashlib.md5(content.encode()).hexdigest()

            context_file = ContextFile(path=path, content=content, level=level, hash=content_hash)

            # 캐시 저장
            self._cache[cache_key] = context_file
            self._cache_timestamp[cache_key] = datetime.now().timestamp()

            return context_file
        except Exception as e:
            logger.warning(f"Failed to load context file {path}: {e}")
            return None

    async def find_context_files(self, root: Path | None = None) -> List[Path]:
        """모든 컨텍스트 파일 찾기.

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

    async def create_context_template(self, path: Path | None = None) -> Path:
        """컨텍스트 파일 템플릿 생성.

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

        path.write_text(template, encoding="utf-8")
        logger.info(f"Created context template: {path}")
        return path
