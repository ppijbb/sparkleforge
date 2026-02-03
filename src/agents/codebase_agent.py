"""
Codebase Agent (완전 자동형 SparkleForge)

코드베이스 전체 구조 분석, 파일 간 의존성 추적, 패턴 및 컨벤션 이해,
대규모 리팩토링 지원, 코드 생성 및 편집 기능 제공.

claude-code의 code-explorer 패턴을 참고하여 구현.
"""

import asyncio
import logging
import ast
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class CodeFile:
    """코드 파일 정보."""
    path: Path
    content: str
    language: str
    imports: List[str]
    exports: List[str]
    classes: List[str]
    functions: List[str]
    dependencies: Set[str]


@dataclass
class CodebaseStructure:
    """코드베이스 구조 정보."""
    root_path: Path
    files: Dict[str, CodeFile]
    dependencies: Dict[str, Set[str]]
    entry_points: List[str]
    patterns: Dict[str, Any]
    conventions: Dict[str, Any]


class CodebaseAgent:
    """코드베이스 이해 및 편집 에이전트."""
    
    SUPPORTED_LANGUAGES = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.c': 'c',
    }
    
    def __init__(self, codebase_path: Optional[Path] = None):
        """
        초기화.
        
        Args:
            codebase_path: 코드베이스 루트 경로 (None이면 현재 디렉토리)
        """
        self.codebase_path = codebase_path or Path.cwd()
        self.codebase_path = self.codebase_path.resolve()
        self.structure: Optional[CodebaseStructure] = None
    
    async def analyze_codebase(self, include_patterns: Optional[List[str]] = None, 
                              exclude_patterns: Optional[List[str]] = None) -> CodebaseStructure:
        """
        코드베이스 전체 구조 분석.
        
        Args:
            include_patterns: 포함할 파일 패턴 (예: ['*.py', '*.js'])
            exclude_patterns: 제외할 파일/디렉토리 패턴 (예: ['__pycache__', 'node_modules'])
        
        Returns:
            CodebaseStructure
        """
        logger.info(f"Analyzing codebase: {self.codebase_path}")
        
        if include_patterns is None:
            include_patterns = ['*.py', '*.js', '*.ts', '*.jsx', '*.tsx']
        
        if exclude_patterns is None:
            exclude_patterns = [
                '__pycache__', 'node_modules', '.git', '.venv', 'venv',
                '*.pyc', '*.pyo', '.pytest_cache', '.mypy_cache'
            ]
        
        files: Dict[str, CodeFile] = {}
        dependencies: Dict[str, Set[str]] = defaultdict(set)
        entry_points: List[str] = []
        
        # 파일 탐색
        for pattern in include_patterns:
            for file_path in self.codebase_path.rglob(pattern):
                # 제외 패턴 확인
                if any(exclude in str(file_path) for exclude in exclude_patterns):
                    continue
                
                try:
                    code_file = await self._analyze_file(file_path)
                    if code_file:
                        rel_path = str(file_path.relative_to(self.codebase_path))
                        files[rel_path] = code_file
                        
                        # 의존성 추적
                        for dep in code_file.dependencies:
                            dependencies[rel_path].add(dep)
                        
                        # 진입점 확인
                        if self._is_entry_point(file_path, code_file):
                            entry_points.append(rel_path)
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # 패턴 및 컨벤션 분석
        patterns = await self._analyze_patterns(files)
        conventions = await self._analyze_conventions(files)
        
        self.structure = CodebaseStructure(
            root_path=self.codebase_path,
            files=files,
            dependencies=dependencies,
            entry_points=entry_points,
            patterns=patterns,
            conventions=conventions
        )
        
        logger.info(f"Codebase analysis complete: {len(files)} files, {len(entry_points)} entry points")
        return self.structure
    
    async def _analyze_file(self, file_path: Path) -> Optional[CodeFile]:
        """파일 분석."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            language = self._detect_language(file_path)
            
            if language == 'python':
                return await self._analyze_python_file(file_path, content)
            elif language in ['javascript', 'typescript']:
                return await self._analyze_js_file(file_path, content)
            else:
                # 기본 분석
                return CodeFile(
                    path=file_path,
                    content=content,
                    language=language,
                    imports=[],
                    exports=[],
                    classes=[],
                    functions=[],
                    dependencies=set()
                )
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            return None
    
    def _detect_language(self, file_path: Path) -> str:
        """파일 언어 감지."""
        suffix = file_path.suffix.lower()
        return self.SUPPORTED_LANGUAGES.get(suffix, 'unknown')
    
    async def _analyze_python_file(self, file_path: Path, content: str) -> CodeFile:
        """Python 파일 분석."""
        try:
            tree = ast.parse(content, filename=str(file_path))
            
            imports: List[str] = []
            classes: List[str] = []
            functions: List[str] = []
            dependencies: Set[str] = set()
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                            dependencies.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                            dependencies.add(node.module.split('.')[0])
                
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            
            return CodeFile(
                path=file_path,
                content=content,
                language='python',
                imports=imports,
                exports=[],  # Python은 명시적 export 없음
                classes=classes,
                functions=functions,
                dependencies=dependencies
            )
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return CodeFile(
                path=file_path,
                content=content,
                language='python',
                imports=[],
                exports=[],
                classes=[],
                functions=[],
                dependencies=set()
            )
    
    async def _analyze_js_file(self, file_path: Path, content: str) -> CodeFile:
        """JavaScript/TypeScript 파일 분석."""
        imports: List[str] = []
        exports: List[str] = []
        classes: List[str] = []
        functions: List[str] = []
        dependencies: Set[str] = set()
        
        # import 문 파싱
        import_pattern = r'import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)(?:\s*,\s*(?:\{[^}]*\}|\*\s+as\s+\w+|\w+))*\s+from\s+)?["\']([^"\']+)["\']'
        for match in re.finditer(import_pattern, content):
            module = match.group(1)
            imports.append(module)
            dependencies.add(module.split('/')[0])
        
        # require 문 파싱
        require_pattern = r'require\(["\']([^"\']+)["\']\)'
        for match in re.finditer(require_pattern, content):
            module = match.group(1)
            imports.append(module)
            dependencies.add(module.split('/')[0])
        
        # export 문 파싱
        export_pattern = r'export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)'
        for match in re.finditer(export_pattern, content):
            exports.append(match.group(1))
        
        # class 선언 파싱
        class_pattern = r'(?:export\s+)?(?:default\s+)?class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            classes.append(match.group(1))
        
        # function 선언 파싱
        function_pattern = r'(?:export\s+)?(?:default\s+)?function\s+(\w+)'
        for match in re.finditer(function_pattern, content):
            functions.append(match.group(1))
        
        return CodeFile(
            path=file_path,
            content=content,
            language='javascript' if file_path.suffix in ['.js', '.jsx'] else 'typescript',
            imports=imports,
            exports=exports,
            classes=classes,
            functions=functions,
            dependencies=dependencies
        )
    
    def _is_entry_point(self, file_path: Path, code_file: CodeFile) -> bool:
        """진입점인지 확인."""
        # main.py, index.js, app.py 등
        entry_point_names = ['main', 'index', 'app', 'server', 'cli']
        file_stem = file_path.stem.lower()
        
        if file_stem in entry_point_names:
            return True
        
        # __main__ 블록이 있는 Python 파일
        if code_file.language == 'python' and '__main__' in code_file.content:
            return True
        
        return False
    
    async def _analyze_patterns(self, files: Dict[str, CodeFile]) -> Dict[str, Any]:
        """패턴 분석."""
        patterns = {
            'architecture': {},
            'design_patterns': [],
            'common_structures': {}
        }
        
        # 아키텍처 레이어 분석
        layer_patterns = {
            'api': ['api', 'endpoint', 'route', 'controller'],
            'service': ['service', 'business', 'logic'],
            'model': ['model', 'entity', 'schema'],
            'repository': ['repository', 'dao', 'data'],
            'util': ['util', 'helper', 'common']
        }
        
        for file_path, code_file in files.items():
            path_lower = file_path.lower()
            for layer, keywords in layer_patterns.items():
                if any(keyword in path_lower for keyword in keywords):
                    if layer not in patterns['architecture']:
                        patterns['architecture'][layer] = []
                    patterns['architecture'][layer].append(file_path)
        
        return patterns
    
    async def _analyze_conventions(self, files: Dict[str, CodeFile]) -> Dict[str, Any]:
        """컨벤션 분석."""
        conventions = {
            'naming': {},
            'structure': {},
            'imports': {}
        }
        
        # 네이밍 컨벤션 분석
        class_names = []
        function_names = []
        
        for code_file in files.values():
            class_names.extend(code_file.classes)
            function_names.extend(code_file.functions)
        
        # 클래스 네이밍 (PascalCase vs snake_case)
        pascal_case = sum(1 for name in class_names if re.match(r'^[A-Z]', name))
        snake_case = sum(1 for name in class_names if re.match(r'^[a-z]', name))
        
        conventions['naming'] = {
            'class_style': 'PascalCase' if pascal_case > snake_case else 'snake_case',
            'function_style': 'snake_case' if any('_' in name for name in function_names) else 'camelCase'
        }
        
        return conventions
    
    async def trace_execution_path(self, entry_point: str, target: str) -> List[str]:
        """
        실행 경로 추적 (진입점에서 목표까지).
        
        Args:
            entry_point: 진입점 파일 경로
            target: 목표 함수/클래스 이름
        
        Returns:
            실행 경로 (파일 경로 리스트)
        """
        if not self.structure:
            await self.analyze_codebase()
        
        visited: Set[str] = set()
        path: List[str] = []
        
        def dfs(current_file: str) -> bool:
            if current_file in visited:
                return False
            
            visited.add(current_file)
            path.append(current_file)
            
            if current_file not in self.structure.files:
                path.pop()
                return False
            
            code_file = self.structure.files[current_file]
            
            # 목표 찾기
            if target in code_file.classes or target in code_file.functions:
                return True
            
            # 의존성 따라가기
            for dep_file in self.structure.dependencies.get(current_file, []):
                # 의존성을 파일 경로로 변환
                dep_path = self._resolve_dependency(dep_file, current_file)
                if dep_path and dfs(dep_path):
                    return True
            
            path.pop()
            return False
        
        if dfs(entry_point):
            return path
        else:
            return []
    
    def _resolve_dependency(self, dependency: str, from_file: str) -> Optional[str]:
        """의존성을 파일 경로로 변환."""
        # 간단한 구현 (실제로는 더 복잡한 해석 필요)
        for file_path in self.structure.files.keys():
            if dependency in file_path or file_path.endswith(f"/{dependency}.py"):
                return file_path
        return None
    
    async def find_similar_code(self, pattern: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        유사한 코드 패턴 찾기.
        
        Args:
            pattern: 검색할 패턴 (정규식 또는 문자열)
            language: 언어 필터 (선택)
        
        Returns:
            매칭된 코드 위치 리스트
        """
        if not self.structure:
            await self.analyze_codebase()
        
        matches = []
        pattern_re = re.compile(pattern, re.IGNORECASE)
        
        for file_path, code_file in self.structure.files.items():
            if language and code_file.language != language:
                continue
            
            for line_num, line in enumerate(code_file.content.split('\n'), 1):
                if pattern_re.search(line):
                    matches.append({
                        'file': file_path,
                        'line': line_num,
                        'content': line.strip(),
                        'language': code_file.language
                    })
        
        return matches
    
    async def generate_code(self, description: str, file_path: str, 
                           context: Optional[Dict[str, Any]] = None) -> str:
        """
        코드 생성 (LLM 사용).
        
        Args:
            description: 생성할 코드 설명
            file_path: 생성할 파일 경로
            context: 추가 컨텍스트 (기존 코드, 패턴 등)
        
        Returns:
            생성된 코드
        """
        from src.core.mcp_integration import get_mcp_hub
        
        mcp_hub = get_mcp_hub()
        
        # 코드베이스 컨텍스트 준비
        codebase_context = ""
        if self.structure:
            codebase_context = f"""
Codebase Structure:
- Root: {self.structure.root_path}
- Files: {len(self.structure.files)}
- Entry points: {', '.join(self.structure.entry_points[:5])}
- Patterns: {json.dumps(self.structure.patterns, indent=2)[:500]}
- Conventions: {json.dumps(self.structure.conventions, indent=2)[:500]}
"""
        
        prompt = f"""Generate code based on the following description.

Description: {description}

Target file: {file_path}

{codebase_context}

{context.get('additional_context', '') if context else ''}

Follow the codebase conventions and patterns. Generate production-ready code."""

        response = await mcp_hub.call_llm_async(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        return response.get("content", "").strip()
    
    async def refactor_code(self, file_path: str, refactoring_type: str, 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        코드 리팩토링.
        
        Args:
            file_path: 리팩토링할 파일 경로
            refactoring_type: 리팩토링 타입 (extract_method, rename, etc.)
            parameters: 리팩토링 파라미터
        
        Returns:
            리팩토링 결과
        """
        if not self.structure:
            await self.analyze_codebase()
        
        full_path = self.codebase_path / file_path
        if not full_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }
        
        content = full_path.read_text(encoding='utf-8')
        
        # LLM을 사용한 리팩토링
        from src.core.mcp_integration import get_mcp_hub
        mcp_hub = get_mcp_hub()
        
        prompt = f"""Refactor the following code.

File: {file_path}
Refactoring type: {refactoring_type}
Parameters: {json.dumps(parameters, indent=2)}

Current code:
```{self.structure.files[file_path].language if file_path in self.structure.files else 'python'}
{content}
```

Apply the refactoring and return only the refactored code."""

        response = await mcp_hub.call_llm_async(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )
        
        refactored_code = response.get("content", "").strip()
        
        # 코드 블록 추출
        code_match = re.search(r'```(?:\w+)?\n(.*?)```', refactored_code, re.DOTALL)
        if code_match:
            refactored_code = code_match.group(1).strip()
        
        return {
            "success": True,
            "refactored_code": refactored_code,
            "file_path": file_path
        }

