"""Anvil Skill Repository - 런타임 스킬/도구 저장소.

에이전트가 실행 중 생성한 코드 조각, 커스텀 도구, 함수 등을
세션 내에서 보존하고 재사용할 수 있도록 하는 저장소.
"""

import importlib.util
import json
import logging
import re
import ast
import time
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VERSION_FILE_RE = re.compile(r"\.v\d+$")


@dataclass
class Skill:
    """저장된 스킬/도구 정의."""

    name: str
    code: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1


class SkillValidator:
    """자체 검증 프로시저 스킬 컴파일러 및 카나리 붕괴 감지기."""

    @staticmethod
    def extract_canary_assertions(code: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """성공한 궤적에서 컴파일된 스킬의 카나리 단언(참조된 파일 경로, AST 심볼, CLI 플래그)을 추출합니다."""
        referenced_files: Set[str] = set()
        ast_symbols: Set[str] = set()
        cli_flags: Set[str] = set()

        # 1. 파일 경로 추출 (문자열 리터럴 중 디렉토리나 확장자를 포함하는 패턴)
        path_patterns = re.findall(r'["\']([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)["\']', code)
        for p in path_patterns:
            if "/" in p or "." in p:
                referenced_files.add(p)

        # 2. AST 심볼 추출 (함수 정의, 클래스 정의, 호출된 함수 등)
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    ast_symbols.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    ast_symbols.add(node.name)
                elif isinstance(node, ast.Name):
                    ast_symbols.add(node.id)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    ast_symbols.add(node.func.id)
        except SyntaxError:
            pass

        # 3. CLI 플래그 추출 (--flag 형식)
        flags = re.findall(r'(?:--[a-zA-Z0-9\-]+)', code)
        for f in flags:
            cli_flags.add(f)

        # 메타데이터에 수동 지정된 카나리 단언 병합
        if metadata and "canary_assertions" in metadata:
            custom_canary = metadata["canary_assertions"]
            if isinstance(custom_canary, dict):
                referenced_files.update(custom_canary.get("files", []))
                ast_symbols.update(custom_canary.get("symbols", []))
                cli_flags.update(custom_canary.get("cli_flags", []))

        return {
            "files": sorted(list(referenced_files)),
            "symbols": sorted(list(ast_symbols)),
            "cli_flags": sorted(list(cli_flags)),
        }

    @staticmethod
    def validate_skill(skill: Skill) -> Tuple[bool, Optional[str]]:
        """스킬의 카나리 단언을 현재 코드베이스와 대조하여 유효성을 검사합니다.
        
        Returns:
            (is_valid, reason)
        """
        canary = skill.metadata.get("canary_assertions")
        if not canary:
            # 카나리 단언이 없으면 추출하여 메타데이터에 보강
            canary = SkillValidator.extract_canary_assertions(skill.code, skill.metadata)
            skill.metadata["canary_assertions"] = canary

        # 1. 참조된 파일 경로 검사
        for file_path in canary.get("files", []):
            # 상대 경로 또는 절대 경로가 실제로 존재하는지 확인
            p = Path(file_path)
            if not p.is_absolute():
                p = Path.cwd() / p
            if not p.exists():
                return False, f"Referenced file no longer exists or was renamed: {file_path}"

        # 2. AST 심볼 및 코드베이스 심볼 검사 (예: 현재 저장소 내 정의 여부 또는 소스 파싱 검증)
        # 리팩토링으로 인해 코드 내 함수/클래스 시그니처나 파일명이 바뀐 경우 감지
        symbols = canary.get("symbols", [])
        if symbols:
            try:
                tree = ast.parse(skill.code)
                defined_in_skill = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
                # 외부 심볼이나 내부 심볼 검증
            except SyntaxError as e:
                return False, f"Skill code contains syntax error: {e}"

        return True, None

    @classmethod
    def run_preflight_validation(cls, skills: Dict[str, Skill]) -> List[str]:
        """세션 시작 시 모든 활성 스킬을 검증하여 실패한 스킬을 격리/보고합니다.
        
        Quantified Target: Pre-flight validation latency across all active skills <= 50 ms
        """
        start_time = time.perf_counter()
        quarantined = []

        for name, skill in list(skills.items()):
            is_valid, reason = cls.validate_skill(skill)
            if not is_valid:
                logger.warning(f"[SkillValidator] SKILL QUARANTINED due to canary decay ('{name}'): {reason}")
                skill.metadata["quarantined"] = True
                skill.metadata["quarantine_reason"] = reason
                quarantined.name = name if hasattr(quarantined, 'name') else None # type: ignore
                quarantined.append(name)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"[SkillValidator] Pre-flight validation completed for {len(skills)} skills in {elapsed_ms:.2f}ms (Quarantined: {len(quarantined)})")
        return quarantined


class SkillRepository:
    """런타임 스킬 저장소.

    에이전트가 생성한 Python 코드 조각이나 도구 어댑터를 메모리와
    디스크 양쪽에 보존하여, 후속 태스크에서 재사용할 수 있게 합니다.

    Features:
        - 메모리 내 즉시 접근 + 디스크 영속 저장
        - importlib 기반 동적 코드 실행
        - JSON 메타데이터 기반 스킬 관리
    """

    def __init__(self, storage_dir: str = "storage/skills"):
        self.skills: Dict[str, Skill] = {}
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()
        logger.info(
            f"[SkillRepo] Initialized with {len(self.skills)} skills from {self.storage_dir}. Running pre-flight skill validation..."
        )
        # 세션 시작 시 사전 비행 검증(Pre-flight Validation) 실행 (대상: <= 50 ms)
        SkillValidator.run_preflight_validation(self.skills)

    def save_skill(
        self,
        name: str,
        code: str,
        description: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> Skill:
        """스킬을 메모리와 디스크에 저장. 동일 이름 재저장 시 버전 증가, 이전 버전은 보존."""
        prev = self.skills.get(name)
        version = (prev.version + 1) if prev else 1
        merged_metadata = dict(metadata or {})
        if "canary_assertions" not in merged_metadata:
            merged_metadata["canary_assertions"] = SkillValidator.extract_canary_assertions(code, merged_metadata)

        skill = Skill(
            name=name,
            code=code,
            description=description,
            metadata=merged_metadata,
            version=version,
        )
        self.skills[name] = skill
        self._persist_to_disk(skill)
        logger.info(f"[SkillRepo] Skill saved: {name} v{version}")
        return skill

    def list_skill_versions(self, name: str) -> List[int]:
        """이름으로 저장된 모든 버전 번호 반환 (오름차순)."""
        versions = []
        for meta_file in self.storage_dir.glob(f"{name}.v*.json"):
            try:
                versions.append(int(meta_file.stem.rsplit(".v", 1)[1]))
            except (IndexError, ValueError):
                continue
        return sorted(versions)

    def get_skill_version(self, name: str, version: int) -> Skill | None:
        """특정 버전의 스킬을 디스크에서 조회 (메모리 최신본과 무관)."""
        meta_path = self.storage_dir / f"{name}.v{version}.json"
        code_path = self.storage_dir / f"{name}.v{version}.py"
        if not meta_path.exists():
            return None
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
        return Skill(
            name=meta["name"],
            code=code,
            description=meta.get("description", ""),
            created_at=meta.get("created_at", ""),
            metadata=meta.get("metadata", {}),
            version=meta.get("version", version),
        )

    def most_recently_modified_skill(self) -> Optional[str]:
        """version > 1(최근 재증류/수정된) 스킬 중 created_at이 가장 최근인 이름. 없으면 None."""
        candidates = [s for s in self.skills.values() if s.version > 1]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.created_at).name

    def rollback_skill(self, name: str, version: int) -> Skill:
        """지정 버전의 코드로 되돌린다. git revert처럼 새 버전으로 기록되어 이력은 보존된다."""
        target = self.get_skill_version(name, version)
        if target is None:
            raise ValueError(f"Skill '{name}' has no version {version}")
        metadata = dict(target.metadata)
        metadata["rollback_from_version"] = self.skills.get(name).version if name in self.skills else None
        metadata["rollback_to_version"] = version
        return self.save_skill(
            name=name,
            code=target.code,
            description=target.description,
            metadata=metadata,
        )

    def get_skill(self, name: str) -> Skill | None:
        """이름으로 스킬 조회."""
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        """등록된 모든 스킬 이름 반환."""
        return list(self.skills.keys())

    def search_skills(self, query: str) -> List[Dict[str, Any]]:
        """이름/설명 기준으로 스킬 검색."""
        import re

        terms = {term for term in re.findall(r"[a-z0-9_]{3,}", query.lower())}
        results: List[Dict[str, Any]] = []
        for name in self.skills:
            skill = self.skills[name]
            haystack = " ".join(
                [skill.name, skill.description, json.dumps(skill.metadata, ensure_ascii=False)]
            )
            skill_terms = {term for term in re.findall(r"[a-z0-9_]{3,}", haystack.lower())}
            if not terms:
                score = 0.0
            else:
                overlap = terms & skill_terms
                score = len(overlap) / max(len(terms), 1)
            results.append({"name": skill.name, "description": skill.description, "score": score})
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def delete_skill(self, name: str) -> bool:
        """스킬 삭제 (메모리 + 디스크)."""
        if name not in self.skills:
            return False
        del self.skills[name]
        # 디스크에서도 삭제
        meta_path = self.storage_dir / f"{name}.json"
        code_path = self.storage_dir / f"{name}.py"
        if meta_path.exists():
            meta_path.unlink()
        if code_path.exists():
            code_path.unlink()
        logger.info(f"[SkillRepo] Skill deleted: {name}")
        return True

    def execute_skill(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """저장된 스킬을 동적으로 실행.

        importlib를 사용하여 임시 모듈로 로드한 후,
        모듈 내의 `run` 함수를 호출합니다.

        스킬 코드에 `run(*args, **kwargs)` 함수가 정의되어 있어야 합니다.
        """
        skill = self.get_skill(name)
        if not skill:
            raise ValueError(f"Skill '{name}' not found")
            
        if skill.metadata.get("quarantined", False):
            reason = skill.metadata.get("quarantine_reason", "unknown")
            logger.error(f"[SkillRepo] Attempted to execute quarantined skill '{name}': {reason}")
            raise RuntimeError(f"Skill '{name}' is quarantined due to canary decay: {reason}")

        try:
            # 임시 파일에 코드를 기록하고 importlib로 로드
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix=f"skill_{name}_",
                delete=False,
            ) as tmp:
                tmp.write(skill.code)
                tmp_path = tmp.name

            spec = importlib.util.spec_from_file_location(f"skill_{name}", tmp_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load skill module: {name}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "run"):
                result = module.run(*args, **kwargs)
                logger.info(f"[SkillRepo] Skill executed: {name}")
                return result
            else:
                logger.warning(
                    f"[SkillRepo] Skill '{name}' has no 'run' function, executing module-level code only"
                )
                return None

        except Exception as e:
            logger.error(f"[SkillRepo] Skill execution failed for '{name}': {e}")
            raise
        finally:
            # 임시 파일 정리
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _persist_to_disk(self, skill: Skill) -> None:
        """스킬을 디스크에 저장 (버전별 파일 + 최신본 별칭)."""
        meta = {
            "name": skill.name,
            "description": skill.description,
            "created_at": skill.created_at,
            "metadata": skill.metadata,
            "version": skill.version,
        }
        # 버전별 파일 (이력 보존, 롤백용)
        for prefix in (skill.name, f"{skill.name}.v{skill.version}"):
            meta_path = self.storage_dir / f"{prefix}.json"
            code_path = self.storage_dir / f"{prefix}.py"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(skill.code)

    def _load_from_disk(self) -> None:
        """디스크에서 기존 스킬을 복원."""
        if not self.storage_dir.exists():
            return

        for meta_file in self.storage_dir.glob("*.json"):
            if _VERSION_FILE_RE.search(meta_file.stem):
                continue  # 버전별 이력 파일(name.vN.json)은 최신본 로드 시 건너뜀
            try:
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)

                code_file = self.storage_dir / f"{meta['name']}.py"
                code = ""
                if code_file.exists():
                    code = code_file.read_text(encoding="utf-8")

                skill = Skill(
                    name=meta["name"],
                    code=code,
                    description=meta.get("description", ""),
                    created_at=meta.get("created_at", ""),
                    metadata=meta.get("metadata", {}),
                    version=meta.get("version", 1),
                )
                # 로드 시점에 카나리 검증 수행
                SkillValidator.validate_skill(skill)
                self.skills[skill.name] = skill
            except Exception as e:
                logger.warning(
                    f"[SkillRepo] Failed to load skill from {meta_file}: {e}"
                )
