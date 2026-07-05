"""Anvil Skill Repository - 런타임 스킬/도구 저장소.

에이전트가 실행 중 생성한 코드 조각, 커스텀 도구, 함수 등을
세션 내에서 보존하고 재사용할 수 있도록 하는 저장소.
"""

import importlib.util
import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """저장된 스킬/도구 정의."""

    name: str
    code: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


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
            f"[SkillRepo] Initialized with {len(self.skills)} skills from {self.storage_dir}"
        )

    def save_skill(
        self,
        name: str,
        code: str,
        description: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> Skill:
        """스킬을 메모리와 디스크에 저장."""
        skill = Skill(
            name=name,
            code=code,
            description=description,
            metadata=metadata or {},
        )
        self.skills[name] = skill
        self._persist_to_disk(skill)
        logger.info(f"[SkillRepo] Skill saved: {name}")
        return skill

    def get_skill(self, name: str) -> Skill | None:
        """이름으로 스킬 조회."""
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        """등록된 모든 스킬 이름 반환."""
        return list(self.skills.keys())

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
        """스킬을 디스크에 저장 (메타데이터 JSON + 코드 .py)."""
        meta_path = self.storage_dir / f"{skill.name}.json"
        code_path = self.storage_dir / f"{skill.name}.py"

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": skill.name,
                    "description": skill.description,
                    "created_at": skill.created_at,
                    "metadata": skill.metadata,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(skill.code)

    def _load_from_disk(self) -> None:
        """디스크에서 기존 스킬을 복원."""
        if not self.storage_dir.exists():
            return

        for meta_file in self.storage_dir.glob("*.json"):
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
                )
                self.skills[skill.name] = skill
            except Exception as e:
                logger.warning(
                    f"[SkillRepo] Failed to load skill from {meta_file}: {e}"
                )
