"""Skill marketplace — cross-instance skill sharing (Anvil Phase Λ).

Provides export/import of Anvil skills as portable manifest bundles, a local
directory (or git repo) backed sharing registry, a remote Supabase registry
backend, and security verification of imported skills before they are loaded
into a SkillRepository.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .skill_repository import Skill, SkillRepository

MANIFEST_VERSION = "1"
MANIFEST_FILENAME = "skill.json"
CODE_FILENAME = "skill.py"


@dataclass
class SkillManifest:
    """Portable deployment unit: code + metadata + dependency declarations."""

    name: str
    description: str
    code: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    manifest_version: str = MANIFEST_VERSION
    code_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "dependencies": list(self.dependencies),
            "manifest_version": self.manifest_version,
            "code_sha256": self.code_sha256 or self._compute_sha(self.code),
        }
        return payload

    @staticmethod
    def _compute_sha(code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    @classmethod
    def from_skill(
        cls,
        skill: Skill,
        *,
        dependencies: Iterable[str] | None = None,
    ) -> "SkillManifest":
        return cls(
            name=skill.name,
            description=skill.description,
            code=skill.code,
            metadata=dict(skill.metadata or {}),
            dependencies=list(dependencies or []),
        )

    @classmethod
    def from_draft(
        cls,
        draft: Any,
        *,
        dependencies: Iterable[str] | None = None,
    ) -> "SkillManifest":
        return cls(
            name=draft.name,
            description=draft.description,
            code=draft.code,
            metadata=dict(getattr(draft, "metadata", {}) or {}),
            dependencies=list(dependencies or []),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillManifest":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            code=data.get("code", ""),
            metadata=data.get("metadata", {}) or {},
            dependencies=list(data.get("dependencies", []) or []),
            manifest_version=data.get("manifest_version", MANIFEST_VERSION),
            code_sha256=data.get("code_sha256", ""),
        )

    def verify_integrity(self) -> bool:
        if not self.code_sha256:
            return True
        return self._compute_sha(self.code) == self.code_sha256


SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def validate_skill_name(name: str) -> tuple[bool, str]:
    if not isinstance(name, str) or not name:
        return False, "skill name must be a non-empty string"
    if not SKILL_NAME_PATTERN.match(name):
        return False, "skill name must match ^[a-z0-9][a-z0-9_-]{0,63}$"
    return True, ""


class SkillSecurityVerifier:
    """Static verification of imported skill code before loading.

    Performs AST-based static scanning for dangerous constructs and optional
    integration with GuardPlane quarantine when available. This is a defense
    against importing arbitrary code from untrusted skill bundles.
    """

    DANGEROUS_NAMES = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
    }
    DANGEROUS_ATTRS = {"__builtins__"}
    DANGEROUS_MODULES = {"subprocess", "os", "socket", "ctypes", "shutil"}

    def __init__(self, guard_plane: Any | None = None) -> None:
        self.guard_plane = guard_plane

    def verify(self, manifest: SkillManifest) -> tuple[bool, list[str]]:
        reasons: list[str] = []

        if not manifest.verify_integrity():
            reasons.append("code_sha256 mismatch — manifest integrity check failed")

        try:
            tree = ast.parse(manifest.code)
        except SyntaxError as exc:
            reasons.append(f"syntax error: {exc}")
            return False, reasons

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in self.DANGEROUS_NAMES:
                    reasons.append(f"call to dangerous builtin: {func.id}")
                if isinstance(func, ast.Attribute) and func.attr in self.DANGEROUS_ATTRS:
                    reasons.append(f"access to dangerous attribute: {func.attr}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in self.DANGEROUS_MODULES:
                        reasons.append(f"import of dangerous module: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".")[0] in self.DANGEROUS_MODULES:
                    reasons.append(f"import from dangerous module: {module}")

        verified = not reasons

        if verified and self.guard_plane is not None:
            quarantine = getattr(self.guard_plane, "quarantine_file", None)
            if callable(quarantine):
                try:
                    quarantine(manifest.name, manifest.code, reason="skill_import")
                except Exception:
                    reasons.append("guard_plane.quarantine_file raised an exception")
                    verified = False

        return verified, reasons


class LocalSkillShareBackend:
    """Local directory (or git repo) backed skill sharing registry.

    A share backend is a directory containing exported skill bundles. Each
    bundle is a subdirectory with a ``skill.json`` manifest and ``skill.py``
    code file. If the directory is a git repository, ``pull``/``push`` can
    synchronize bundles across instances.
    """

    def __init__(self, share_dir: str | Path) -> None:
        self.share_dir = Path(share_dir)
        self.share_dir.mkdir(parents=True, exist_ok=True)

    def publish(self, manifest: SkillManifest) -> Path:
        ok, reason = validate_skill_name(manifest.name)
        if not ok:
            raise ValueError(reason)
        bundle_dir = self.share_dir / manifest.name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = bundle_dir / MANIFEST_FILENAME
        code_path = bundle_dir / CODE_FILENAME

        payload = manifest.to_dict()
        payload["code"] = manifest.code
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        code_path.write_text(manifest.code, encoding="utf-8")
        return bundle_dir

    def list_shared(self) -> list[str]:
        if not self.share_dir.exists():
            return []
        names: list[str] = []
        for child in sorted(self.share_dir.iterdir()):
            if child.is_dir() and (child / MANIFEST_FILENAME).exists():
                names.append(child.name)
        return names

    def read_manifest(self, name: str) -> SkillManifest | None:
        ok, reason = validate_skill_name(name)
        if not ok:
            return None
        bundle_dir = self.share_dir / name
        manifest_path = bundle_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            return None
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        code_path = bundle_dir / CODE_FILENAME
        if code_path.exists():
            data["code"] = code_path.read_text(encoding="utf-8")
        return SkillManifest.from_dict(data)

    def search(self, query: str) -> list[dict[str, Any]]:
        terms = _terms(query)
        results: list[dict[str, Any]] = []
        for name in self.list_shared():
            manifest = self.read_manifest(name)
            if manifest is None:
                continue
            haystack = " ".join(
                [manifest.name, manifest.description, json.dumps(manifest.metadata, ensure_ascii=False)]
            )
            skill_terms = _terms(haystack)
            if not terms:
                score = 0.0
            else:
                overlap = terms & skill_terms
                score = len(overlap) / max(len(terms), 1)
            results.append(
                {
                    "name": manifest.name,
                    "description": manifest.description,
                    "score": score,
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def pull(self) -> bool:
        if not (self.share_dir / ".git").exists():
            return False
        try:
            subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=str(self.share_dir),
                check=True,
                capture_output=True,
            )
            return True
        except Exception:
            return False

    def push(self) -> bool:
        if not (self.share_dir / ".git").exists():
            return False
        try:
            subprocess.run(
                ["git", "add", "."],
                cwd=str(self.share_dir),
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "Publish shared skills"],
                cwd=str(self.share_dir),
                check=False,
                capture_output=True,
            )
            subprocess.run(
                ["git", "push"],
                cwd=str(self.share_dir),
                check=True,
                capture_output=True,
            )
            return True
        except Exception:
            return False


class RemoteSkillRegistryBackend:
    """Supabase (Postgres + Storage) backed skill sharing registry.

    Implements the same ``publish``/``list_shared``/``read_manifest``/``search``
    interface as :class:`LocalSkillShareBackend` so :class:`SkillMarketplace`
    can swap backends without changes. Credentials (Supabase URL, anon/service
    key) are loaded via :class:`CredentialVault` — never hardcoded.
    """

    METADATA_TABLE = "skill_registry_metadata"
    STORAGE_BUCKET = "skill-registry-bundles"

    def __init__(
        self,
        *,
        supabase_url: str | None = None,
        credential_vault: Any | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.credential_vault = credential_vault
        self.supabase_url = supabase_url or self._load_credential("supabase_url")
        self.supabase_key = self._load_credential("supabase_anon_key") or self._load_credential(
            "supabase_service_key"
        )
        self.http_client = http_client

    def _load_credential(self, key: str) -> str | None:
        if self.credential_vault is not None:
            try:
                return self.credential_vault.retrieve(key)
            except Exception:
                return None
        return None

    def _get_http_client(self):
        if self.http_client is not None:
            return self.http_client
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "RemoteSkillRegistryBackend requires the 'requests' package for HTTP access"
            ) from exc
        return requests

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.supabase_key:
            headers["apikey"] = self.supabase_key
            headers["Authorization"] = f"Bearer {self.supabase_key}"
        return headers

    def _rest_url(self, path: str) -> str:
        if not self.supabase_url:
            raise RuntimeError("Supabase URL not configured in CredentialVault")
        base = self.supabase_url.rstrip("/")
        return f"{base}/rest/v1/{path.lstrip('/')}"

    def publish(self, manifest: SkillManifest) -> Path:
        ok, reason = validate_skill_name(manifest.name)
        if not ok:
            raise ValueError(reason)
        if not self.supabase_url or not self.supabase_key:
            raise RuntimeError("Supabase credentials not configured in CredentialVault")
        client = self._get_http_client()
        payload = manifest.to_dict()
        payload["code"] = manifest.code
        row = {
            "name": manifest.name,
            "description": manifest.description,
            "tags": list(manifest.metadata.get("tags", []) or []),
            "author": manifest.metadata.get("author", "") or "",
            "version": manifest.metadata.get("version", "1") or "1",
            "code_sha256": payload["code_sha256"],
            "risk_level": manifest.metadata.get("risk_level", "unknown") or "unknown",
            "download_count": 0,
            "metadata": json.dumps(manifest.metadata, ensure_ascii=False),
            "dependencies": json.dumps(list(manifest.dependencies), ensure_ascii=False),
        }
        client.post(
            self._rest_url(self.METADATA_TABLE),
            json=row,
            headers=self._headers(),
            timeout=30,
        )
        bundle_path = Path(tempfile.mkdtemp()) / f"{manifest.name}.json"
        bundle_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return bundle_path

    def list_shared(self) -> list[str]:
        if not self.supabase_url or not self.supabase_key:
            return []
        client = self._get_http_client()
        select_headers = dict(self._headers())
        select_headers["Accept"] = "application/json"
        select_headers["Range"] = "0-999"
        response = client.get(
            self._rest_url(self.METADATA_TABLE),
            params={"select": "name"},
            headers=select_headers,
            timeout=30,
        )
        if response.status_code >= 400:
            return []
        data = response.json()
        return [item["name"] for item in data if isinstance(item, dict) and "name" in item]

    def read_manifest(self, name: str) -> SkillManifest | None:
        ok, _reason = validate_skill_name(name)
        if not ok:
            return None
        if not self.supabase_url or not self.supabase_key:
            return None
        client = self._get_http_client()
        select_headers = dict(self._headers())
        select_headers["Accept"] = "application/vnd.pgrst.object+json"
        response = client.get(
            self._rest_url(self.METADATA_TABLE),
            params={"name": f"eq.{name}", "select": "*"},
            headers=select_headers,
            timeout=30,
        )
        if response.status_code >= 400 or not response.text:
            return None
        row = response.json()
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        dependencies = row.get("dependencies") or []
        if isinstance(dependencies, str):
            try:
                dependencies = json.loads(dependencies)
            except json.JSONDecodeError:
                dependencies = []
        return SkillManifest.from_dict(
            {
                "name": row.get("name", name),
                "description": row.get("description", "") or "",
                "code": row.get("code", "") or "",
                "metadata": metadata,
                "dependencies": dependencies,
                "manifest_version": MANIFEST_VERSION,
                "code_sha256": row.get("code_sha256", "") or "",
            }
        )

    def search(self, query: str) -> list[dict[str, Any]]:
        terms = _terms(query)
        results: list[dict[str, Any]] = []
        for name in self.list_shared():
            manifest = self.read_manifest(name)
            if manifest is None:
                continue
            haystack = " ".join(
                [manifest.name, manifest.description, json.dumps(manifest.metadata, ensure_ascii=False)]
            )
            skill_terms = _terms(haystack)
            if not terms:
                score = 0.0
            else:
                overlap = terms & skill_terms
                score = len(overlap) / max(len(terms), 1)
            results.append(
                {
                    "name": manifest.name,
                    "description": manifest.description,
                    "score": score,
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results


class SkillMarketplace:
    """Coordinates export/import/search of skills across instances."""

    def __init__(
        self,
        repository: SkillRepository,
        share_backend: Any,
        *,
        verifier: SkillSecurityVerifier | None = None,
        trust_gate: Any | None = None,
    ) -> None:
        self.repository = repository
        self.share_backend = share_backend
        self.verifier = verifier or SkillSecurityVerifier()
        self.trust_gate = trust_gate

    def export_skill(self, name: str, *, dependencies: Iterable[str] | None = None) -> Path:
        skill = self.repository.get_skill(name)
        if skill is None:
            raise ValueError(f"Skill '{name}' not found in repository")
        if self.trust_gate is not None and not self.trust_gate.is_trusted(skill):
            raise ValueError(
                f"Skill '{name}' is not trusted by the Skill Gym gate and cannot be exported"
            )
        manifest = SkillManifest.from_skill(skill, dependencies=dependencies)
        return self.share_backend.publish(manifest)

    def export_draft(self, draft: Any, *, dependencies: Iterable[str] | None = None) -> Path:
        if self.trust_gate is not None:
            report = self.trust_gate.evaluate_draft(draft)
            if not report.passed:
                raise ValueError(
                    f"Skill draft '{draft.name}' failed the Skill Gym gate "
                    f"(average_score={report.average_score}) and cannot be exported"
                )
        manifest = SkillManifest.from_draft(draft, dependencies=dependencies)
        return self.share_backend.publish(manifest)

    def export_to_file(self, name: str, output_path: str | Path, *, dependencies: Iterable[str] | None = None) -> Path:
        skill = self.repository.get_skill(name)
        if skill is None:
            raise ValueError(f"Skill '{name}' not found in repository")
        manifest = SkillManifest.from_skill(skill, dependencies=dependencies)
        return _write_bundle(manifest, Path(output_path))

    def import_skill(self, source: str | Path, *, overwrite: bool = False) -> tuple[bool, Skill | None, list[str]]:
        manifest = _read_bundle(Path(source))
        if manifest is None:
            return False, None, ["could not read skill bundle from source"]
        return self._import_manifest(manifest, overwrite=overwrite)

    def import_shared(self, name: str, *, overwrite: bool = False) -> tuple[bool, Skill | None, list[str]]:
        ok, reason = validate_skill_name(name)
        if not ok:
            return False, None, [reason]
        manifest = self.share_backend.read_manifest(name)
        if manifest is None:
            return False, None, [f"shared skill '{name}' not found"]
        return self._import_manifest(manifest, overwrite=overwrite)

    def _import_manifest(self, manifest: SkillManifest, *, overwrite: bool) -> tuple[bool, Skill | None, list[str]]:
        verified, reasons = self.verifier.verify(manifest)
        if not verified:
            return False, None, reasons

        if self.trust_gate is not None:
            existing = self.repository.get_skill(manifest.name)
            if existing is not None and not self.trust_gate.is_trusted(existing):
                return False, None, [
                    f"skill '{manifest.name}' is not trusted by the Skill Gym gate"
                ]

        if not overwrite and self.repository.get_skill(manifest.name) is not None:
            return False, None, [f"skill '{manifest.name}' already exists; use overwrite=True"]

        metadata = dict(manifest.metadata)
        metadata["imported"] = True
        metadata["security_verified"] = True
        metadata["dependencies"] = list(manifest.dependencies)
        skill = self.repository.save_skill(
            manifest.name,
            manifest.code,
            description=manifest.description,
            metadata=metadata,
        )
        return True, skill, []

    def search(self, query: str) -> list[dict[str, Any]]:
        return self.share_backend.search(query)


def _write_bundle(manifest: SkillManifest, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".zip":
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
            payload = manifest.to_dict()
            payload["code"] = manifest.code
            archive.writestr(
                MANIFEST_FILENAME, json.dumps(payload, ensure_ascii=False, indent=2)
            )
            archive.writestr(CODE_FILENAME, manifest.code)
        return output_path
    bundle_dir = output_path
    bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    payload["code"] = manifest.code
    (bundle_dir / MANIFEST_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (bundle_dir / CODE_FILENAME).write_text(manifest.code, encoding="utf-8")
    return bundle_dir


def _read_bundle(source: Path) -> SkillManifest | None:
    if source.is_dir():
        manifest_path = source / MANIFEST_FILENAME
        if not manifest_path.exists():
            return None
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        code_path = source / CODE_FILENAME
        if code_path.exists():
            data["code"] = code_path.read_text(encoding="utf-8")
        return SkillManifest.from_dict(data)
    if source.is_file() and source.suffix == ".zip":
        with zipfile.ZipFile(source, "r") as archive:
            data = json.loads(archive.read(MANIFEST_FILENAME).decode("utf-8"))
            try:
                data["code"] = archive.read(CODE_FILENAME).decode("utf-8")
            except KeyError:
                pass
        return SkillManifest.from_dict(data)
    if source.is_file() and source.suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        return SkillManifest.from_dict(data)
    return None


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9_]{3,}", text.lower())}
