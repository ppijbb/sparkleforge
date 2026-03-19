"""Open Skill Provider - External skill sources and sync.

Fetches skills from registry URLs (.well-known/skills.json style), Composio (when
COMPOSIO_API_KEY is set), and syncs them into the local skills directory.
"""

import json
import logging
import os
import re
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx

from src.core.skills_loader import SkillMetadata

logger = logging.getLogger(__name__)


def _default_skills_dir() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    return root / "skills"


def _safe_filename(skill_id: str) -> str:
    return re.sub(r"[^\w\-.]", "_", skill_id).strip("_") or "skill"


def _default_metadata(
    skill_id: str,
    name: str = "",
    description: str = "",
    category: str = "general",
) -> SkillMetadata:
    now = datetime.now(UTC).isoformat()
    return SkillMetadata(
        skill_id=skill_id,
        name=name or skill_id.replace("_", " ").title(),
        description=description or "",
        version="1.0.0",
        category=category,
        tags=[],
        author="external",
        created_at=now,
        updated_at=now,
        path=f"skills/{skill_id}",
        enabled=True,
        dependencies=[],
        required_tools=[],
        capabilities=[],
        allowed_tools=[],
        compatibility="",
    )


class OpenSkillProvider:
    """Fetch and sync skills from external registries and Composio."""

    def __init__(
        self,
        skills_dir: Path | None = None,
        request_timeout: float = 15.0,
    ) -> None:
        self.skills_dir = Path(skills_dir) if skills_dir else _default_skills_dir()
        self.request_timeout = request_timeout

    async def fetch_from_registry_url(
        self,
        registry_url: str,
    ) -> List[SkillMetadata]:
        """Fetch skill list from a URL returning JSON (e.g. .well-known/skills.json)."""
        results: List[SkillMetadata] = []
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                resp = await client.get(registry_url)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            logger.warning("fetch_from_registry_url HTTP error: %s", e)
            return results
        except json.JSONDecodeError as e:
            logger.warning("fetch_from_registry_url JSON error: %s", e)
            return results

        items = data if isinstance(data, list) else data.get("skills", data.get("tools", []))
        if not isinstance(items, list):
            return results

        base_url = f"{urlparse(registry_url).scheme}://{urlparse(registry_url).netloc}"
        for item in items:
            if not isinstance(item, dict):
                continue
            skill_id = item.get("id") or item.get("skill_id") or item.get("name") or ""
            if not skill_id:
                continue
            skill_id = str(skill_id).strip()
            name = item.get("name", skill_id)
            description = item.get("description", "")
            category = self.auto_categorize_from_text(
                f"{name} {description} {' '.join(item.get('tags', []))}"
            )
            meta = _default_metadata(
                skill_id=_safe_filename(skill_id),
                name=str(name),
                description=str(description),
                category=category,
            )
            meta.tags = list(item.get("tags", []))[:20]
            meta.capabilities = list(item.get("capabilities", item.get("inputs", [])))[:20]
            results.append(meta)
            if len(results) >= 500:
                break
        logger.info("Fetched %d skills from registry URL %s", len(results), registry_url)
        return results

    def auto_categorize_from_text(self, text: str) -> str:
        """Assign category from skill name/description/tags."""
        t = text.lower()
        if any(k in t for k in ("plan", "planning", "strategy", "objective", "계획", "전략")):
            return "research"
        if any(k in t for k in ("search", "find", "gather", "research", "검색", "수집", "연구")):
            return "research"
        if any(k in t for k in ("verify", "validate", "evaluate", "check", "검증", "평가")):
            return "evaluation"
        if any(k in t for k in ("synthesize", "summarize", "report", "종합", "요약", "리포트")):
            return "synthesis"
        if any(k in t for k in ("analyze", "analysis", "분석")):
            return "analysis"
        return "general"

    def auto_categorize(self, skill: SkillMetadata) -> str:
        """Assign category from SkillMetadata."""
        text = f"{skill.name} {skill.description} {' '.join(skill.tags)}"
        return self.auto_categorize_from_text(text)

    async def fetch_from_composio(
        self,
        categories: List[str] | None = None,
        limit: int = 100,
    ) -> List[SkillMetadata]:
        """Fetch tool/skill metadata from Composio when COMPOSIO_API_KEY is set."""
        results: List[SkillMetadata] = []
        api_key = os.environ.get("COMPOSIO_API_KEY", "").strip()
        if not api_key:
            logger.debug("COMPOSIO_API_KEY not set, skipping Composio fetch")
            return results
        try:
            from composio import Composio
            client = Composio(api_key=api_key)
            entities = getattr(client, "get_entities", None) or getattr(client, "entities", None)
            if entities is None:
                if hasattr(client, "get_actions"):
                    actions = client.get_actions()
                else:
                    return results
            else:
                actions = []
                try:
                    ent_list = entities() if callable(entities) else entities
                    for ent in (ent_list or [])[:limit]:
                        app_name = getattr(ent, "name", None) or (ent.get("name") if isinstance(ent, dict) else str(ent))
                        actions_list = getattr(ent, "actions", None) or (ent.get("actions", []) if isinstance(ent, dict) else [])
                        for act in (actions_list or [])[:20]:
                            if isinstance(act, dict):
                                actions.append({"app": app_name, "name": act.get("name", ""), "description": act.get("description", "")})
                            else:
                                actions.append({"app": app_name, "name": getattr(act, "name", ""), "description": getattr(act, "description", "")})
                except Exception as e:
                    logger.debug("Composio entities iteration: %s", e)
            if not actions and hasattr(client, "get_actions"):
                try:
                    actions = client.get_actions()
                except Exception as e:
                    logger.warning("Composio get_actions: %s", e)
                    return results
            if isinstance(actions, list):
                for i, act in enumerate(actions[:limit]):
                    if isinstance(act, dict):
                        name = act.get("name", act.get("action", "")) or f"composio_{i}"
                        desc = act.get("description", "")
                        app = act.get("app", "composio")
                    else:
                        name = getattr(act, "name", "") or getattr(act, "action", "") or f"composio_{i}"
                        desc = getattr(act, "description", "")
                        app = getattr(act, "app", "composio")
                    skill_id = _safe_filename(f"composio_{app}_{name}")
                    category = self.auto_categorize_from_text(f"{name} {desc}")
                    meta = _default_metadata(skill_id=skill_id, name=name, description=desc, category=category)
                    meta.tags = [app, "composio"]
                    results.append(meta)
            logger.info("Fetched %d skills from Composio", len(results))
        except Exception as e:
            logger.warning("fetch_from_composio failed: %s", e)
        return results

    async def sync_external_skills(
        self,
        registry_urls: List[str] | None = None,
        use_composio: bool = True,
        write_skill_md: bool = True,
    ) -> int:
        """Fetch from all sources and optionally write SKILL.md into skills_dir."""
        all_meta: Dict[str, SkillMetadata] = {}
        if registry_urls:
            for url in registry_urls:
                for meta in await self.fetch_from_registry_url(url):
                    all_meta[meta.skill_id] = meta
        if use_composio:
            for meta in await self.fetch_from_composio():
                if meta.skill_id not in all_meta:
                    all_meta[meta.skill_id] = meta
        if not write_skill_md or not all_meta:
            return len(all_meta)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for skill_id, meta in all_meta.items():
            path = self.skills_dir / skill_id
            path.mkdir(parents=True, exist_ok=True)
            skill_md = path / "SKILL.md"
            content = self._skill_md_content(meta)
            try:
                skill_md.write_text(content, encoding="utf-8")
                written += 1
            except Exception as e:
                logger.warning("Write SKILL.md %s: %s", skill_md, e)
        return written

    def _skill_md_content(self, meta: SkillMetadata) -> str:
        """Generate minimal SKILL.md content."""
        lines = [
            "---",
            f"name: {meta.name}",
            f"description: {meta.description[:500] if meta.description else ''}",
            "---",
            "",
            "## Overview",
            meta.description or meta.name,
            "",
            "## Instructions",
            "Use this skill as instructed by the agent.",
            "",
        ]
        if meta.tags:
            lines.append("## Capabilities")
            for t in meta.tags[:15]:
                lines.append(f"- {t}")
            lines.append("")
        return "\n".join(lines)
