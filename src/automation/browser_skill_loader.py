import logging
import os
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class BrowserSkillLoader:
    """Loads and manages domain-specific skills and interaction mechanics for the browser."""

    def __init__(self, workspace_root: str = None):
        if workspace_root is None:
            # Assuming src/automation is current dir, go up two levels
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        self.skills_dir = os.path.join(workspace_root, "browser-skills")
        self.domain_skills_dir = os.path.join(self.skills_dir, "domain-skills")
        self.interaction_skills_dir = os.path.join(self.skills_dir, "interaction-skills")

        # Ensure directories exist
        os.makedirs(self.domain_skills_dir, exist_ok=True)
        os.makedirs(self.interaction_skills_dir, exist_ok=True)

    def _extract_domain(self, url: str) -> str:
        """Extract the base domain from a URL (e.g. https://github.com/foo -> github)."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www and tld for simple matching
            parts = domain.replace("www.", "").split(".")
            if len(parts) >= 2:
                return parts[-2]
            return parts[0] if parts else "unknown"
        except Exception:
            return "unknown"

    def get_domain_skills(self, url: str) -> str | None:
        """Get the domain skills markdown file content if it exists."""
        domain = self._extract_domain(url)
        skill_file = os.path.join(self.domain_skills_dir, f"{domain}.md")

        if os.path.exists(skill_file):
            try:
                with open(skill_file, encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to load domain skill for {domain}: {e}")

        return None

    def get_interaction_skill(self, skill_name: str) -> str | None:
        """Get an interaction skill by name (e.g., 'dialogs', 'iframes')."""
        skill_file = os.path.join(self.interaction_skills_dir, f"{skill_name}.md")

        if os.path.exists(skill_file):
            try:
                with open(skill_file, encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to load interaction skill {skill_name}: {e}")

        return None

    def inject_skills_into_prompt(self, url: str, system_message: str) -> str:
        """Appends relevant domain skills to the system message."""
        domain_skills = self.get_domain_skills(url)

        if not domain_skills:
            return system_message

        skill_context = f"\n\n# DOMAIN SKILLS FOR {self._extract_domain(url).upper()}\n"
        skill_context += "The following are learned skills and quirks for this domain. Use them to guide your actions:\n"
        skill_context += domain_skills

        return system_message + skill_context


# Singleton loader
_loader_instance = None


def get_skill_loader() -> BrowserSkillLoader:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = BrowserSkillLoader()
    return _loader_instance
