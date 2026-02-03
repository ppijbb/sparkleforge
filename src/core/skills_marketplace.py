"""
Skills Marketplace - Skills repository and sharing structure

GitHub 기반 Skills 마켓플레이스 구조.
Skills 설치/업그레이드 CLI 도구 및 검증 프레임워크.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import shutil

from src.core.skills_manager import get_skill_manager
from src.core.skills_loader import Skill

logger = logging.getLogger(__name__)


class SkillsMarketplace:
    """Skills 마켓플레이스 관리."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """초기화."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = Path(project_root)
        self.skills_dir = self.project_root / "skills"
        self.marketplace_config = self.project_root / ".skills_marketplace.json"
        self.skill_manager = get_skill_manager()
    
    def install_skill_from_github(self, repo_url: str, skill_name: Optional[str] = None) -> bool:
        """GitHub 저장소에서 Skill 설치."""
        try:
            # 저장소 이름 추출
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            skill_id = skill_name or repo_name
            
            # 임시 디렉토리에 클론
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                clone_dir = Path(tmpdir) / repo_name
                
                # Git clone
                subprocess.run(
                    ["git", "clone", repo_url, str(clone_dir)],
                    check=True,
                    capture_output=True
                )
                
                # Skill 디렉토리로 복사
                target_dir = self.skills_dir / skill_id
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                
                shutil.copytree(clone_dir, target_dir)
                
                # 검증
                if self.validate_skill(skill_id):
                    # 마켓플레이스 설정 업데이트
                    self._update_marketplace_config(skill_id, repo_url)
                    
                    logger.info(f"✅ Installed skill '{skill_id}' from {repo_url}")
                    return True
                else:
                    logger.error(f"❌ Skill validation failed: {skill_id}")
                    shutil.rmtree(target_dir)
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to install skill from {repo_url}: {e}")
            return False
    
    def validate_skill(self, skill_id: str) -> bool:
        """Skill 검증."""
        skill_path = self.skills_dir / skill_id
        
        # 1. 디렉토리 존재 확인
        if not skill_path.exists():
            logger.error(f"Skill directory not found: {skill_id}")
            return False
        
        # 2. SKILL.md 존재 확인
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            logger.error(f"SKILL.md not found for skill: {skill_id}")
            return False
        
        # 3. 메타데이터 파싱 확인
        try:
            skill = self.skill_manager.load_skill(skill_id)
            if not skill:
                return False
        except Exception as e:
            logger.error(f"Failed to parse skill metadata: {e}")
            return False
        
        # 4. 필수 필드 확인
        if not skill.metadata.skill_id or not skill.metadata.version:
            logger.error(f"Missing required metadata fields for skill: {skill_id}")
            return False
        
        logger.info(f"✅ Skill validated: {skill_id}")
        return True
    
    def upgrade_skill(self, skill_id: str) -> bool:
        """Skill 업그레이드."""
        # 마켓플레이스 설정에서 저장소 URL 확인
        config = self._load_marketplace_config()
        
        if skill_id not in config.get("installed_skills", {}):
            logger.error(f"Skill '{skill_id}' not found in marketplace config")
            return False
        
        repo_url = config["installed_skills"][skill_id].get("repo_url")
        if not repo_url:
            logger.error(f"No repository URL found for skill: {skill_id}")
            return False
        
        # 재설치
        return self.install_skill_from_github(repo_url, skill_id)
    
    def list_installed_skills(self) -> List[Dict[str, Any]]:
        """설치된 Skills 목록."""
        config = self._load_marketplace_config()
        installed = config.get("installed_skills", {})
        
        results = []
        for skill_id, info in installed.items():
            skill = self.skill_manager.load_skill(skill_id)
            if skill:
                results.append({
                    "skill_id": skill_id,
                    "name": skill.metadata.name,
                    "version": skill.metadata.version,
                    "repo_url": info.get("repo_url"),
                    "installed_at": info.get("installed_at")
                })
        
        return results
    
    def uninstall_skill(self, skill_id: str) -> bool:
        """Skill 제거."""
        skill_dir = self.skills_dir / skill_id
        
        if not skill_dir.exists():
            logger.warning(f"Skill directory not found: {skill_id}")
            return False
        
        try:
            shutil.rmtree(skill_dir)
            
            # 마켓플레이스 설정에서 제거
            config = self._load_marketplace_config()
            if skill_id in config.get("installed_skills", {}):
                del config["installed_skills"][skill_id]
                self._save_marketplace_config(config)
            
            logger.info(f"✅ Uninstalled skill: {skill_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to uninstall skill {skill_id}: {e}")
            return False
    
    def _load_marketplace_config(self) -> Dict[str, Any]:
        """마켓플레이스 설정 로드."""
        if self.marketplace_config.exists():
            with open(self.marketplace_config, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "installed_skills": {},
            "updated_at": datetime.now().isoformat()
        }
    
    def _save_marketplace_config(self, config: Dict[str, Any]):
        """마켓플레이스 설정 저장."""
        config["updated_at"] = datetime.now().isoformat()
        with open(self.marketplace_config, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _update_marketplace_config(self, skill_id: str, repo_url: str):
        """마켓플레이스 설정 업데이트."""
        config = self._load_marketplace_config()
        
        if "installed_skills" not in config:
            config["installed_skills"] = {}
        
        config["installed_skills"][skill_id] = {
            "repo_url": repo_url,
            "installed_at": datetime.now().isoformat()
        }
        
        self._save_marketplace_config(config)


def main():
    """CLI 진입점."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Skills Marketplace CLI")
    parser.add_argument("command", choices=["install", "upgrade", "list", "uninstall", "validate"],
                       help="Command to execute")
    parser.add_argument("--skill-id", help="Skill ID")
    parser.add_argument("--repo-url", help="Repository URL for installation")
    
    args = parser.parse_args()
    
    marketplace = SkillsMarketplace()
    
    if args.command == "install":
        if not args.repo_url:
            print("❌ --repo-url is required for install")
            return
        marketplace.install_skill_from_github(args.repo_url, args.skill_id)
    
    elif args.command == "upgrade":
        if not args.skill_id:
            print("❌ --skill-id is required for upgrade")
            return
        marketplace.upgrade_skill(args.skill_id)
    
    elif args.command == "list":
        skills = marketplace.list_installed_skills()
        print("\n📦 Installed Skills:")
        for skill in skills:
            print(f"  - {skill['skill_id']} (v{skill['version']})")
    
    elif args.command == "uninstall":
        if not args.skill_id:
            print("❌ --skill-id is required for uninstall")
            return
        marketplace.uninstall_skill(args.skill_id)
    
    elif args.command == "validate":
        if not args.skill_id:
            print("❌ --skill-id is required for validate")
            return
        valid = marketplace.validate_skill(args.skill_id)
        print("✅ Valid" if valid else "❌ Invalid")


if __name__ == "__main__":
    main()

