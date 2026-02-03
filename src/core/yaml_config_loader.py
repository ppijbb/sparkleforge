"""
YAML Config Loader - YAML 설정 파일 로더

기존 config 시스템과 병행하여 YAML 설정 파일을 로드하고 병합합니다.
기존 config 객체는 수정하지 않고, 새로운 병합된 config를 반환합니다.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available. YAML config loading will be disabled.")


class YAMLConfigLoader:
    """YAML 설정 로더 - 기존 config 시스템과 병행"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        초기화
        
        Args:
            config_dir: 설정 파일 디렉토리 (None이면 프로젝트 루트의 configs 디렉토리 사용)
        """
        if not YAML_AVAILABLE:
            logger.warning("YAML support not available. Install PyYAML to use YAML config loading.")
        
        # 프로젝트 루트 찾기
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            self.config_dir = project_root / "configs"
        
        logger.info(f"YAML Config Loader initialized (config_dir: {self.config_dir})")
    
    def load_agents_config(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        agents.yaml 로드
        
        Args:
            path: 설정 파일 경로 (None이면 config_dir/agents.yaml 사용)
            
        Returns:
            설정 딕셔너리 (파일이 없으면 빈 dict)
        """
        if not YAML_AVAILABLE:
            return {}
        
        if path is None:
            path = self.config_dir / "agents.yaml"
        else:
            path = Path(path)
        
        if not path.exists():
            logger.debug(f"Agents config file not found: {path}")
            return {}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded agents config from {path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load agents config from {path}: {e}")
            return {}
    
    def load_tasks_config(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        tasks.yaml 로드
        
        Args:
            path: 설정 파일 경로 (None이면 config_dir/tasks.yaml 사용)
            
        Returns:
            설정 딕셔너리 (파일이 없으면 빈 dict)
        """
        if not YAML_AVAILABLE:
            return {}
        
        if path is None:
            path = self.config_dir / "tasks.yaml"
        else:
            path = Path(path)
        
        if not path.exists():
            logger.debug(f"Tasks config file not found: {path}")
            return {}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded tasks config from {path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load tasks config from {path}: {e}")
            return {}
    
    def merge_with_existing(
        self,
        yaml_config: Dict[str, Any],
        existing_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        기존 config와 병합 (기존 config 수정 없음)
        
        Args:
            yaml_config: YAML에서 로드한 설정
            existing_config: 기존 설정
            
        Returns:
            병합된 새로운 설정 딕셔너리 (기존 config는 수정하지 않음)
        """
        # 기존 config 복사 (깊은 복사)
        merged = self._deep_copy_dict(existing_config)
        
        # YAML 설정으로 업데이트 (재귀적 병합)
        merged = self._deep_merge(merged, yaml_config)
        
        logger.debug(f"Merged YAML config with existing config")
        return merged
    
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """딕셔너리 깊은 복사"""
        import copy
        return copy.deepcopy(d)
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """재귀적 딕셔너리 병합"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 둘 다 딕셔너리면 재귀적 병합
                result[key] = self._deep_merge(result[key], value)
            else:
                # 그 외에는 업데이트 값으로 덮어쓰기
                result[key] = value
        
        return result

