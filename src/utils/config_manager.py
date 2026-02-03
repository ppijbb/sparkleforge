"""
Configuration Manager for Local Researcher Project
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Import configuration classes from researcher_config
try:
    from src.core.researcher_config import MCPConfig, LLMConfig, AgentConfig, ResearchConfig
    from src.core.researcher_config import CompressionConfig, VerificationConfig, ContextWindowConfig, ReliabilityConfig
    from src.core.researcher_config import OutputConfig, ResearcherSystemConfig
except ImportError:
    # Fallback if researcher_config is not available
    MCPConfig = None
    LLMConfig = None
    AgentConfig = None
    ResearchConfig = None
    CompressionConfig = None
    VerificationConfig = None
    ContextWindowConfig = None
    ReliabilityConfig = None
    OutputConfig = None
    ResearcherSystemConfig = None


class ConfigManager:
    """Configuration manager for the Local Researcher system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or environment."""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
                self.config = {}
        
        # Load from environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_config = {
            'monitoring.interval': int(os.getenv('MONITORING_INTERVAL', 5)),
            'monitoring.history_size': int(os.getenv('MONITORING_HISTORY_SIZE', 1000)),
            'monitoring.thresholds': {
                'cpu_usage': float(os.getenv('CPU_THRESHOLD', 80.0)),
                'memory_usage': float(os.getenv('MEMORY_THRESHOLD', 85.0)),
                'disk_usage': float(os.getenv('DISK_THRESHOLD', 90.0)),
                'error_rate': float(os.getenv('ERROR_RATE_THRESHOLD', 10.0))
            }
        }
        
        # Merge with existing config
        self.config.update(env_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, file_path: Optional[str] = None):
        """Save configuration to file."""
        save_path = file_path or self.config_path
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
