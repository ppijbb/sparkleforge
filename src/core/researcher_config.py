"""
Local Researcher Project Configuration (v2.0 - 9대 혁신)

Centralized configuration management for advanced multi-agent research system.
Supports 9 core innovations: Adaptive Supervisor, Hierarchical Compression, 
Multi-Model Orchestration, Continuous Verification, Streaming Pipeline,
Universal MCP Hub, Adaptive Context Window, Production-Grade Reliability,
Adaptive Research Depth.
"""

import os
from typing import List, Dict, Any, Optional, Literal
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict, field_validator


class TaskType(Enum):
    """Task types for Multi-Model Orchestration (혁신 3)."""
    PLANNING = "planning"
    DEEP_REASONING = "deep_reasoning"
    VERIFICATION = "verification"
    GENERATION = "generation"
    COMPRESSION = "compression"
    RESEARCH = "research"

class LLMConfig(BaseModel):
    """LLM configuration settings - Multi-Model Orchestration (혁신 3)."""
    
    # Primary provider (OpenRouter + Gemini 2.5 Flash Lite) - NO DEFAULTS
    provider: str = Field(description="LLM provider")
    primary_model: str = Field(description="Primary model")
    temperature: float = Field(ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(gt=0, description="Max tokens")
    api_key: str = Field(description="Google API key for Gemini models")
    
    # Multi-Model Orchestration (혁신 3) - OpenRouter Gemini 2.5 Flash-Lite 우선 - NO DEFAULTS
    planning_model: str = Field(description="Planning model")
    reasoning_model: str = Field(description="Reasoning model")
    verification_model: str = Field(description="Verification model")
    generation_model: str = Field(description="Generation model")
    compression_model: str = Field(description="Compression model")
    
    # OpenRouter API Key (Required) - NO DEFAULTS
    openrouter_api_key: str = Field(description="OpenRouter API key")
    
    # Cost optimization - NO DEFAULTS
    budget_limit: float = Field(ge=0.0, description="Budget limit")
    enable_cost_optimization: bool = Field(description="Enable cost optimization")

    # CLI Agent Support - 선택적 설정
    enable_cli_agents: bool = Field(default=False, description="Enable CLI agents")
    cli_agents_config: Dict[str, Any] = Field(default_factory=dict, description="CLI agent configurations")

    # Claude Code settings
    claude_code_api_key: Optional[str] = Field(default=None, description="Claude Code API key")

    # OpenCode settings
    open_code_model_path: Optional[str] = Field(default=None, description="OpenCode model path")

    # Gemini CLI settings
    gemini_cli_api_key: Optional[str] = Field(default=None, description="Gemini CLI API key")
    gemini_cli_model: str = Field(default="gemini-pro", description="Gemini CLI model")

    # Cline CLI settings
    cline_cli_config_path: Optional[str] = Field(default=None, description="Cline CLI config path")
    
    @field_validator('openrouter_api_key')
    @classmethod
    def validate_openrouter_api_key(cls, v):
        # 무료 모델 사용시 API 키 불필요
        if not v or v.strip() == "":
            return ""  # 빈 문자열 허용 (무료 모델)
        if not v.startswith('sk-or-'):
            raise ValueError("OPENROUTER_API_KEY must start with 'sk-or-'")
        return v
    
    @field_validator('primary_model')
    @classmethod
    def validate_primary_model(cls, v):
        # 모든 OpenRouter 모델 허용
        if not v or v.strip() == "":
            raise ValueError("Primary model must be specified")
        return v
    
    @classmethod
    def validate_environment(cls):
        """환경 변수 검증 및 필수 설정 확인."""
        import os
        
        # 필수 환경 변수 검증
        required_vars = {
            'OPENROUTER_API_KEY': 'OpenRouter API key is required',
            'LLM_MODEL': 'LLM model must be specified'
        }
        
        missing_vars = []
        for var, message in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"{var}: {message}")
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables:\n" + "\n".join(missing_vars))
        
        # OpenRouter API 키 형식 검증
        api_key = os.getenv('OPENROUTER_API_KEY')
        if bool(api_key) and not api_key.startswith('sk-or-'):
            raise ValueError("OPENROUTER_API_KEY must start with 'sk-or-'")
        
        return True


class AgentConfig(BaseModel):
    """Agent-specific settings with Adaptive Supervisor (혁신 1)."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    # Basic settings - NO DEFAULTS
    max_retries: int = Field(ge=0, description="Max retries")
    timeout_seconds: int = Field(gt=0, description="Timeout seconds")
    enable_self_planning: bool = Field(description="Enable self planning")
    enable_agent_communication: bool = Field(description="Enable agent communication")
    
    # Adaptive Supervisor (혁신 1) - NO DEFAULTS
    max_concurrent_research_units: int = Field(gt=0, description="Max concurrent research units")
    min_researchers: int = Field(gt=0, description="Min researchers")
    max_researchers: int = Field(gt=0, description="Max researchers")
    enable_fast_track: bool = Field(description="Enable fast track")
    enable_auto_retry: bool = Field(description="Enable auto retry")
    priority_queue_enabled: bool = Field(description="Priority queue enabled")
    
    # Quality monitoring - NO DEFAULTS
    enable_quality_monitoring: bool = Field(description="Enable quality monitoring")
    quality_threshold: float = Field(ge=0.0, le=1.0, description="Quality threshold")


@dataclass
class ResearchPresetConfig:
    """연구 깊이 프리셋 설정 (9번째 혁신: Adaptive Research Depth)."""
    description: str
    planning: Dict[str, Any]  # Planning 단계 설정
    researching: Dict[str, Any]  # Researching 단계 설정
    reporting: Dict[str, Any]  # Reporting 단계 설정


@dataclass
class AdaptiveResearchDepthConfig:
    """Adaptive Research Depth 설정 (9번째 혁신)."""
    enabled: bool = True
    default_preset: str = "auto"  # quick, medium, deep, auto
    enable_progressive_deepening: bool = True
    enable_self_adjusting: bool = True
    enable_dynamic_iteration: bool = True
    
    # Preset configurations
    presets: Dict[str, ResearchPresetConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        """기본 프리셋 설정 초기화."""
        if not self.presets:
            self.presets = {
                "quick": ResearchPresetConfig(
                    description="빠른 연구, 최소 깊이",
                    planning={
                        "decompose": {
                            "mode": "manual",
                            "initial_subtopics": 1,
                            "auto_max_subtopics": 2
                        }
                    },
                    researching={
                        "max_iterations": 1,
                        "iteration_mode": "fixed"
                    },
                    reporting={
                        "min_section_length": 300
                    }
                ),
                "medium": ResearchPresetConfig(
                    description="균형잡힌 연구",
                    planning={
                        "decompose": {
                            "mode": "manual",
                            "initial_subtopics": 5,
                            "auto_max_subtopics": 5
                        }
                    },
                    researching={
                        "max_iterations": 4,
                        "iteration_mode": "fixed"
                    },
                    reporting={
                        "min_section_length": 500
                    }
                ),
                "deep": ResearchPresetConfig(
                    description="깊이 있는 연구",
                    planning={
                        "decompose": {
                            "mode": "manual",
                            "initial_subtopics": 8,
                            "auto_max_subtopics": 8
                        }
                    },
                    researching={
                        "max_iterations": 7,
                        "iteration_mode": "fixed"
                    },
                    reporting={
                        "min_section_length": 800
                    }
                ),
                "auto": ResearchPresetConfig(
                    description="자율 결정 (에이전트가 최적 깊이 선택)",
                    planning={
                        "decompose": {
                            "mode": "auto",
                            "auto_max_subtopics": 8
                        }
                    },
                    researching={
                        "max_iterations": 6,
                        "iteration_mode": "flexible"
                    },
                    reporting={
                        "min_section_length": 500
                    }
                )
            }


@dataclass
class ResearchConfig:
    """Research-specific settings with Streaming Pipeline (혁신 5) and Adaptive Research Depth (혁신 9)."""
    # Basic settings - NO DEFAULTS
    max_sources: int
    search_timeout: int
    enable_academic_search: bool
    enable_web_search: bool
    enable_browser_automation: bool
    
    # Streaming Pipeline (혁신 5) - NO DEFAULTS
    enable_streaming: bool
    stream_chunk_size: int
    enable_progressive_reporting: bool
    enable_incremental_save: bool
    
    # Parallel processing - NO DEFAULTS
    enable_parallel_compression: bool
    enable_parallel_verification: bool
    
    # Adaptive Research Depth (혁신 9) - Optional with defaults
    research_depth: AdaptiveResearchDepthConfig = field(default_factory=AdaptiveResearchDepthConfig)


@dataclass
class MCPConfig:
    """MCP integration settings - Universal MCP Hub (혁신 6)."""
    enabled: bool
    server_names: List[str]
    connection_timeout: int
    
    # Universal MCP Hub (혁신 6) - MCP만 사용 - NO DEFAULTS
    enable_plugin_architecture: bool
    enable_smart_tool_selection: bool
    enable_auto_fallback: bool
    
    # Tool categories - NO DEFAULTS
    search_tools: List[str]
    data_tools: List[str]
    code_tools: List[str]
    academic_tools: List[str]
    business_tools: List[str]
    
    # MCP Server Builder configuration - Optional with defaults
    builder_enabled: bool = True
    builder_temp_dir: str = "temp/mcp_servers"
    builder_auto_cleanup: bool = True
    builder_cache_enabled: bool = True


@dataclass
class CompressionConfig:
    """Hierarchical Compression settings (혁신 2)."""
    enabled: bool
    enable_hierarchical_compression: bool
    compression_levels: int
    preserve_important_info: bool
    enable_compression_validation: bool
    compression_history_enabled: bool
    min_compression_ratio: float
    target_compression_ratio: float

@dataclass
class VerificationConfig:
    """Continuous Verification settings (혁신 4)."""
    enabled: bool
    enable_continuous_verification: bool
    verification_stages: int
    confidence_threshold: float
    enable_early_warning: bool
    enable_fact_check: bool
    enable_uncertainty_marking: bool

@dataclass
class ContextWindowConfig:
    """Adaptive Context Window settings (혁신 7)."""
    enabled: bool
    enable_adaptive_context: bool
    min_tokens: int
    max_tokens: int
    importance_based_preservation: bool
    enable_auto_compression: bool
    enable_long_term_memory: bool
    memory_refresh_interval: int

@dataclass
class ReliabilityConfig:
    """Production-Grade Reliability settings (혁신 8) - MCP만 사용."""
    enabled: bool
    enable_circuit_breaker: bool
    enable_exponential_backoff: bool
    enable_state_persistence: bool
    enable_health_check: bool
    enable_graceful_degradation: bool
    enable_detailed_logging: bool
    
    # Circuit breaker settings - NO DEFAULTS
    failure_threshold: int
    recovery_timeout: int
    
    # State persistence - NO DEFAULTS
    state_backend: str
    state_ttl: int

@dataclass
class OutputConfig:
    """Output and reporting settings."""
    output_dir: str
    enable_pdf_generation: bool
    enable_markdown_generation: bool
    enable_json_export: bool
    
    # Multi-format support - NO DEFAULTS
    enable_docx_export: bool
    enable_html_export: bool
    enable_latex_export: bool


@dataclass
class PromptRefinerConfig:
    """Prompt Refiner configuration for prompt optimization."""
    enabled: bool = True
    strategy: str = "aggressive"  # minimal, standard, aggressive, custom
    max_tokens: Optional[int] = None  # None means adaptive based on model
    collect_stats: bool = True


@dataclass
class OverseerConfig:
    """Greedy Overseer configuration for research quality enforcement."""
    enabled: bool = True
    max_iterations: int = 5
    completeness_threshold: float = 0.9
    quality_threshold: float = 0.85
    min_academic_sources: int = 3
    min_verified_sources: int = 5
    require_cross_validation: bool = True
    enable_human_loop: bool = True


class CouncilConfig(BaseModel):
    """LLM Council configuration for multi-model consensus - Optional with defaults."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    # Basic settings - Optional with defaults (상시 활성화 기본값)
    enabled: bool = True  # 기본 활성화
    auto_activate: bool = True  # 자동 활성화 기본값
    
    # Council models - Optional with defaults (gemini-2.5-flash 계열 우선)
    council_models: List[str] = Field(default_factory=lambda: [
        "google/gemini-2.5-flash-lite",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku"
    ])
    chairman_model: str = "google/gemini-2.5-flash-lite"
    
    # Auto-activation thresholds - Optional with defaults
    min_complexity_for_auto: float = 0.6
    
    # Process-specific activation - Optional with defaults (모두 기본 활성화)
    enable_for_planning: bool = True
    enable_for_execution: bool = True
    enable_for_evaluation: bool = True
    enable_for_verification: bool = True
    enable_for_synthesis: bool = True
    
    # OpenRouter settings - Optional (없으면 Council 비활성화)
    openrouter_api_key: Optional[str] = None
    openrouter_api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    request_timeout: float = 120.0
    
    @field_validator('council_models')
    def validate_council_models(cls, v):
        """Validate council models - must use gemini-2.5-flash 계열."""
        if not v:
            raise ValueError("council_models cannot be empty")
        # gpt-4 사용 금지 검증
        for model in v:
            if 'gpt-4' in model.lower() and 'gpt-4o-mini' not in model.lower():
                raise ValueError(f"gpt-4 models are not allowed. Use gemini-2.5-flash 계열 instead. Found: {model}")
        return v
    
    @field_validator('chairman_model')
    def validate_chairman_model(cls, v):
        """Validate chairman model - must use gemini-2.5-flash 계열."""
        if 'gpt-4' in v.lower() and 'gpt-4o-mini' not in v.lower():
            raise ValueError(f"gpt-4 models are not allowed. Use gemini-2.5-flash 계열 instead. Found: {v}")
        return v


class CascadeConfig(BaseModel):
    """Provider Cascade configuration for cost-efficient model selection."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    # Basic settings - Optional with defaults
    enabled: bool = True
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Confidence threshold for draft acceptance")
    min_models_for_cascade: int = Field(default=2, ge=2, description="Minimum number of models required for cascade")
    enable_adaptive_threshold: bool = Field(default=True, description="Enable complexity-based adaptive threshold")
    
    # Model classification thresholds
    drafter_cost_threshold: float = Field(default=0.0002, ge=0.0, description="Cost threshold for drafter classification")
    verifier_quality_threshold: float = Field(default=8.0, ge=0.0, le=10.0, description="Quality threshold for verifier classification")
    drafter_speed_threshold: float = Field(default=7.0, ge=0.0, le=10.0, description="Speed threshold for drafter classification")


class AgentToolConfig(BaseModel):
    """에이전트별 MCP 도구 할당 설정."""

    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    # 에이전트별 서버 할당
    planner_servers: List[str] = Field(default_factory=list, description="PlannerAgent용 MCP 서버 목록")
    executor_servers: List[str] = Field(default_factory=list, description="ExecutorAgent용 MCP 서버 목록")
    verifier_servers: List[str] = Field(default_factory=list, description="VerifierAgent용 MCP 서버 목록")
    generator_servers: List[str] = Field(default_factory=list, description="GeneratorAgent용 MCP 서버 목록")

    # 도구 카테고리 필터링
    planner_categories: List[str] = Field(default_factory=lambda: ["planning", "search", "utility"], description="PlannerAgent용 도구 카테고리")
    executor_categories: List[str] = Field(default_factory=lambda: ["search", "data", "academic", "business", "code", "browser", "file"], description="ExecutorAgent용 도구 카테고리")
    verifier_categories: List[str] = Field(default_factory=lambda: ["verification", "search", "data", "academic"], description="VerifierAgent용 도구 카테고리")
    generator_categories: List[str] = Field(default_factory=lambda: ["generation", "utility", "search", "document", "file"], description="GeneratorAgent용 도구 카테고리")

    # 도구 할당 제한
    max_tools_per_agent: int = Field(default=5, ge=1, le=20, description="에이전트당 최대 도구 수")
    enable_auto_discovery: bool = Field(default=True, description="자동 도구 발견 활성화")

    # Cross-Agent 통신 설정
    enable_cross_agent_tools: bool = Field(default=True, description="Cross-Agent 도구 활성화")
    cross_agent_timeout: float = Field(default=30.0, ge=5.0, le=300.0, description="Cross-Agent 호출 타임아웃(초)")


class ERAConfig(BaseModel):
    """ERA Agent Configuration - Safe Code Execution."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    enabled: bool = Field(default=True, description="Enable ERA for safe code execution")
    server_url: str = Field(default="http://localhost:8080", description="ERA server URL")
    auto_start: bool = Field(default=True, description="Auto-start ERA server if not running")
    agent_binary_path: Optional[str] = Field(default=None, description="ERA Agent binary path (auto-detect if None)")
    default_cpu: int = Field(default=1, ge=1, description="Default CPU count")
    default_memory: int = Field(default=256, ge=128, description="Default memory in MB")
    default_timeout: int = Field(default=30, gt=0, description="Default timeout in seconds")
    network_mode: str = Field(default="none", description="Network policy (none, allow_all)")
    api_key: Optional[str] = Field(default=None, description="ERA API key (optional)")
    start_timeout: int = Field(default=30, gt=0, description="서버 시작 타임아웃 (초)")
    max_retries: int = Field(default=3, ge=1, le=10, description="최대 재시도 횟수")
    retry_backoff: float = Field(default=2.0, gt=1.0, le=10.0, description="재시도 백오프 배수")
    
    @classmethod
    def from_env(cls) -> "ERAConfig":
        """Create ERAConfig from environment variables."""
        return cls(
            enabled=os.getenv("ERA_ENABLED", "true").lower() in ("true", "1", "yes"),
            server_url=os.getenv("ERA_SERVER_URL", "http://localhost:8080"),
            auto_start=os.getenv("ERA_AUTO_START", "true").lower() in ("true", "1", "yes"),
            agent_binary_path=os.getenv("ERA_AGENT_BINARY") or None,
            default_cpu=int(os.getenv("ERA_DEFAULT_CPU", "1")),
            default_memory=int(os.getenv("ERA_DEFAULT_MEMORY", "256")),
            default_timeout=int(os.getenv("ERA_DEFAULT_TIMEOUT", "30")),
            network_mode=os.getenv("ERA_NETWORK_MODE", "none"),
            api_key=os.getenv("ERA_API_KEY") or None,
            start_timeout=int(os.getenv("ERA_START_TIMEOUT", "30")),
            max_retries=int(os.getenv("ERA_MAX_RETRIES", "3")),
            retry_backoff=float(os.getenv("ERA_RETRY_BACKOFF", "2.0"))
        )


class ResearcherSystemConfig(BaseModel):
    """Overall system configuration with 8 core innovations."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    # Core configurations - NO DEFAULTS
    llm: LLMConfig = Field(description="LLM configuration")
    agent: AgentConfig = Field(description="Agent configuration")
    research: ResearchConfig = Field(description="Research configuration")
    mcp: MCPConfig = Field(description="MCP configuration")
    output: OutputConfig = Field(description="Output configuration")
    
    # Innovation configurations - NO DEFAULTS
    compression: CompressionConfig = Field(description="Compression configuration")
    verification: VerificationConfig = Field(description="Verification configuration")
    context_window: ContextWindowConfig = Field(description="Context window configuration")
    reliability: ReliabilityConfig = Field(description="Reliability configuration")
    council: CouncilConfig = Field(description="Council configuration")
    cascade: CascadeConfig = Field(default_factory=CascadeConfig, description="Cascade configuration")
    agent_tools: AgentToolConfig = Field(description="Agent tools configuration")
    prompt_refiner: PromptRefinerConfig = Field(default_factory=lambda: PromptRefinerConfig(), description="Prompt Refiner configuration")
    overseer: OverseerConfig = Field(default_factory=lambda: OverseerConfig(), description="Overseer configuration")
    
    def model_post_init(self, __context):
        # Ensure output directory exists
        os.makedirs(self.output.output_dir, exist_ok=True)
        
        # Validate configurations
        self._validate_configurations()
    
    def _validate_configurations(self):
        """Validate configuration consistency."""
        # Validate token limits
        if self.context_window.min_tokens >= self.context_window.max_tokens:
            raise ValueError("min_tokens must be less than max_tokens")
        
        # Validate researcher limits
        if self.agent.min_researchers > self.agent.max_researchers:
            raise ValueError("min_researchers must be less than or equal to max_researchers")
        
        # Validate confidence thresholds
        if not 0.0 <= self.verification.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.compression.min_compression_ratio <= 1.0:
            raise ValueError("min_compression_ratio must be between 0.0 and 1.0")


# Global configuration instance - will be loaded from environment
config = None


def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.llm


def get_cli_agents_config() -> Dict[str, Any]:
    """Get CLI agents configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")

    cli_config = {}

    # CLI 에이전트 활성화 여부
    cli_config['enabled'] = config.llm.enable_cli_agents

    # 각 CLI 에이전트 설정
    cli_config['agents'] = {
        'claude_code': {
            'enabled': bool(os.getenv('CLAUDE_CODE_API_KEY') or config.llm.claude_code_api_key),
            'api_key': os.getenv('CLAUDE_CODE_API_KEY') or config.llm.claude_code_api_key,
        },
        'open_code': {
            'enabled': bool(os.getenv('OPEN_CODE_MODEL_PATH') or config.llm.open_code_model_path),
            'model_path': os.getenv('OPEN_CODE_MODEL_PATH') or config.llm.open_code_model_path,
        },
        'gemini_cli': {
            'enabled': bool(os.getenv('GEMINI_CLI_API_KEY') or config.llm.gemini_cli_api_key),
            'api_key': os.getenv('GEMINI_CLI_API_KEY') or config.llm.gemini_cli_api_key,
            'model': os.getenv('GEMINI_CLI_MODEL', config.llm.gemini_cli_model),
        },
        'cline_cli': {
            'enabled': bool(os.getenv('CLINE_CLI_CONFIG_PATH') or config.llm.cline_cli_config_path),
            'config_path': os.getenv('CLINE_CLI_CONFIG_PATH') or config.llm.cline_cli_config_path,
        }
    }

    return cli_config


def initialize_cli_agents():
    """CLI 에이전트들을 초기화하고 설정"""
    cli_config = get_cli_agents_config()

    if not cli_config['enabled']:
        return False

    try:
        from src.core.cli_agents.cli_agent_manager import get_cli_agent_manager
        cli_manager = get_cli_agent_manager()

        # 각 CLI 에이전트 설정
        for agent_name, agent_config in cli_config['agents'].items():
            if agent_config['enabled']:
                cli_manager.configure_agent(agent_name, agent_config)

        return True

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize CLI agents: {e}")
        return False


def get_agent_config() -> AgentConfig:
    """Get agent configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.agent


def get_research_config() -> ResearchConfig:
    """Get research configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.research


def get_mcp_config() -> MCPConfig:
    """Get MCP configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.mcp


def get_output_config() -> OutputConfig:
    """Get output configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.output


def get_compression_config() -> CompressionConfig:
    """Get compression configuration (혁신 2)."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.compression


def get_verification_config() -> VerificationConfig:
    """Get verification configuration (혁신 4)."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.verification


def get_context_window_config() -> ContextWindowConfig:
    """Get context window configuration (혁신 7)."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.context_window


def get_reliability_config() -> ReliabilityConfig:
    """Get reliability configuration (혁신 8)."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.reliability


def get_cascade_config() -> CascadeConfig:
    """Get Cascade configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.cascade


def get_council_config() -> CouncilConfig:
    """Get council configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.council


def get_era_config() -> ERAConfig:
    """Get ERA configuration."""
    return ERAConfig.from_env()


def get_prompt_refiner_config() -> PromptRefinerConfig:
    """Get prompt refiner configuration."""
    if config is None:
        raise RuntimeError("Configuration not loaded. Call load_config_from_env() first.")
    return config.prompt_refiner


def load_config_from_env() -> ResearcherSystemConfig:
    """Load configuration from environment variables - ALL REQUIRED, NO DEFAULTS."""
    
    # Load .env file if it exists
    from pathlib import Path
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = lambda: None
    
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    def get_required_env(key: str, var_type: type = str):
        """Get required environment variable, raise error if missing."""
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        
        if var_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif var_type == int:
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"Environment variable {key} must be an integer, got: {value}")
        elif var_type == float:
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Environment variable {key} must be a float, got: {value}")
        return value
    
    def get_required_list_env(key: str, separator: str = ","):
        """Get required environment variable as list."""
        value = get_required_env(key)
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def get_optional_env(key: str, default_value: Any = None, var_type: type = str):
        """Get optional environment variable with default value."""
        value = os.getenv(key)
        if value is None:
            return default_value
        
        if var_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif var_type == int:
            try:
                return int(value)
            except ValueError:
                return default_value
        elif var_type == float:
            try:
                return float(value)
            except ValueError:
                return default_value
        return value
    
    def get_optional_list_env(key: str, default_value: List[str] = None, separator: str = ","):
        """Get optional environment variable as list with default."""
        if default_value is None:
            default_value = []
        value = os.getenv(key)
        if value is None:
            return default_value
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    # Load LLM configuration
    llm_config = LLMConfig(
        provider=get_required_env("LLM_PROVIDER"),
        primary_model=get_required_env("LLM_MODEL"),
        temperature=get_required_env("LLM_TEMPERATURE", float),
        max_tokens=get_required_env("LLM_MAX_TOKENS", int),
        api_key=get_required_env("GOOGLE_API_KEY"),
        planning_model=get_required_env("PLANNING_MODEL"),
        reasoning_model=get_required_env("REASONING_MODEL"),
        verification_model=get_required_env("VERIFICATION_MODEL"),
        generation_model=get_required_env("GENERATION_MODEL"),
        compression_model=get_required_env("COMPRESSION_MODEL"),
        openrouter_api_key=get_required_env("OPENROUTER_API_KEY"),
        budget_limit=get_required_env("BUDGET_LIMIT", float),
        enable_cost_optimization=get_required_env("ENABLE_COST_OPTIMIZATION", bool)
    )
    
    # Load Agent configuration
    agent_config = AgentConfig(
        max_retries=get_required_env("AGENT_MAX_RETRIES", int),
        timeout_seconds=get_required_env("AGENT_TIMEOUT", int),
        enable_self_planning=get_required_env("ENABLE_SELF_PLANNING", bool),
        enable_agent_communication=get_required_env("ENABLE_AGENT_COMMUNICATION", bool),
        max_concurrent_research_units=get_required_env("MAX_CONCURRENT_RESEARCH_UNITS", int),
        min_researchers=get_required_env("MIN_RESEARCHERS", int),
        max_researchers=get_required_env("MAX_RESEARCHERS", int),
        enable_fast_track=get_required_env("ENABLE_FAST_TRACK", bool),
        enable_auto_retry=get_required_env("ENABLE_AUTO_RETRY", bool),
        priority_queue_enabled=get_required_env("PRIORITY_QUEUE_ENABLED", bool),
        enable_quality_monitoring=get_required_env("ENABLE_QUALITY_MONITORING", bool),
        quality_threshold=get_required_env("QUALITY_THRESHOLD", float)
    )
    
    # Load Research configuration
    research_config = ResearchConfig(
        max_sources=get_required_env("MAX_SOURCES", int),
        search_timeout=get_required_env("SEARCH_TIMEOUT", int),
        enable_academic_search=get_required_env("ENABLE_ACADEMIC_SEARCH", bool),
        enable_web_search=get_required_env("ENABLE_WEB_SEARCH", bool),
        enable_browser_automation=get_required_env("ENABLE_BROWSER_AUTOMATION", bool),
        enable_streaming=get_required_env("ENABLE_STREAMING", bool),
        stream_chunk_size=get_required_env("STREAM_CHUNK_SIZE", int),
        enable_progressive_reporting=get_required_env("ENABLE_PROGRESSIVE_REPORTING", bool),
        enable_incremental_save=get_required_env("ENABLE_INCREMENTAL_SAVE", bool),
        enable_parallel_compression=get_required_env("ENABLE_PARALLEL_COMPRESSION", bool),
        enable_parallel_verification=get_required_env("ENABLE_PARALLEL_VERIFICATION", bool)
    )
    
    # Load MCP configuration
    mcp_config = MCPConfig(
        enabled=get_required_env("MCP_ENABLED", bool),
        server_names=get_required_list_env("MCP_SERVER_NAMES"),
        connection_timeout=get_required_env("MCP_TIMEOUT", int),
        enable_plugin_architecture=get_required_env("ENABLE_PLUGIN_ARCHITECTURE", bool),
        enable_smart_tool_selection=get_required_env("ENABLE_SMART_TOOL_SELECTION", bool),
        enable_auto_fallback=get_required_env("ENABLE_AUTO_FALLBACK", bool),
        search_tools=get_required_list_env("MCP_SEARCH_TOOLS"),
        data_tools=get_required_list_env("MCP_DATA_TOOLS"),
        code_tools=get_required_list_env("MCP_CODE_TOOLS"),
        academic_tools=get_required_list_env("MCP_ACADEMIC_TOOLS"),
        business_tools=get_required_list_env("MCP_BUSINESS_TOOLS"),
        builder_enabled=get_optional_env("MCP_BUILDER_ENABLED", True, bool),
        builder_temp_dir=get_optional_env("MCP_BUILDER_TEMP_DIR", "temp/mcp_servers"),
        builder_auto_cleanup=get_optional_env("MCP_BUILDER_AUTO_CLEANUP", True, bool),
        builder_cache_enabled=get_optional_env("MCP_BUILDER_CACHE_ENABLED", True, bool)
    )
    
    # Load Compression configuration
    compression_config = CompressionConfig(
        enabled=get_required_env("ENABLE_HIERARCHICAL_COMPRESSION", bool),
        enable_hierarchical_compression=get_required_env("ENABLE_HIERARCHICAL_COMPRESSION", bool),
        compression_levels=get_required_env("COMPRESSION_LEVELS", int),
        preserve_important_info=get_required_env("PRESERVE_IMPORTANT_INFO", bool),
        enable_compression_validation=get_required_env("ENABLE_COMPRESSION_VALIDATION", bool),
        compression_history_enabled=get_required_env("COMPRESSION_HISTORY_ENABLED", bool),
        min_compression_ratio=get_required_env("MIN_COMPRESSION_RATIO", float),
        target_compression_ratio=get_required_env("TARGET_COMPRESSION_RATIO", float)
    )
    
    # Load Verification configuration
    verification_config = VerificationConfig(
        enabled=get_required_env("ENABLE_CONTINUOUS_VERIFICATION", bool),
        enable_continuous_verification=get_required_env("ENABLE_CONTINUOUS_VERIFICATION", bool),
        verification_stages=get_required_env("VERIFICATION_STAGES", int),
        confidence_threshold=get_required_env("CONFIDENCE_THRESHOLD", float),
        enable_early_warning=get_required_env("ENABLE_EARLY_WARNING", bool),
        enable_fact_check=get_required_env("ENABLE_FACT_CHECK", bool),
        enable_uncertainty_marking=get_required_env("ENABLE_UNCERTAINTY_MARKING", bool)
    )
    
    # Load Context Window configuration
    context_window_config = ContextWindowConfig(
        enabled=get_required_env("ENABLE_ADAPTIVE_CONTEXT", bool),
        enable_adaptive_context=get_required_env("ENABLE_ADAPTIVE_CONTEXT", bool),
        min_tokens=get_required_env("MIN_TOKENS", int),
        max_tokens=get_required_env("MAX_TOKENS", int),
        importance_based_preservation=get_required_env("IMPORTANCE_BASED_PRESERVATION", bool),
        enable_auto_compression=get_required_env("ENABLE_AUTO_COMPRESSION", bool),
        # Force disable long term memory to prevent stale context issues
        enable_long_term_memory=False, # get_required_env("ENABLE_LONG_TERM_MEMORY", bool),
        memory_refresh_interval=get_required_env("MEMORY_REFRESH_INTERVAL", int)
    )
    
    # Load Reliability configuration
    reliability_config = ReliabilityConfig(
        enabled=get_required_env("ENABLE_PRODUCTION_RELIABILITY", bool),
        enable_circuit_breaker=get_required_env("ENABLE_CIRCUIT_BREAKER", bool),
        enable_exponential_backoff=get_required_env("ENABLE_EXPONENTIAL_BACKOFF", bool),
        # Force disable state persistence
        enable_state_persistence=False, # get_required_env("ENABLE_STATE_PERSISTENCE", bool),
        enable_health_check=get_required_env("ENABLE_HEALTH_CHECK", bool),
        enable_graceful_degradation=get_required_env("ENABLE_GRACEFUL_DEGRADATION", bool),
        enable_detailed_logging=get_required_env("ENABLE_DETAILED_LOGGING", bool),
        failure_threshold=get_required_env("FAILURE_THRESHOLD", int),
        recovery_timeout=get_required_env("RECOVERY_TIMEOUT", int),
        state_backend=get_required_env("STATE_BACKEND"),
        state_ttl=get_required_env("STATE_TTL", int)
    )

    # Load Agent Tool configuration
    agent_tool_config = AgentToolConfig(
        planner_servers=get_required_list_env("AGENT_TOOL_PLANNER_SERVERS") if os.getenv("AGENT_TOOL_PLANNER_SERVERS") else [],
        executor_servers=get_required_list_env("AGENT_TOOL_EXECUTOR_SERVERS") if os.getenv("AGENT_TOOL_EXECUTOR_SERVERS") else [],
        verifier_servers=get_required_list_env("AGENT_TOOL_VERIFIER_SERVERS") if os.getenv("AGENT_TOOL_VERIFIER_SERVERS") else [],
        generator_servers=get_required_list_env("AGENT_TOOL_GENERATOR_SERVERS") if os.getenv("AGENT_TOOL_GENERATOR_SERVERS") else [],
        planner_categories=get_required_list_env("AGENT_TOOL_PLANNER_CATEGORIES") if os.getenv("AGENT_TOOL_PLANNER_CATEGORIES") else ["planning", "search", "utility"],
        executor_categories=get_required_list_env("AGENT_TOOL_EXECUTOR_CATEGORIES") if os.getenv("AGENT_TOOL_EXECUTOR_CATEGORIES") else ["search", "data", "academic", "business", "code"],
        verifier_categories=get_required_list_env("AGENT_TOOL_VERIFIER_CATEGORIES") if os.getenv("AGENT_TOOL_VERIFIER_CATEGORIES") else ["verification", "search", "data", "academic"],
        generator_categories=get_required_list_env("AGENT_TOOL_GENERATOR_CATEGORIES") if os.getenv("AGENT_TOOL_GENERATOR_CATEGORIES") else ["generation", "utility", "search"],
        max_tools_per_agent=get_required_env("AGENT_TOOL_MAX_PER_AGENT", int) if os.getenv("AGENT_TOOL_MAX_PER_AGENT") else 5,
        enable_auto_discovery=get_required_env("AGENT_TOOL_ENABLE_AUTO_DISCOVERY", bool) if os.getenv("AGENT_TOOL_ENABLE_AUTO_DISCOVERY") else True,
        enable_cross_agent_tools=get_required_env("AGENT_TOOL_ENABLE_CROSS_AGENT", bool) if os.getenv("AGENT_TOOL_ENABLE_CROSS_AGENT") else True,
        cross_agent_timeout=get_required_env("AGENT_TOOL_CROSS_AGENT_TIMEOUT", float) if os.getenv("AGENT_TOOL_CROSS_AGENT_TIMEOUT") else 30.0
    )
    
    # Load Output configuration
    output_config = OutputConfig(
        output_dir=get_required_env("OUTPUT_DIR"),
        enable_pdf_generation=get_required_env("ENABLE_PDF", bool),
        enable_markdown_generation=get_required_env("ENABLE_MARKDOWN", bool),
        enable_json_export=get_required_env("ENABLE_JSON", bool),
        enable_docx_export=get_required_env("ENABLE_DOCX", bool),
        enable_html_export=get_required_env("ENABLE_HTML", bool),
        enable_latex_export=get_required_env("ENABLE_LATEX", bool)
    )
    
    # Load Council configuration (Optional with defaults)
    # OpenRouter API 키가 없으면 Council 비활성화
    openrouter_api_key = get_optional_env("OPENROUTER_API_KEY")
    council_enabled = get_optional_env("COUNCIL_ENABLED", True, bool) if openrouter_api_key else False
    
    council_config = CouncilConfig(
        enabled=council_enabled,
        auto_activate=get_optional_env("COUNCIL_AUTO_ACTIVATE", True, bool),
        council_models=get_optional_list_env(
            "COUNCIL_MODELS",
            ["google/gemini-2.5-flash-lite", "openai/gpt-4o-mini", "anthropic/claude-3-haiku"]
        ),
        chairman_model=get_optional_env("COUNCIL_CHAIRMAN_MODEL", "google/gemini-2.5-flash-lite"),
        min_complexity_for_auto=get_optional_env("COUNCIL_MIN_COMPLEXITY", 0.6, float),
        enable_for_planning=get_optional_env("COUNCIL_ENABLE_FOR_PLANNING", True, bool),
        enable_for_execution=get_optional_env("COUNCIL_ENABLE_FOR_EXECUTION", True, bool),
        enable_for_evaluation=get_optional_env("COUNCIL_ENABLE_FOR_EVALUATION", True, bool),
        enable_for_verification=get_optional_env("COUNCIL_ENABLE_FOR_VERIFICATION", True, bool),
        enable_for_synthesis=get_optional_env("COUNCIL_ENABLE_FOR_SYNTHESIS", True, bool),
        openrouter_api_key=openrouter_api_key,
        openrouter_api_url=get_optional_env("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
        request_timeout=get_optional_env("COUNCIL_REQUEST_TIMEOUT", 120.0, float)
    )
    
    # Load Cascade configuration (Optional with defaults)
    cascade_config = CascadeConfig(
        enabled=get_optional_env("CASCADE_ENABLED", True, bool),
        confidence_threshold=get_optional_env("CASCADE_CONFIDENCE_THRESHOLD", 0.75, float),
        min_models_for_cascade=get_optional_env("CASCADE_MIN_MODELS", 2, int),
        enable_adaptive_threshold=get_optional_env("CASCADE_ENABLE_ADAPTIVE_THRESHOLD", True, bool),
        drafter_cost_threshold=get_optional_env("CASCADE_DRAFTER_COST_THRESHOLD", 0.0002, float),
        verifier_quality_threshold=get_optional_env("CASCADE_VERIFIER_QUALITY_THRESHOLD", 8.0, float),
        drafter_speed_threshold=get_optional_env("CASCADE_DRAFTER_SPEED_THRESHOLD", 7.0, float)
    )
    
    # Load PromptRefiner configuration (Optional with defaults)
    prompt_refiner_config = PromptRefinerConfig(
        enabled=get_optional_env("PROMPT_REFINER_ENABLED", True, bool),
        strategy=get_optional_env("PROMPT_REFINER_STRATEGY", "aggressive"),
        max_tokens=get_optional_env("PROMPT_REFINER_MAX_TOKENS", None, int) if get_optional_env("PROMPT_REFINER_MAX_TOKENS") else None,
        collect_stats=get_optional_env("PROMPT_REFINER_COLLECT_STATS", True, bool)
    )
    
    # Load Overseer configuration (Optional with defaults)
    overseer_config = OverseerConfig(
        enabled=get_optional_env("OVERSEER_ENABLED", True, bool),
        max_iterations=get_optional_env("OVERSEER_MAX_ITERATIONS", 5, int),
        completeness_threshold=get_optional_env("OVERSEER_COMPLETENESS_THRESHOLD", 0.9, float),
        quality_threshold=get_optional_env("OVERSEER_QUALITY_THRESHOLD", 0.85, float),
        min_academic_sources=get_optional_env("OVERSEER_MIN_ACADEMIC_SOURCES", 3, int),
        min_verified_sources=get_optional_env("OVERSEER_MIN_VERIFIED_SOURCES", 5, int),
        require_cross_validation=get_optional_env("OVERSEER_REQUIRE_CROSS_VALIDATION", True, bool),
        enable_human_loop=get_optional_env("OVERSEER_ENABLE_HUMAN_LOOP", True, bool)
    )
    
    # Create and store global config instance
    global config
    config = ResearcherSystemConfig(
        llm=llm_config,
        agent=agent_config,
        research=research_config,
        mcp=mcp_config,
        output=output_config,
        compression=compression_config,
        verification=verification_config,
        context_window=context_window_config,
        reliability=reliability_config,
        council=council_config,
        cascade=cascade_config,
        agent_tools=agent_tool_config,
        prompt_refiner=prompt_refiner_config,
        overseer=overseer_config
    )
    
    return config


# Note: AgentToolConfig is already defined above (line 278)
# This duplicate definition has been removed to avoid conflicts


def get_required_list_env(key: str) -> List[str]:
    """환경 변수에서 콤마로 구분된 리스트를 가져옵니다."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"필수 환경 변수가 설정되지 않음: {key}")
    return [item.strip() for item in value.split(',') if item.strip()]


# Configuration is loaded via load_config_from_env() function
