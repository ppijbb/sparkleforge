"""
Multi-Model Orchestration (혁신 3)

역할 기반 모델 선택, 동적 모델 전환, 비용 최적화, Weighted Ensemble을 통한
다중 모델 오케스트레이션 시스템.
"""

import asyncio
import time
import logging
import requests
import os
from typing import Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from src.core.researcher_config import get_llm_config, get_agent_config, get_cascade_config
from src.core.reliability import execute_with_reliability
from src.core.prompt_refiner_wrapper import refine_llm_call

logger = logging.getLogger(__name__)

# Safety settings to block nothing (allow all content)
# This is required for the research agent to function without being blocked by safety filters
# for harmless queries or research topics.
# Note: ChatGoogleGenerativeAI does not support safety_settings parameter directly.
# Safety settings are handled at the genai.GenerativeModel level, not in LangChain wrapper.
# Setting to None to avoid validation errors.
SAFETY_SETTINGS_BLOCK_NONE = None


class TaskType(Enum):
    """작업 유형."""
    PLANNING = "planning"
    DEEP_REASONING = "deep_reasoning"
    VERIFICATION = "verification"
    GENERATION = "generation"
    COMPRESSION = "compression"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    CREATIVE = "creative"
    MEMORY_EXTRACTION = "memory_extraction"  # 메모리 추출 (백서 요구사항)
    MEMORY_CONSOLIDATION = "memory_consolidation"  # 메모리 통합 (백서 요구사항)


class Provider(Enum):
    """LLM 제공자."""
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    OPENAI = "openai"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """모델 설정."""
    name: str
    provider: str
    model_id: str
    temperature: float
    max_tokens: int
    cost_per_token: float
    speed_rating: float  # 1-10, 높을수록 빠름
    quality_rating: float  # 1-10, 높을수록 품질 좋음
    capabilities: List[TaskType]


@dataclass
class ModelResult:
    """모델 실행 결과."""
    content: str
    model_used: str
    execution_time: float
    confidence: float
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectionPool:
    """LLM Connection Pool for performance optimization."""
    
    def __init__(self, pool_size: int = 5, max_idle_time: int = 300):
        """
        Initialize connection pool.
        
        Args:
            pool_size: Maximum number of connections per model
            max_idle_time: Maximum idle time before connection cleanup (seconds)
        """
        self.pool_size = pool_size
        self.max_idle_time = max_idle_time
        self.pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.connection_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"created": 0, "reused": 0})
        self.last_cleanup = time.time()
    
    def get_connection(self, model_name: str, provider: str, create_func: Callable) -> Any:
        """
        Get or create a connection from the pool.
        
        Args:
            model_name: Model name
            provider: Provider name  
            create_func: Function to create new connection
            
        Returns:
            Connection object
        """
        pool_key = f"{provider}:{model_name}"
        current_time = time.time()
        
        # Cleanup old connections periodically
        if current_time - self.last_cleanup > 60:  # Every minute
            self._cleanup_old_connections()
            self.last_cleanup = current_time
        
        # Check for available connection
        if pool_key in self.pools and self.pools[pool_key]:
            conn_info = self.pools[pool_key].pop()
            if current_time - conn_info["created_at"] < self.max_idle_time:
                self.connection_stats[pool_key]["reused"] += 1
                logger.debug(f"Reusing connection for {pool_key}")
                return conn_info["connection"]
        
        # Create new connection
        connection = create_func()
        self.connection_stats[pool_key]["created"] += 1
        logger.debug(f"Created new connection for {pool_key}")
        
        return connection
    
    def return_connection(self, model_name: str, provider: str, connection: Any) -> None:
        """
        Return a connection to the pool.
        
        Args:
            model_name: Model name
            provider: Provider name
            connection: Connection object to return
        """
        pool_key = f"{provider}:{model_name}"
        
        if len(self.pools[pool_key]) < self.pool_size:
            self.pools[pool_key].append({
                "connection": connection,
                "created_at": time.time()
            })
        # If pool is full, let connection be garbage collected
    
    def _cleanup_old_connections(self) -> None:
        """Remove old idle connections from all pools."""
        current_time = time.time()
        removed_count = 0
        
        for pool_key, connections in list(self.pools.items()):
            valid_connections = []
            for conn_info in connections:
                if current_time - conn_info["created_at"] < self.max_idle_time:
                    valid_connections.append(conn_info)
                else:
                    removed_count += 1
            
            if valid_connections:
                self.pools[pool_key] = valid_connections
            else:
                del self.pools[pool_key]
        
        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} old connections")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "pool_size": self.pool_size,
            "active_pools": len(self.pools),
            "total_connections": sum(len(conns) for conns in self.pools.values()),
            "connection_stats": dict(self.connection_stats)
        }


class ModelPerformanceTracker:
    """모델 성능 추적기."""
    
    def __init__(self):
        self.performance_stats: Dict[str, Dict[str, Any]] = {}
    
    def record_execution(
        self,
        model_name: str,
        task_type: TaskType,
        execution_time: float,
        success: bool,
        quality_score: float = None
    ):
        """실행 기록."""
        if model_name not in self.performance_stats:
            self.performance_stats[model_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time": 0.0,
                "avg_quality": 0.0,
                "task_performance": {}
            }
        
        stats = self.performance_stats[model_name]
        stats["total_executions"] += 1
        stats["total_time"] += execution_time
        
        if success:
            stats["successful_executions"] += 1
        
        if quality_score is not None:
            current_avg = stats["avg_quality"]
            total = stats["successful_executions"]
            stats["avg_quality"] = (current_avg * (total - 1) + quality_score) / total
        
        # 작업별 성능 추적
        task_key = task_type.value
        if task_key not in stats["task_performance"]:
            stats["task_performance"][task_key] = {
                "executions": 0,
                "successes": 0,
                "avg_time": 0.0,
                "avg_quality": 0.0
            }
        
        task_stats = stats["task_performance"][task_key]
        task_stats["executions"] += 1
        if success:
            task_stats["successes"] += 1
        
        # 평균 시간 업데이트
        current_avg_time = task_stats["avg_time"]
        task_stats["avg_time"] = (current_avg_time * (task_stats["executions"] - 1) + execution_time) / task_stats["executions"]
        
        # 평균 품질 업데이트
        if quality_score is not None and success:
            current_avg_quality = task_stats["avg_quality"]
            task_stats["avg_quality"] = (current_avg_quality * (task_stats["successes"] - 1) + quality_score) / task_stats["successes"]
    
    def get_model_score(self, model_name: str, task_type: TaskType = None) -> float:
        """모델 점수 반환."""
        if model_name not in self.performance_stats:
            return 0.0
        
        stats = self.performance_stats[model_name]
        
        if task_type:
            task_key = task_type.value
            if task_key not in stats["task_performance"]:
                return 0.0
            
            task_stats = stats["task_performance"][task_key]
            if task_stats["executions"] == 0:
                return 0.0
            
            success_rate = task_stats["successes"] / task_stats["executions"]
            avg_quality = task_stats["avg_quality"]
            speed_score = 1.0 / (1.0 + task_stats["avg_time"])  # 시간이 짧을수록 높은 점수
            
            return (success_rate * 0.4 + avg_quality * 0.4 + speed_score * 0.2)
        else:
            # 전체 성능 점수
            if stats["total_executions"] == 0:
                return 0.0
            
            success_rate = stats["successful_executions"] / stats["total_executions"]
            avg_quality = stats["avg_quality"]
            avg_time = stats["total_time"] / stats["total_executions"]
            speed_score = 1.0 / (1.0 + avg_time)
            
            return (success_rate * 0.4 + avg_quality * 0.4 + speed_score * 0.2)


class MultiModelOrchestrator:
    """다중 모델 오케스트레이터 (혁신 3)."""
    
    def __init__(self):
        self.llm_config = get_llm_config()
        self.agent_config = get_agent_config()
        
        # Provider별 API 키 검증
        self._validate_provider_config()
        
        self.models: Dict[str, ModelConfig] = {}
        self.performance_tracker = ModelPerformanceTracker()
        self.model_clients: Dict[str, Any] = {}
        
        # Connection pooling for performance optimization
        self.connection_pool = ConnectionPool(pool_size=5, max_idle_time=300)
        
        # Provider 로테이션 추적
        self.provider_rotation_index = 0  # 현재 Provider 인덱스
        self.provider_rotation_order = ["openrouter", "groq", "cerebras"]  # 로테이션 순서
        self.provider_rate_limited = {}  # Rate limit에 걸린 Provider (timestamp)
        self.provider_usage_count = {provider: 0 for provider in self.provider_rotation_order}  # 사용 횟수 추적
        
        self._initialize_models()
        self._initialize_clients()
    
    def _validate_provider_config(self):
        """Provider별 API 키 검증."""
        # API 키가 없어도 경고만 출력 (폴백 메커니즘 사용)
        if not os.getenv("OPENROUTER_API_KEY"):
            logger.warning("OPENROUTER_API_KEY not found - OpenRouter models will be unavailable")
        if not os.getenv("GROQ_API_KEY"):
            logger.warning("GROQ_API_KEY not found - Groq models will be unavailable")
        if not self.llm_config.api_key:
            logger.warning("GOOGLE_API_KEY not found - Gemini models will be unavailable")
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found - GPT models will be unavailable")
    
    def _initialize_models(self):
        """모델 초기화."""
        # Gemini 2.5 Flash Lite (빠른 계획, 압축)
        self.models["gemini-flash-lite"] = ModelConfig(
            name="gemini-flash-lite",
            provider="google",
            model_id="gemini-2.5-flash-lite",
            temperature=0.1,
            max_tokens=4000,
            cost_per_token=0.0001,
            speed_rating=9.0,
            quality_rating=7.0,
            capabilities=[TaskType.PLANNING, TaskType.COMPRESSION, TaskType.RESEARCH]
        )
        
        # Gemini 2.5 Pro (복잡한 추론, 분석)
        self.models["gemini-pro"] = ModelConfig(
            name="gemini-pro",
            provider="google",
            model_id="gemini-2.5-pro",
            temperature=0.2,
            max_tokens=8000,
            cost_per_token=0.0005,
            speed_rating=6.0,
            quality_rating=9.0,
            capabilities=[TaskType.DEEP_REASONING, TaskType.ANALYSIS, TaskType.SYNTHESIS]
        )
        
        # Gemini 2.5 Flash (균형잡힌 성능)
        self.models["gemini-flash"] = ModelConfig(
            name="gemini-flash",
            provider="google",
            model_id="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=4000,
            cost_per_token=0.0002,
            speed_rating=8.0,
            quality_rating=8.0,
            capabilities=[TaskType.GENERATION, TaskType.VERIFICATION, TaskType.RESEARCH]
        )
        
        # 모델 로딩: 우선순위에 따라 로드
        # 1. OpenRouter 모델 (최우선)
        if os.getenv("OPENROUTER_API_KEY"):
            try:
                self._load_openrouter_models()
                logger.info("✅ OpenRouter models loaded")
            except Exception as e:
                logger.warning(f"OpenRouter models not loaded: {e}")
        else:
            logger.info("OpenRouter disabled - OPENROUTER_API_KEY not found")
        
        # 2. Groq 모델
        if os.getenv("GROQ_API_KEY"):
            try:
                self._load_groq_models()
                logger.info("✅ Groq models loaded")
            except Exception as e:
                logger.warning(f"Groq models not loaded: {e}")
        else:
            logger.info("Groq disabled - GROQ_API_KEY not found")
        
        # 3. Gemini 모델 (이미 위에서 로드됨)
        logger.info("✅ Gemini models loaded")
        
        # 4. GPT 모델
        if os.getenv("OPENAI_API_KEY"):
            try:
                self._load_openai_models()
                logger.info("✅ OpenAI/GPT models loaded")
            except Exception as e:
                logger.warning(f"OpenAI/GPT models not loaded: {e}")
        else:
            logger.info("OpenAI/GPT disabled - OPENAI_API_KEY not found")
    
    def _load_openrouter_models(self):
        """OpenRouter API에서 무료 모델들을 동적으로 로드 (선택적)."""
        try:
            openrouter_models = self._fetch_openrouter_models()
            for model_data in openrouter_models:
                if self._is_free_model(model_data):
                    model_name = self._generate_model_name(model_data)
                    model_config = self._create_model_config(model_data, model_name)
                    self.models[model_name] = model_config
                    logger.info(f"Loaded OpenRouter model: {model_name} ({model_data['id']})")
        except Exception as e:
            logger.warning(f"Failed to load OpenRouter models: {e}")
            # 예외를 raise하지 않음 - Gemini만 사용하도록 함
    
    def _fetch_openrouter_models(self):
        """OpenRouter API에서 모델 목록을 가져옴."""
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        else:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
    
    def _is_free_model(self, model_data):
        """모델이 무료인지 확인."""
        model_id = model_data.get("id", "").lower()
        
        # 사용 중단된 모델 필터링
        deprecated_models = [
            "kwaipilot/kat-coder-pro:free",
            "mistralai/mistral-7b-instruct:free",
            "qwen/qwen3-4b:free",
            "google/gemma-3n-e2b-it:free",
            "google/gemma-3-4b-it:free",
            "meta-llama/llama-3.2-3b-instruct:free",
        ]
        if any(deprecated in model_id for deprecated in deprecated_models):
            logger.debug(f"Skipping deprecated model: {model_id}")
            return False
        
        pricing = model_data.get("pricing", {})
        prompt_price = pricing.get("prompt", "0")
        completion_price = pricing.get("completion", "0")
        
        # 무료 모델 조건: prompt와 completion 가격이 모두 0이거나 매우 낮음
        try:
            prompt_cost = float(prompt_price) if prompt_price else 0
            completion_cost = float(completion_price) if completion_price else 0
            return prompt_cost == 0 and completion_cost == 0
        except (ValueError, TypeError):
            return False
    
    def _generate_model_name(self, model_data):
        """모델 데이터에서 고유한 이름 생성."""
        model_id = model_data.get("id", "")
        # "provider/model-name:tag" 형식을 "provider-model-name-tag"로 변환
        name = model_id.replace("/", "-").replace(":", "-").replace("_", "-")
        return name
    
    def _create_model_config(self, model_data, model_name):
        """모델 데이터에서 ModelConfig 생성."""
        model_id = model_data.get("id", "")
        context_length = model_data.get("context_length", 4000)
        
        # 모델별 기본 설정
        capabilities = self._determine_capabilities(model_id)
        speed_rating = self._estimate_speed_rating(model_id)
        quality_rating = self._estimate_quality_rating(model_id)
        
        return ModelConfig(
            name=model_name,
            provider="openrouter",
            model_id=model_id,
            temperature=0.1,
            max_tokens=min(context_length, 4000),
            cost_per_token=0.0,  # 무료 모델
            speed_rating=speed_rating,
            quality_rating=quality_rating,
            capabilities=capabilities
        )
    
    def _determine_capabilities(self, model_id):
        """모델 ID를 기반으로 capabilities 결정."""
        capabilities = [TaskType.GENERATION]  # 기본
        
        # 모델명에 따른 capabilities 추가
        if any(keyword in model_id.lower() for keyword in ["reasoning", "reason", "think"]):
            capabilities.extend([TaskType.DEEP_REASONING, TaskType.ANALYSIS])
        
        if any(keyword in model_id.lower() for keyword in ["code", "coder", "programming"]):
            capabilities.extend([TaskType.RESEARCH, TaskType.COMPRESSION])
        
        if any(keyword in model_id.lower() for keyword in ["verify", "check", "validate"]):
            capabilities.extend([TaskType.VERIFICATION])
        
        if any(keyword in model_id.lower() for keyword in ["plan", "planning", "strategy"]):
            capabilities.append(TaskType.PLANNING)
        
        # 기본적으로 모든 작업 가능
        if not any(keyword in model_id.lower() for keyword in ["reasoning", "code", "verify", "plan"]):
            capabilities = [TaskType.PLANNING, TaskType.DEEP_REASONING, TaskType.VERIFICATION, 
                          TaskType.GENERATION, TaskType.COMPRESSION, TaskType.RESEARCH]
        
        return list(set(capabilities))  # 중복 제거
    
    def _estimate_speed_rating(self, model_id):
        """모델 ID를 기반으로 속도 등급 추정."""
        if any(keyword in model_id.lower() for keyword in ["flash", "fast", "lite", "small"]):
            return 8.0
        elif any(keyword in model_id.lower() for keyword in ["large", "big", "70b", "72b"]):
            return 6.0
        else:
            return 7.0
    
    def _estimate_quality_rating(self, model_id):
        """모델 ID를 기반으로 품질 등급 추정."""
        if any(keyword in model_id.lower() for keyword in ["70b", "72b", "large", "pro"]):
            return 9.0
        elif any(keyword in model_id.lower() for keyword in ["27b", "medium"]):
            return 7.5
        else:
            return 8.0
    
    def _get_valid_openrouter_model_id(self, model_id: str, model_name: str) -> str:
        """OpenRouter에 실제 존재하는 모델 ID 반환."""
        # 이미 OpenRouter 형식인 경우 (provider/model:tag)
        if "/" in model_id or ":" in model_id:
            # OpenRouter 형식 확인
            if model_id.startswith(("google/", "openai/", "anthropic/", "meta-llama/", 
                                   "mistralai/", "cerebras/", "groq/", "moonshotai/")):
                return model_id
        
        # Google 모델 ID를 OpenRouter 형식으로 변환
        google_to_openrouter = {
            "gemini-2.5-flash-lite": "google/gemini-2.0-flash-exp:free",
            "gemini-2.5-flash": "google/gemini-2.0-flash-exp:free",
            "gemini-2.5-pro": "google/gemini-2.0-flash-thinking-exp:free",
            "gemini-flash-lite": "google/gemini-2.0-flash-exp:free",
            "gemini-flash": "google/gemini-2.0-flash-exp:free",
            "gemini-pro": "google/gemini-2.0-flash-thinking-exp:free",
        }
        
        if model_id in google_to_openrouter:
            return google_to_openrouter[model_id]
        
        # 모델 이름에서 추론
        if "gemini" in model_name.lower() or "gemini" in model_id.lower():
            # 기본적으로 무료 Gemini 모델 사용
            return "google/gemini-2.0-flash-exp:free"
        
        # 최소 Fallback 정책: LLM 모델 요청 실패 시에만 fallback 사용
        # Fallback은 Agent 서비스 안정성을 위해 필수적이지만, 명확한 로깅과 함께 최소한으로만 사용됩니다.
        fallback_models = [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.2-3b-instruct:free",
        ]
        
        logger.warning(
            f"Model ID {model_id} not in OpenRouter format. "
            f"Using minimal fallback for agent service stability: {fallback_models[0]}"
        )
        return fallback_models[0]
    
    def _get_valid_groq_model_id(self, model_id: str) -> str:
        """Groq에 실제 존재하는 모델 ID 반환."""
        # Groq에 실제 존재하는 모델 목록
        valid_groq_models = [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "llama-3.3-70b-versatile",
            "llama-3.2-90b-vision-preview",
            "llama-3.2-11b-vision-preview",
            "llama-3.2-3b-text-preview",
            "llama-3.2-1b-text-preview",
        ]
        
        # 이미 유효한 Groq 모델 ID인 경우
        if model_id in valid_groq_models:
            return model_id
        
        # OpenRouter 형식인 경우 Groq 모델 ID 추출
        if "/" in model_id:
            parts = model_id.split("/")
            if len(parts) == 2:
                groq_id = parts[1].split(":")[0]  # 태그 제거
                if groq_id in valid_groq_models:
                    return groq_id
        
        # 최소 Fallback 정책: LLM 모델 요청 실패 시에만 fallback 사용
        # Fallback은 Agent 서비스 안정성을 위해 필수적이지만, 명확한 로깅과 함께 최소한으로만 사용됩니다.
        logger.warning(
            f"Model ID {model_id} not a valid Groq model. "
            f"Using minimal fallback for agent service stability: llama-3.1-8b-instant"
        )
        return "llama-3.1-8b-instant"
    
    def _load_groq_models(self):
        """Groq 모델 로딩."""
        # 주요 Groq 모델들
        # 참고: llama-3.1-70b-versatile과 llama-3.2-70b-versatile은 존재하지 않음
        # 실제 사용 가능한 모델: llama-3.1-70b-versatile (decommissioned), llama-3.1-8b-instant, mixtral-8x7b-32768
        # llama-3.1-70b-versatile 대체: mixtral-8x7b-32768 사용 (더 안정적)
        # 실제 Groq에 존재하는 모델만 사용
        groq_models = [
            {
                "name": "llama-3.1-8b-instant",
                "model_id": "llama-3.1-8b-instant",  # 실제 존재하는 모델
                "speed_rating": 9.5,
                "quality_rating": 8.0,
                "capabilities": [TaskType.GENERATION, TaskType.RESEARCH, TaskType.ANALYSIS]
            },
            {
                "name": "mixtral-8x7b-32768",
                "model_id": "mixtral-8x7b-32768",  # 실제 존재하는 모델
                "speed_rating": 9.0,
                "quality_rating": 8.5,
                "capabilities": [TaskType.GENERATION, TaskType.RESEARCH, TaskType.ANALYSIS, TaskType.PLANNING]
            },
            {
                "name": "llama-3.3-70b-versatile",
                "model_id": "llama-3.3-70b-versatile",  # 실제 존재하는 모델
                "speed_rating": 8.5,
                "quality_rating": 9.0,
                "capabilities": [TaskType.GENERATION, TaskType.RESEARCH, TaskType.ANALYSIS, TaskType.DEEP_REASONING]
            }
        ]
        
        for model_data in groq_models:
            self.models[model_data["name"]] = ModelConfig(
                name=model_data["name"],
                provider="groq",
                model_id=model_data["model_id"],
                temperature=0.1,
                max_tokens=4000,
                cost_per_token=0.0,  # Groq는 무료 티어 제공
                speed_rating=model_data["speed_rating"],
                quality_rating=model_data["quality_rating"],
                capabilities=model_data["capabilities"]
            )
            logger.info(f"Loaded Groq model: {model_data['name']} ({model_data['model_id']})")
    
    def _load_openai_models(self):
        """OpenAI/GPT 모델 로딩."""
        # 주요 GPT 모델들
        gpt_models = [
            {
                "name": "gpt-5.2-mini",
                "model_id": "gpt-5.2-mini",
                "speed_rating": 8.0,
                "quality_rating": 8.5,
                "capabilities": [TaskType.GENERATION, TaskType.VERIFICATION, TaskType.RESEARCH]
            },
            {
                "name": "gpt-5-nano",
                "model_id": "gpt-5-nano",
                "speed_rating": 7.0,
                "quality_rating": 9.5,
                "capabilities": [TaskType.DEEP_REASONING, TaskType.ANALYSIS, TaskType.SYNTHESIS]
            },
            {
                "name": "gpt-4o-mini",
                "model_id": "gpt-4o-mini",
                "speed_rating": 9.0,
                "quality_rating": 7.0,
                "capabilities": [TaskType.PLANNING, TaskType.COMPRESSION, TaskType.RESEARCH]
            }
        ]
        
        for model_data in gpt_models:
            self.models[model_data["name"]] = ModelConfig(
                name=model_data["name"],
                provider="openai",
                model_id=model_data["model_id"],
                temperature=0.1,
                max_tokens=4000,
                cost_per_token=0.0001,  # GPT는 유료
                speed_rating=model_data["speed_rating"],
                quality_rating=model_data["quality_rating"],
                capabilities=model_data["capabilities"]
            )
            logger.info(f"Loaded OpenAI/GPT model: {model_data['name']} ({model_data['model_id']})")
    
    def refresh_openrouter_models(self):
        """OpenRouter 모델 목록을 새로고침."""
        logger.info("Refreshing OpenRouter models...")
        # 기존 OpenRouter 모델들 제거
        openrouter_models = [name for name, config in self.models.items() 
                           if config.provider == "openrouter"]
        for model_name in openrouter_models:
            del self.models[model_name]
        
        # 새로 로드
        self._load_openrouter_models()
        logger.info(f"Refreshed OpenRouter models: {len([name for name, config in self.models.items() if config.provider == 'openrouter'])} models loaded")
    
    def _initialize_clients(self):
        """모델 클라이언트 초기화."""
        try:
            genai.configure(api_key=self.llm_config.api_key)
            
            for model_name, model_config in self.models.items():
                if model_config.provider == "google":
                    # Google Generative AI 클라이언트
                    self.model_clients[model_name] = genai.GenerativeModel(model_config.model_id)
                    
                    # LangChain 클라이언트 (선택적)
                    # Note: ChatGoogleGenerativeAI does not support safety_settings parameter
                    # Safety settings are handled at the genai.GenerativeModel level
                    self.model_clients[f"{model_name}_langchain"] = ChatGoogleGenerativeAI(
                        model=model_config.model_id,
                        temperature=model_config.temperature,
                        max_tokens=model_config.max_tokens,
                        google_api_key=self.llm_config.api_key
                    )
                
                elif model_config.provider == "openrouter":
                    # OpenRouter 클라이언트는 HTTP 요청으로 직접 처리
                    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
                    if not openrouter_api_key:
                        raise ValueError(f"OpenRouter API key not found for {model_name}")
                    # OpenRouter는 HTTP 요청으로 직접 처리하므로 클라이언트 저장하지 않음
                    logger.info(f"OpenRouter model {model_name} configured for HTTP requests")
                
                elif model_config.provider == "groq":
                    # Groq 클라이언트 초기화
                    try:
                        from groq import Groq
                        groq_api_key = os.getenv("GROQ_API_KEY")
                        if not groq_api_key:
                            raise ValueError(f"GROQ_API_KEY not found for {model_name}")
                        self.model_clients[model_name] = Groq(api_key=groq_api_key)
                        logger.info(f"Groq model {model_name} configured")
                    except ImportError:
                        logger.warning(f"groq library not installed. Install with: pip install groq")
                    except Exception as e:
                        logger.warning(f"Failed to initialize Groq client for {model_name}: {e}")
                
                elif model_config.provider == "openai":
                    # OpenAI 클라이언트 초기화
                    try:
                        from openai import OpenAI
                        openai_api_key = os.getenv("OPENAI_API_KEY")
                        if not openai_api_key:
                            raise ValueError(f"OPENAI_API_KEY not found for {model_name}")
                        self.model_clients[model_name] = OpenAI(api_key=openai_api_key)
                        logger.info(f"OpenAI/GPT model {model_name} configured")
                    except ImportError:
                        logger.warning(f"openai library not installed. Install with: pip install openai")
                    except Exception as e:
                        logger.warning(f"Failed to initialize OpenAI client for {model_name}: {e}")
            
            logger.info("Model clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model clients: {e}")
            raise
    
    def _is_provider_rate_limited(self, provider: str) -> bool:
        """Provider가 rate limit에 걸렸는지 확인 (5분 후 자동 해제)."""
        if provider not in self.provider_rate_limited:
            return False
        
        # Rate limit 타임스탬프 확인 (5분 = 300초)
        rate_limit_time = self.provider_rate_limited[provider]
        if time.time() - rate_limit_time > 300:
            # 5분 경과 시 자동 해제
            del self.provider_rate_limited[provider]
            logger.info(f"Provider {provider} rate limit automatically cleared after 5 minutes")
            return False
        
        return True
    
    def _mark_provider_rate_limited(self, provider: str):
        """Provider를 rate limit 상태로 표시."""
        self.provider_rate_limited[provider] = time.time()
        logger.warning(f"Provider {provider} marked as rate-limited (will retry after 5 minutes)")
    
    def _get_available_providers(self) -> List[str]:
        """Rate limit되지 않은 사용 가능한 Provider 목록 반환."""
        available = []
        for provider in self.provider_rotation_order:
            # Cerebras는 OpenRouter를 통해 접근하므로 openrouter로 처리
            check_provider = "openrouter" if provider == "cerebras" else provider
            
            # Rate limit 확인
            if not self._is_provider_rate_limited(check_provider):
                # Provider에 사용 가능한 모델이 있는지 확인
                has_models = any(
                    config.provider == check_provider 
                    for config in self.models.values()
                )
                if has_models:
                    available.append(provider)
        
        return available
    
    def select_model(
        self,
        task_type: TaskType,
        complexity: float = 5.0,
        budget: float = None
    ) -> str:
        """작업에 최적 모델 선택 - Provider 로테이션: OpenRouter -> Groq -> Cerebras."""
        if budget is None:
            budget = self.llm_config.budget_limit
        
        # 작업 유형에 적합한 모델 필터링
        suitable_models = [
            name for name, config in self.models.items()
            if task_type in config.capabilities
        ]
        
        if not suitable_models:
            # 기본 모델 사용
            return "gemini-flash-lite"
        
        # 사용 가능한 Provider 목록 가져오기
        available_providers = self._get_available_providers()
        
        if not available_providers:
            # 모든 Provider가 rate limit에 걸린 경우, 가장 오래된 것부터 재시도
            logger.warning("All providers are rate-limited, using oldest rate-limited provider")
            if self.provider_rate_limited:
                oldest_provider = min(self.provider_rate_limited.items(), key=lambda x: x[1])[0]
                available_providers = [oldest_provider]
            else:
                # 폴백: Gemini 모델 사용
                gemini_models = [
                    name for name in suitable_models
                    if self.models[name].provider == "google"
                ]
                if gemini_models:
                    return gemini_models[0]
                return "gemini-flash-lite"
        
        # Provider 로테이션: 사용 횟수가 가장 적은 Provider 선택
        provider_usage = {
            provider: self.provider_usage_count.get(provider, 0)
            for provider in available_providers
        }
        
        # 사용 횟수가 가장 적은 Provider 선택 (동일하면 순서대로)
        selected_provider = min(available_providers, key=lambda p: (provider_usage[p], available_providers.index(p)))
        
        # 사용 횟수 증가
        self.provider_usage_count[selected_provider] = provider_usage[selected_provider] + 1
        
        # Cerebras는 OpenRouter를 통해 접근
        check_provider = "openrouter" if selected_provider == "cerebras" else selected_provider
        
        # 해당 Provider의 모델 선택
        provider_models = [
            name for name in suitable_models
            if self.models[name].provider == check_provider
        ]
        
        # Cerebras인 경우 cerebras 모델 필터링
        if selected_provider == "cerebras":
            provider_models = [
                name for name in provider_models
                if "cerebras" in self.models[name].model_id.lower()
            ]
        
        if provider_models:
            selected_model = provider_models[0]
            logger.info(f"Selected {selected_provider.upper()} model (rotation #{self.provider_usage_count[selected_provider]}): {selected_model}")
            return selected_model
        
        # 해당 Provider에 모델이 없으면 다음 Provider 시도
        remaining_providers = [p for p in available_providers if p != selected_provider]
        if remaining_providers:
            return self.select_model(task_type, complexity, budget)
        
        # 모든 Provider 실패 시 폴백: Gemini 모델 사용
        gemini_models = [
            name for name in suitable_models
            if self.models[name].provider == "google"
        ]
        if gemini_models:
            return gemini_models[0]
        
        return "gemini-flash-lite"
    
    async def execute_with_model(
        self,
        prompt: str,
        task_type: TaskType,
        model_name: str = None,
        system_message: str = None,
        use_cascade: bool = True,
        complexity: float = 5.0,
        **kwargs
    ) -> ModelResult:
        """모델로 실행 - Cascade 지원."""
        if model_name is None:
            model_name = self.select_model(task_type)
        
        # 모델 클라이언트 확인
        model_name_clean = model_name.replace("_langchain", "")
        
        # 모델 provider 확인
        if model_name_clean not in self.models:
            raise ValueError(f"Model {model_name_clean} not found in models")
        
        model_provider = self.models[model_name_clean].provider
        model_config = self.models[model_name_clean]
        start_time = time.time()
        actual_model_used = model_name_clean  # 실제 사용된 모델 추적
        
        # prompt와 system_message는 execute_llm_task의 decorator에서 자동으로 최적화됨
        
        try:
            # Cascade 설정 확인
            try:
                cascade_config = get_cascade_config()
            except RuntimeError:
                cascade_config = None
            
            cascade_enabled = (
                use_cascade and
                cascade_config and
                cascade_config.enabled
            ) if cascade_config else use_cascade
            
            # Provider의 모든 모델 리스트 가져오기
            provider_models = self._get_provider_models(model_provider, task_type)
            
            # Cascade 실행 조건 체크
            use_cascade_for_provider = (
                cascade_enabled and
                len(provider_models) >= (cascade_config.min_models_for_cascade if cascade_config else 2)
            )
            
            if use_cascade_for_provider:
                # Cascade 실행
                logger.info(f"Using cascade for provider {model_provider} with {len(provider_models)} models")
                try:
                    result, actual_model_used = await self._execute_provider_cascade(
                        provider_models, prompt, system_message, task_type, complexity, **kwargs
                    )
                except Exception as cascade_error:
                    logger.warning(f"Cascade execution failed: {cascade_error}, falling back to single model...")
                    # Cascade 실패 시 기존 단일 모델 실행 로직으로 fallback
                    use_cascade_for_provider = False
            
            if not use_cascade_for_provider:
                # 기존 단일 모델 실행 로직
                # 우선순위에 따라 모델 실행 및 폴백
                if model_provider == "openrouter":
                    logger.info(f"Executing with OpenRouter model: {model_name_clean}")
                try:
                    result = await self._execute_openrouter_model(
                        model_name_clean, prompt, system_message, **kwargs
                    )
                except Exception as error:
                    error_str = str(error).lower()
                    # Rate limit 에러인 경우 Provider를 rate-limited로 표시
                    if "rate-limited" in error_str or "429" in error_str:
                        self._mark_provider_rate_limited("openrouter")
                    logger.warning(f"OpenRouter model {model_name_clean} failed: {error}, trying fallback...")
                    result, actual_model_used = await self._try_fallback_models(
                        task_type, prompt, system_message, skip_providers=["openrouter"], **kwargs
                    )
            elif model_provider == "groq":
                logger.info(f"Executing with Groq model: {model_name_clean}")
                try:
                    result = await self._execute_groq_model(
                        model_name_clean, prompt, system_message, **kwargs
                    )
                except Exception as error:
                    error_str = str(error).lower()
                    # Rate limit 에러인 경우 Provider를 rate-limited로 표시
                    if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                        self._mark_provider_rate_limited("groq")
                    logger.warning(f"Groq model {model_name_clean} failed: {error}, trying fallback...")
                    result, actual_model_used = await self._try_fallback_models(
                        task_type, prompt, system_message, skip_providers=["openrouter", "groq"], **kwargs
                    )
            elif model_provider == "google":
                logger.info(f"Executing with Gemini model: {model_name_clean}")
                try:
                    if model_name.endswith("_langchain"):
                        result = await self._execute_langchain_model(
                            model_name, prompt, system_message, **kwargs
                        )
                    else:
                        result = await self._execute_gemini_model(
                            model_name, prompt, system_message, **kwargs
                        )
                except Exception as error:
                    logger.warning(f"Gemini model {model_name_clean} failed: {error}, trying fallback...")
                    result, actual_model_used = await self._try_fallback_models(
                        task_type, prompt, system_message, skip_providers=["openrouter", "groq", "google"], **kwargs
                    )
            elif model_provider == "openai":
                logger.info(f"Executing with GPT model: {model_name_clean}")
                try:
                    result = await self._execute_openai_model(
                        model_name_clean, prompt, system_message, **kwargs
                    )
                except Exception as error:
                    logger.warning(f"GPT model {model_name_clean} failed: {error}, trying fallback...")
                    result, actual_model_used = await self._try_fallback_models(
                        task_type, prompt, system_message, skip_providers=["openrouter", "groq", "google", "openai"], **kwargs
                    )
                else:
                    raise ValueError(f"Unknown provider: {model_provider}")
            
            execution_time = time.time() - start_time
            
            # 비용 계산 (실제 사용된 모델 기준)
            if actual_model_used in self.models:
                model_config = self.models[actual_model_used]
                cost = len(prompt.split()) * model_config.cost_per_token
            else:
                cost = 0.0
            
            # 성능 기록 (실제 사용된 모델 기준)
            self.performance_tracker.record_execution(
                actual_model_used, task_type, execution_time, True, result.get("quality_score", 0.8)
            )
            
            return ModelResult(
                content=result["content"],
                model_used=actual_model_used,  # 실제 사용된 모델 반환
                execution_time=execution_time,
                confidence=result.get("confidence", 0.8),
                cost=cost,
                metadata=result.get("metadata", {})
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Model execution failed: {e}")
            
            # 실패 기록 (실제 사용된 모델 기준)
            self.performance_tracker.record_execution(
                actual_model_used, task_type, execution_time, False
            )
            
            raise
    
    async def _execute_gemini_model(
        self,
        model_name: str,
        prompt: str,
        system_message: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Gemini 모델 실행 (rate limit 재시도 포함)."""
        client = self.model_clients[model_name]
        model_config = self.models[model_name]
        
        # 프롬프트 구성
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        
        # Rate limit 재시도 로직
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 실행
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.generate_content(
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=model_config.temperature,
                            max_output_tokens=model_config.max_tokens
                        ),
                    )
                )
                break  # 성공 시 루프 종료
            except Exception as e:
                error_str = str(e).lower()
                # Rate limit 에러 감지
                if ("429" in error_str or "rate limit" in error_str or "quota exceeded" in error_str or 
                    "resource_exhausted" in error_str) and attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt)  # 지수 백오프: 5초, 10초, 20초
                    logger.warning(f"Gemini API rate limit (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Rate limit이 아니거나 최대 재시도 횟수 초과
                    raise
        
        # finish_reason 체크 및 안전한 응답 처리
        finish_reason = None
        finish_reason_int = None
        has_valid_part = False
        
        # 먼저 candidates와 parts 확인
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else None
            
            # parts 확인
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    # text 속성을 가진 part가 있는지 확인
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            has_valid_part = True
                            break
            
            # finish_reason이 enum인 경우 숫자로 변환
            if finish_reason is not None:
                try:
                    # FinishReason enum인 경우
                    if hasattr(finish_reason, 'value'):
                        finish_reason_int = finish_reason.value
                    elif hasattr(finish_reason, 'name'):
                        # SAFETY = 2
                        if 'SAFETY' in finish_reason.name or 'SAFETY' in str(finish_reason):
                            finish_reason_int = 2
                    # 숫자인 경우
                    elif isinstance(finish_reason, int):
                        finish_reason_int = finish_reason
                    # 문자열인 경우
                    elif isinstance(finish_reason, str):
                        if 'SAFETY' in finish_reason.upper() or finish_reason == '2':
                            finish_reason_int = 2
                except Exception:
                    # 변환 실패 시 그대로 사용
                    finish_reason_int = finish_reason if isinstance(finish_reason, int) else None
        
        # finish_reason이 2 (SAFETY)이거나 유효한 Part가 없는 경우
        if finish_reason_int == 2 or (finish_reason is not None and ('SAFETY' in str(finish_reason).upper())) or not has_valid_part:
            if finish_reason_int == 2 or (finish_reason is not None and ('SAFETY' in str(finish_reason).upper())):
                logger.warning(f"Gemini API safety filter triggered (finish_reason={finish_reason}). Returning empty content.")
            else:
                logger.warning(f"Gemini API response has no valid Part (finish_reason={finish_reason}). Returning empty content.")
            return {
                "content": "[Content blocked by safety filters. Please try rephrasing the request.]",
                "confidence": 0.0,
                "quality_score": 0.0,
                "metadata": {
                    "model": model_name,
                    "temperature": model_config.temperature,
                    "max_tokens": model_config.max_tokens,
                    "finish_reason": finish_reason,
                    "safety_filter_triggered": True
                }
            }
        
        # 안전한 텍스트 추출 (has_valid_part가 True이면 안전하게 접근 가능)
        try:
            content = response.text
        except ValueError as e:
            # 예외 발생 시 직접 추출 시도
            logger.warning(f"Gemini API response.text failed: {e}. Trying direct extraction.")
            content = ""
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    content += part.text
            
            if not content:
                content = "[Unable to extract content from response. This may be due to safety filters or other restrictions.]"
        
        return {
            "content": content,
            "confidence": 0.8,  # 기본 신뢰도
            "quality_score": 0.8,
            "metadata": {
                "model": model_name,
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_tokens,
                "finish_reason": finish_reason
            }
        }
    
    async def _execute_openrouter_model(
        self,
        model_name: str,
        prompt: str,
        system_message: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """OpenRouter 모델 실행."""
        model_config = self.models[model_name]
        
        # OpenRouter에 실제 존재하는 모델 ID 확인 및 변환
        model_id = self._get_valid_openrouter_model_id(model_config.model_id, model_name)
        
        # 메시지 구성
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # OpenRouter API 직접 호출
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mcp-agent.local",
            "X-Title": "MCP Agent Hub"
        }
        
        payload = {
            "model": model_id,  # 실제 OpenRouter 모델 ID 사용
            "messages": messages,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            **kwargs
        }
        
        # 재시도 로직: 502, 500, 503, 429 등 서버 에러와 rate limit은 재시도, 401/403/404는 재시도 안 함
        max_retries = 3
        retryable_status_codes = [429, 500, 502, 503, 504]  # Rate limit과 서버 에러 재시도
        response = None
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                )
                
                if response.status_code == 200:
                    # 성공
                    break
                
                # HTML 에러 페이지 필터링
                error_text = response.text
                if '<!DOCTYPE html>' in error_text or '<html' in error_text.lower():
                    # HTML에서 간단한 에러 메시지 추출
                    import re
                    status_code = response.status_code
                    title_match = re.search(r'<title>([^<]+)</title>', error_text, re.IGNORECASE)
                    if title_match:
                        error_msg = f"HTTP {status_code}: {title_match.group(1).strip()}"
                    elif status_code == 502:
                        error_msg = f"HTTP {status_code}: Bad Gateway - Server temporarily unavailable"
                    elif status_code == 500:
                        error_msg = f"HTTP {status_code}: Internal Server Error"
                    else:
                        error_msg = f"HTTP {status_code}: Server Error"
                else:
                    error_msg = f"HTTP {response.status_code}: {error_text[:200]}"
                
                # Rate limit (429) 에러 처리: Provider를 rate-limited로 표시하고 다음 Provider로 전환
                if response.status_code == 429:
                    # OpenRouter Provider를 rate-limited로 표시
                    self._mark_provider_rate_limited("openrouter")
                    logger.warning(f"OpenRouter rate-limited (429), will use next provider in rotation")
                    # Rate limit은 재시도하지 않고 즉시 다음 Provider로 전환
                    raise RuntimeError(f"OpenRouter API rate-limited (429): {error_msg}")
                
                # 재시도 가능한 에러인지 확인 (429는 이미 처리했으므로 제외)
                if response.status_code in retryable_status_codes and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 지수 백오프: 1초, 2초, 4초
                    logger.warning(f"OpenRouter API error (attempt {attempt + 1}/{max_retries}): {error_msg}, retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue  # 재시도
                else:
                    # 재시도 불가능한 에러 (401, 403, 404 등) 또는 최대 재시도 횟수 초과
                    # 400 에러는 모델 ID가 잘못된 경우일 수 있음
                    if response.status_code == 400 and ("not a valid model ID" in error_text.lower() or "400" in error_text):
                        # OpenRouter에 존재하는 모델로 자동 대체 시도
                        logger.warning(f"Model {model_id} not found in OpenRouter, trying fallback models...")
                        fallback_models = [
                            "google/gemini-2.0-flash-exp:free",
                            "meta-llama/llama-3.2-3b-instruct:free",
                            "mistralai/mistral-7b-instruct:free",
                        ]
                        
                        for fallback_model in fallback_models:
                            try:
                                logger.info(f"Trying fallback model: {fallback_model}")
                                payload["model"] = fallback_model
                                fallback_response = await asyncio.get_event_loop().run_in_executor(
                                    None,
                                    lambda: requests.post(
                                        "https://openrouter.ai/api/v1/chat/completions",
                                        headers=headers,
                                        json=payload,
                                        timeout=30
                                    )
                                )
                                
                                if fallback_response.status_code == 200:
                                    response = fallback_response
                                    model_id = fallback_model  # 실제 사용된 모델 ID 업데이트
                                    logger.info(f"✅ Successfully used fallback model: {fallback_model}")
                                    break
                            except Exception as fallback_error:
                                logger.debug(f"Fallback model {fallback_model} failed: {fallback_error}")
                                continue
                        
                        if response.status_code != 200:
                            logger.error(f"OpenRouter API error: {error_msg}")
                            raise RuntimeError(f"OpenRouter API error: {error_msg}")
                    else:
                        logger.error(f"OpenRouter API error: {error_msg}")
                        raise RuntimeError(f"OpenRouter API error: {error_msg}")
                    
            except requests.exceptions.RequestException as e:
                # 네트워크 에러도 재시도
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"OpenRouter API request failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"OpenRouter API request failed after {max_retries} attempts: {e}")
                    raise RuntimeError(f"OpenRouter API request failed: {e}")
        
        # 마지막 시도 결과 확인
        if not response or response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error after {max_retries} attempts: HTTP {response.status_code if response else 'No response'}")
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        return {
            "content": content,
            "confidence": 0.8,
            "quality_score": 0.8,
            "metadata": {
                "model": model_name,
                "provider": "openrouter",
                "model_id": model_config.model_id,
                "tokens_used": len(content.split()),
                "usage": data.get("usage", {})
            }
        }
    
    def _get_provider_models(
        self,
        provider: str,
        task_type: TaskType
    ) -> List[str]:
        """
        Provider의 모든 사용 가능한 모델 리스트 반환.
        
        Args:
            provider: Provider 이름 (openrouter, groq, google, openai 등)
            task_type: 작업 유형
        
        Returns:
            해당 provider의 모델 이름 리스트 (비용 오름차순 정렬)
        """
        if provider == "cerebras":
            # Cerebras는 OpenRouter를 통해 접근
            models = [
                name for name, config in self.models.items()
                if config.provider == "openrouter"
                and "cerebras" in config.model_id.lower()
                and task_type in config.capabilities
            ]
        else:
            models = [
                name for name, config in self.models.items()
                if config.provider == provider
                and task_type in config.capabilities
            ]
        
        # 비용 기준 정렬 (저비용 우선)
        models.sort(key=lambda name: self.models[name].cost_per_token)
        
        return models
    
    def _classify_models_for_cascade(
        self,
        provider_models: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Provider 모델 리스트를 Drafter/Verifier로 분류.
        
        기준:
        - Drafter: cost_per_token < threshold 또는 speed_rating > threshold
        - Verifier: cost_per_token >= threshold 또는 quality_rating > threshold
        
        Returns:
            (drafter_models, verifier_models) 튜플
        """
        try:
            cascade_config = get_cascade_config()
        except RuntimeError:
            cascade_config = None
        
        if cascade_config:
            drafter_cost_threshold = cascade_config.drafter_cost_threshold
            drafter_speed_threshold = cascade_config.drafter_speed_threshold
            verifier_quality_threshold = cascade_config.verifier_quality_threshold
        else:
            # 기본값
            drafter_cost_threshold = 0.0002
            drafter_speed_threshold = 7.0
            verifier_quality_threshold = 8.0
        
        drafter_models = []
        verifier_models = []
        
        for model_name in provider_models:
            config = self.models[model_name]
            
            is_drafter = (
                config.cost_per_token < drafter_cost_threshold or
                config.speed_rating > drafter_speed_threshold
            )
            
            is_verifier = (
                config.cost_per_token >= drafter_cost_threshold or
                config.quality_rating > verifier_quality_threshold
            )
            
            if is_drafter:
                drafter_models.append(model_name)
            if is_verifier:
                verifier_models.append(model_name)
        
        # 기본값: 첫 번째 모델을 drafter, 마지막 모델을 verifier로
        if not drafter_models:
            drafter_models = [provider_models[0]] if provider_models else []
        if not verifier_models:
            verifier_models = [provider_models[-1]] if provider_models else []
        
        return drafter_models, verifier_models
    
    def _validate_draft_quality(
        self,
        draft_result: Dict[str, Any],
        prompt: str,
        task_type: TaskType,
        complexity: float = 5.0
    ) -> bool:
        """
        Draft 품질 검증.
        
        Cascadeflow의 validation 방식을 참고:
        - Confidence score 체크
        - Complexity 기반 threshold 조정
        - 기본 threshold: 0.75
        
        Returns:
            True if draft should be accepted, False if needs verifier
        """
        try:
            cascade_config = get_cascade_config()
        except RuntimeError:
            cascade_config = None
        
        base_threshold = cascade_config.confidence_threshold if cascade_config else 0.75
        enable_adaptive = cascade_config.enable_adaptive_threshold if cascade_config else True
        
        # Complexity 기반 threshold 조정
        if enable_adaptive:
            if complexity > 7.0:
                threshold = 0.85  # 높은 복잡도는 더 높은 threshold
            elif complexity > 5.0:
                threshold = 0.80
            else:
                threshold = base_threshold
        else:
            threshold = base_threshold
        
        # Confidence 체크
        confidence = draft_result.get("confidence", 0.0)
        if confidence >= threshold:
            return True
        
        # Quality score 체크 (있는 경우)
        quality_score = draft_result.get("quality_score", None)
        if quality_score is not None and quality_score >= threshold:
            return True
        
        return False
    
    async def _execute_single_model_by_provider(
        self,
        model_name: str,
        prompt: str,
        system_message: str = None,
        task_type: TaskType = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Provider별 단일 모델 실행 (기존 로직 재사용).
        
        기존의 _execute_openrouter_model, _execute_groq_model 등을 호출.
        """
        model_provider = self.models[model_name].provider
        
        if model_provider == "openrouter":
            return await self._execute_openrouter_model(
                model_name, prompt, system_message, **kwargs
            )
        elif model_provider == "groq":
            return await self._execute_groq_model(
                model_name, prompt, system_message, **kwargs
            )
        elif model_provider == "google":
            if model_name.endswith("_langchain"):
                return await self._execute_langchain_model(
                    model_name, prompt, system_message, **kwargs
                )
            else:
                return await self._execute_gemini_model(
                    model_name, prompt, system_message, **kwargs
                )
        elif model_provider == "openai":
            return await self._execute_openai_model(
                model_name, prompt, system_message, **kwargs
            )
        else:
            raise ValueError(f"Unknown provider: {model_provider}")
    
    async def _execute_provider_cascade(
        self,
        provider_models: List[str],
        prompt: str,
        system_message: str = None,
        task_type: TaskType = None,
        complexity: float = 5.0,
        **kwargs
    ) -> Tuple[Dict[str, Any], str]:
        """
        Provider 내부 모델 리스트로 Cascade 실행.
        
        흐름:
        1. Drafter/Verifier 분류
        2. Drafter로 실행
        3. Quality validation
        4. Accept → Drafter 결과 반환
        5. Reject → Verifier로 승격
        
        Returns:
            (result_dict, actual_model_used) 튜플
        """
        # 1. Drafter/Verifier 분류
        drafter_models, verifier_models = self._classify_models_for_cascade(
            provider_models
        )
        
        drafter = drafter_models[0]
        verifier = verifier_models[0] if verifier_models else provider_models[-1]
        
        # 2. Drafter 실행
        logger.info(f"Executing cascade drafter: {drafter}")
        draft_result = await self._execute_single_model_by_provider(
            drafter, prompt, system_message, task_type, **kwargs
        )
        
        # 3. Quality validation
        should_accept = self._validate_draft_quality(
            draft_result, prompt, task_type, complexity
        )
        
        if should_accept:
            logger.info(f"✓ Draft accepted: {drafter}")
            return draft_result, drafter
        
        # 4. Verifier로 승격
        logger.info(f"✗ Draft rejected, escalating to verifier: {verifier}")
        verifier_result = await self._execute_single_model_by_provider(
            verifier, prompt, system_message, task_type, **kwargs
        )
        
        return verifier_result, verifier
    
    async def _try_fallback_models(
        self,
        task_type: TaskType,
        prompt: str,
        system_message: str = None,
        skip_providers: List[str] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], str]:
        """우선순위에 따라 폴백 모델 시도: OpenRouter -> Groq -> Cerebras (OpenRouter) -> Gemini -> GPT -> Claude."""
        if skip_providers is None:
            skip_providers = []
        
        # 사용자 지정 우선순위: openrouter -> groq -> cerebras (openrouter) -> google -> openai -> claude
        fallback_order = ["openrouter", "groq", "cerebras", "google", "openai", "claude"]
        
        for provider in fallback_order:
            if provider in skip_providers:
                continue
            
            # 해당 provider의 사용 가능한 모델 찾기
            if provider == "cerebras":
                # Cerebras는 OpenRouter를 통해 접근 (cerebras/llama-3.1-70b-instruct 등)
                available_models = [
                    name for name, config in self.models.items()
                    if config.provider == "openrouter" 
                    and "cerebras" in config.model_id.lower()
                    and task_type in config.capabilities
                ]
            else:
                available_models = [
                    name for name, config in self.models.items()
                    if config.provider == provider and task_type in config.capabilities
                ]
            
            if not available_models:
                continue
            
            # 첫 번째 사용 가능한 모델 시도
            fallback_model = available_models[0]
            logger.info(f"Trying fallback model: {fallback_model} (provider: {provider})")
            
            try:
                if provider == "openrouter" or provider == "cerebras":
                    # Cerebras도 OpenRouter를 통해 접근
                    result = await self._execute_openrouter_model(
                        fallback_model, prompt, system_message, **kwargs
                    )
                elif provider == "groq":
                    result = await self._execute_groq_model(
                        fallback_model, prompt, system_message, **kwargs
                    )
                elif provider == "google":
                    result = await self._execute_gemini_model(
                        fallback_model, prompt, system_message, **kwargs
                    )
                elif provider == "openai":
                    result = await self._execute_openai_model(
                        fallback_model, prompt, system_message, **kwargs
                    )
                elif provider == "claude":
                    # Claude는 OpenAI API 호환 또는 OpenRouter를 통해 접근
                    # OpenRouter를 통해 Claude 모델 찾기
                    claude_models = [
                        name for name, config in self.models.items()
                        if config.provider == "openrouter" 
                        and "claude" in config.model_id.lower()
                        and task_type in config.capabilities
                    ]
                    if claude_models:
                        result = await self._execute_openrouter_model(
                            claude_models[0], prompt, system_message, **kwargs
                        )
                    else:
                        continue
                else:
                    continue
                
                logger.info(f"✅ Fallback successful with {fallback_model}")
                return result, fallback_model
            except Exception as e:
                # 에러 메시지에서 HTML 필터링 및 중첩 방지
                error_str = str(e)
                
                # 모델 존재하지 않음 (404) 또는 Decommissioned 모델 감지
                if ("does not exist" in error_str.lower() or "model_not_found" in error_str.lower() or
                    "decommissioned" in error_str.lower() or "model_decommissioned" in error_str.lower()):
                    logger.warning(f"Fallback model {fallback_model} is not available (404/decommissioned), trying next...")
                    # Groq 모델이 존재하지 않는 경우 모델 목록에서 제거
                    if provider == "groq" and fallback_model in self.models:
                        logger.warning(f"Removing unavailable Groq model from available models: {fallback_model}")
                        del self.models[fallback_model]
                    continue
                
                # Rate limit 에러 (429)는 재시도 가능하지만 fallback에서는 다음 모델로
                if "429" in error_str or "rate limit" in error_str.lower() or "rate-limited" in error_str.lower() or "rate limit exceeded" in error_str.lower():
                    logger.warning(f"Fallback model {fallback_model} rate limited (429), trying next...")
                    continue
                
                if '<!DOCTYPE html>' in error_str or '<html' in error_str.lower():
                    import re
                    status_match = re.search(r'(\d{3})', error_str)
                    status_code = status_match.group(1) if status_match else "Unknown"
                    title_match = re.search(r'<title>([^<]+)</title>', error_str, re.IGNORECASE)
                    if title_match:
                        error_msg = f"HTTP {status_code}: {title_match.group(1).strip()}"
                    else:
                        error_msg = f"HTTP {status_code}: Server Error"
                else:
                    # 중첩된 메시지 방지: "OpenRouter model X failed: OpenRouter model X failed: ..." 형식 제거
                    if "failed:" in error_str:
                        # 마지막 "failed:" 이후의 메시지만 사용
                        parts = error_str.split("failed:")
                        if len(parts) > 1:
                            error_msg = parts[-1].strip()
                        else:
                            error_msg = error_str[:200]
                    else:
                        error_msg = error_str[:200]
                
                logger.warning(f"Fallback model {fallback_model} failed: {error_msg}, trying next...")
                continue
        
        # 모든 폴백 실패
        raise RuntimeError("All fallback models failed. No available models.")
    
    async def _execute_groq_model(
        self,
        model_name: str,
        prompt: str,
        system_message: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Groq 모델 실행."""
        if model_name not in self.model_clients:
            raise ValueError(f"Groq client not initialized for {model_name}")
        
        client = self.model_clients[model_name]
        model_config = self.models[model_name]
        
        # Groq에 실제 존재하는 모델 ID 사용
        model_id = self._get_valid_groq_model_id(model_config.model_id)
        
        # 메시지 구성
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Groq API 호출
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model_id,  # 실제 Groq 모델 ID 사용
                    messages=messages,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    **kwargs
                )
            )
            
            content = response.choices[0].message.content
            
            return {
                "content": content,
                "confidence": 0.8,
                "quality_score": 0.8,
                "metadata": {
                    "model": model_name,
                    "provider": "groq",
                    "model_id": model_id,  # 실제 사용된 모델 ID (변환된 것일 수 있음)
                    "original_model_id": model_config.model_id,  # 원래 요청한 모델 ID
                    "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else len(content.split())
                }
            }
        except Exception as e:
            error_str = str(e).lower()
            
            # 모델 존재하지 않음 (404) 또는 Decommissioned 모델 감지 및 자동 대체
            if ("does not exist" in error_str or "model_not_found" in error_str or 
                "decommissioned" in error_str or "model_decommissioned" in error_str):
                logger.warning(f"Groq model {model_name} ({model_config.model_id}) is not available")
                
                # 실제 존재하는 Groq 모델로 대체 시도
                replacement_models = [
                    "llama-3.1-8b-instant",  # 실제 존재하는 모델
                    "mixtral-8x7b-32768"  # 실제 존재하는 모델
                ]
                
                for replacement_model in replacement_models:
                    logger.info(f"Attempting to use replacement model: {replacement_model}")
                    try:
                        # 대체 모델로 재시도
                        replacement_response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda rm=replacement_model: client.chat.completions.create(
                                model=rm,
                                messages=messages,
                                temperature=model_config.temperature,
                                max_tokens=model_config.max_tokens,
                                **kwargs
                            )
                        )
                        content = replacement_response.choices[0].message.content
                        logger.info(f"✅ Successfully used replacement model: {replacement_model}")
                        
                        # 모델 설정 업데이트 (다음 요청을 위해)
                        self.models[model_name].model_id = replacement_model
                        
                        return {
                            "content": content,
                            "confidence": 0.8,
                            "quality_score": 0.8,
                            "metadata": {
                                "model": model_name,
                                "provider": "groq",
                                "model_id": replacement_model,  # 실제 사용된 모델
                                "original_model_id": model_config.model_id,  # 원래 요청한 모델
                                "tokens_used": replacement_response.usage.total_tokens if hasattr(replacement_response, 'usage') else len(content.split())
                            }
                        }
                    except Exception as replacement_error:
                        logger.debug(f"Replacement model {replacement_model} failed: {replacement_error}, trying next...")
                        continue
                
                # 모든 대체 모델 실패
                logger.error(f"All replacement models failed for unavailable model {model_config.model_id}")
                raise RuntimeError(f"Groq model {model_name} ({model_config.model_id}) is not available and all replacement models failed")
            
            logger.error(f"Groq API error: {e}")
            raise RuntimeError(f"Groq model {model_name} failed: {e}")
    
    async def _execute_openai_model(
        self,
        model_name: str,
        prompt: str,
        system_message: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """OpenAI/GPT 모델 실행."""
        if model_name not in self.model_clients:
            raise ValueError(f"OpenAI client not initialized for {model_name}")
        
        client = self.model_clients[model_name]
        model_config = self.models[model_name]
        
        # 메시지 구성
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # OpenAI API 호출
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model_config.model_id,
                    messages=messages,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    **kwargs
                )
            )
            
            content = response.choices[0].message.content
            
            return {
                "content": content,
                "confidence": 0.8,
                "quality_score": 0.8,
                "metadata": {
                    "model": model_name,
                    "provider": "openai",
                    "model_id": model_config.model_id,
                    "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else len(content.split())
                }
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI model {model_name} failed: {e}")
    
    async def _execute_langchain_model(
        self,
        model_name: str,
        prompt: str,
        system_message: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """LangChain 모델 실행."""
        client = self.model_clients[model_name]
        
        # 메시지 구성
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
        # 실행
        response = await client.ainvoke(messages)
        
        return {
            "content": response.content,
            "confidence": 0.8,
            "quality_score": 0.8,
            "metadata": {
                "model": model_name,
                "response_type": type(response).__name__
            }
        }
    
    async def weighted_ensemble(
        self,
        prompt: str,
        task_type: TaskType,
        models: List[str] = None,
        weights: List[float] = None
    ) -> ModelResult:
        """Weighted Ensemble 실행."""
        if models is None:
            models = [self.select_model(task_type) for _ in range(3)]
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # weights 타입 검증 및 변환 (엄격한 검증)
        if weights:
            try:
                validated_weights = []
                for w in weights:
                    if isinstance(w, (int, float)):
                        # 숫자 타입은 그대로 사용 (0 이상인지 확인)
                        validated_weights.append(max(0.0, float(w)))
                    elif isinstance(w, str):
                        # 문자열인 경우 숫자로 변환 시도
                        # 먼저 숫자가 아닌 문자 제거 (공백, 문자 등)
                        cleaned_str = ''.join(c for c in w if c.isdigit() or c == '.' or c == '-' or c == '+')
                        
                        # '.' 만 있거나 숫자가 없는 경우 처리
                        if not cleaned_str or cleaned_str == '.' or cleaned_str in ['-', '+', '-.', '+.']:
                            logger.warning(f"Invalid weight value '{w}' (no valid number), using 1.0")
                            validated_weights.append(1.0)
                        else:
                            try:
                                float_val = float(cleaned_str)
                                validated_weights.append(max(0.0, float_val))
                            except (ValueError, TypeError):
                                # 변환 실패 시 기본값 1.0 사용
                                logger.warning(f"Invalid weight value '{w}' (cleaned: '{cleaned_str}'), using 1.0")
                                validated_weights.append(1.0)
                    else:
                        # 기타 타입은 기본값 사용
                        logger.warning(f"Invalid weight type '{type(w)}', using 1.0")
                        validated_weights.append(1.0)
                
                # 검증된 weights 사용
                if len(validated_weights) == len(weights):
                    weights = validated_weights
                else:
                    raise ValueError("Weight validation failed")
            except Exception as e:
                logger.warning(f"Invalid weights format, using equal weights: {e}")
                weights = [1.0 / len(models)] * len(models)
        
        # 모든 모델로 실행
        results = []
        for model in models:
            try:
                result = await self.execute_with_model(prompt, task_type, model)
                results.append(result)
            except Exception as e:
                logger.warning(f"Model {model} failed in ensemble: {e}")
                continue
        
        if not results:
            raise RuntimeError("All models failed in ensemble")
        
        # weights 개수를 results 개수에 맞춤
        if len(weights) > len(results):
            weights = weights[:len(results)]
        elif len(weights) < len(results):
            # 부족한 weights는 동일하게 분배
            remaining = 1.0 - sum(weights[:len(weights)])
            weights.extend([remaining / (len(results) - len(weights))] * (len(results) - len(weights)))
        
        # 가중 평균으로 결과 통합
        try:
            total_weight = sum(weights)
            if total_weight <= 0:
                # 모든 weight가 0이거나 음수면 동일하게 분배
                logger.warning("Total weight is 0 or negative, using equal weights")
                weights = [1.0 / len(results)] * len(results)
                total_weight = 1.0
        except (TypeError, ValueError) as e:
            logger.error(f"Error calculating total weight: {e}, weights: {weights}")
            # 기본값 사용
            weights = [1.0 / len(results)] * len(results)
            total_weight = 1.0
        weighted_content = ""
        total_confidence = 0.0
        total_cost = 0.0
        total_time = 0.0
        
        for i, result in enumerate(results):
            # 안전한 weight 계산
            try:
                weight = float(weights[i]) / total_weight if total_weight > 0 else 1.0 / len(results)
            except (TypeError, ValueError, IndexError) as e:
                logger.warning(f"Error calculating weight for result {i}: {e}, using equal weight")
                weight = 1.0 / len(results)
            
            weighted_content += f"[{result.model_used}] {result.content}\n\n"
            total_confidence += result.confidence * weight
            total_cost += result.cost * weight
            total_time += result.execution_time * weight
        
        return ModelResult(
            content=weighted_content.strip(),
            model_used=f"ensemble({','.join([r.model_used for r in results])})",
            execution_time=total_time,
            confidence=total_confidence,
            cost=total_cost,
            metadata={
                "ensemble_models": [r.model_used for r in results],
                "weights": weights[:len(results)],
                "individual_results": [
                    {
                        "model": r.model_used,
                        "confidence": r.confidence,
                        "cost": r.cost
                    }
                    for r in results
                ]
            }
        )
    
    def get_model_performance_stats(self) -> Dict[str, Any]:
        """모델 성능 통계 반환."""
        stats = {}
        for model_name in self.models:
            stats[model_name] = {
                "overall_score": self.performance_tracker.get_model_score(model_name),
                "task_scores": {
                    task_type.value: self.performance_tracker.get_model_score(model_name, task_type)
                    for task_type in TaskType
                }
            }
        return stats
    
    def get_best_model_for_task(self, task_type: TaskType) -> str:
        """작업에 최적 모델 반환."""
        best_model = None
        best_score = 0.0
        
        for model_name in self.models:
            score = self.performance_tracker.get_model_score(model_name, task_type)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model or "gemini-flash-lite"


# Global orchestrator instance (lazy initialization)
_llm_orchestrator = None

def get_llm_orchestrator() -> 'MultiModelOrchestrator':
    """Get or initialize global LLM orchestrator."""
    global _llm_orchestrator
    if _llm_orchestrator is None:
        _llm_orchestrator = MultiModelOrchestrator()
    return _llm_orchestrator


@refine_llm_call
async def execute_llm_task(
    prompt: str,
    task_type: TaskType,
    model_name: str = None,
    system_message: str = None,
    use_ensemble: bool = False,
    **kwargs
) -> ModelResult:
    """LLM 작업 실행 (API 모델 + CLI 에이전트 지원)."""
    try:
        # CLI 에이전트 체크
        if model_name and _is_cli_agent(model_name):
            return await _execute_cli_agent_task(
                prompt, task_type, model_name, system_message, **kwargs
            )

        # 기존 API 모델 처리
        if use_ensemble:
            return await get_llm_orchestrator().weighted_ensemble(
                prompt, task_type, model_name, system_message, **kwargs
            )
        else:
            return await get_llm_orchestrator().execute_with_model(
                prompt, task_type, model_name, system_message, **kwargs
            )
    except Exception as e:
        logger.error(f"LLM task execution failed: {e}")
        raise


def get_best_model_for_task(task_type: TaskType) -> str:
    """작업에 최적 모델 반환."""
    return get_llm_orchestrator().get_best_model_for_task(task_type)


def get_model_performance_stats() -> Dict[str, Any]:
    """모델 성능 통계 반환."""
    return get_llm_orchestrator().get_model_performance_stats()


# CLI 에이전트 지원 함수들
def _is_cli_agent(model_name: str) -> bool:
    """모델 이름이 CLI 에이전트인지 확인"""
    cli_agents = {
        'claude_code', 'open_code', 'gemini_cli', 'cline_cli',
        'claudecode', 'opencode', 'gemini-cli', 'cline-cli'
    }
    return model_name.lower() in cli_agents


async def _execute_cli_agent_task(
    prompt: str,
    task_type: TaskType,
    agent_name: str,
    system_message: str = None,
    **kwargs
) -> ModelResult:
    """CLI 에이전트 작업 실행"""
    from src.core.cli_agents.cli_agent_manager import get_cli_agent_manager

    try:
        cli_manager = get_cli_agent_manager()

        # 시스템 메시지와 프롬프트 결합
        full_query = prompt
        if system_message:
            full_query = f"{system_message}\n\n{prompt}"

        # 작업 유형에 따른 추가 파라미터 설정
        agent_kwargs = kwargs.copy()

        # 작업 유형별 특화 파라미터
        if task_type == TaskType.GENERATION:
            agent_kwargs.setdefault('mode', 'generate')
        elif task_type == TaskType.ANALYSIS:
            agent_kwargs.setdefault('mode', 'analyze')
        elif task_type == TaskType.RESEARCH:
            agent_kwargs.setdefault('mode', 'chat')

        # CLI 에이전트 실행
        result = await cli_manager.execute_with_agent(agent_name, full_query, **agent_kwargs)

        # ModelResult 형식으로 변환
        return ModelResult(
            success=result.get('success', False),
            response=result.get('response', ''),
            model_name=f"cli:{agent_name}",
            confidence=result.get('confidence', 0.0),
            usage=result.get('usage', {}),
            metadata={
                'agent_type': 'cli',
                'agent_name': agent_name,
                'task_type': task_type.value,
                'execution_time': result.get('metadata', {}).get('execution_time', 0),
                **result.get('metadata', {})
            }
        )

    except Exception as e:
        logger.error(f"CLI agent execution failed: {agent_name} - {e}")
        return ModelResult(
            success=False,
            response="",
            model_name=f"cli:{agent_name}",
            confidence=0.0,
            usage={},
            metadata={
                'agent_type': 'cli',
                'agent_name': agent_name,
                'error': str(e)
            }
        )
