#!/usr/bin/env python3
"""
Adaptive Research Depth (9번째 혁신)

작업 복잡도, 사용자 목표, 시간 제약에 따라 연구 깊이를 자동으로 조정하는 시스템.
Preset Modes (quick/medium/deep/auto), Progressive Deepening, Self-Adjusting, Dynamic Iteration Control.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ResearchPreset(Enum):
    """연구 깊이 프리셋 (9번째 혁신: Adaptive Research Depth)."""
    QUICK = "quick"    # 빠른 연구 (1-2 subtopics, 1-2 iterations)
    MEDIUM = "medium"  # 균형잡힌 연구 (5 subtopics, 4 iterations)
    DEEP = "deep"      # 깊이 있는 연구 (8 subtopics, 7 iterations)
    AUTO = "auto"      # 자율 결정 (에이전트가 최적 깊이 선택)


@dataclass
class DepthConfig:
    """연구 깊이 설정."""
    preset: ResearchPreset
    planning: Dict[str, Any]  # Planning 단계 설정
    researching: Dict[str, Any]  # Researching 단계 설정
    reporting: Dict[str, Any]  # Reporting 단계 설정
    complexity_score: float = 0.0  # 복잡도 점수 (0.0 ~ 1.0)


class AdaptiveResearchDepth:
    """
    적응형 연구 깊이 관리 (9번째 혁신).
    
    작업 복잡도와 목표에 따라 연구 깊이를 자동 조정합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        초기화.
        
        Args:
            config: ResearchConfig의 research_depth 설정
        """
        self.config = config
        # Cache preset configurations for better performance
        self._preset_configs: Optional[Dict[str, Any]] = None
        logger.info("AdaptiveResearchDepth initialized")
    
    def _load_preset_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        프리셋 설정 로드 (cached for performance).
        
        Returns:
            Dict[str, Dict[str, Any]]: 프리셋 설정 매핑
        """
        # Return cached configs if available
        if self._preset_configs is not None:
            return self._preset_configs
            
        if isinstance(self.config, dict):
            presets = self.config.get("presets", {})
        else:
            # AdaptiveResearchDepthConfig 객체인 경우
            presets = self.config.presets if hasattr(self.config, "presets") else {}
        
        # 기본 프리셋 설정
        default_presets = {
            "quick": {
                "description": "빠른 연구, 최소 깊이",
                "planning": {
                    "decompose": {
                        "mode": "manual",
                        "initial_subtopics": 1,
                        "auto_max_subtopics": 2
                    }
                },
                "researching": {
                    "max_iterations": 1,
                    "iteration_mode": "fixed"
                },
                "reporting": {
                    "min_section_length": 300
                }
            },
            "medium": {
                "description": "균형잡힌 연구",
                "planning": {
                    "decompose": {
                        "mode": "manual",
                        "initial_subtopics": 5,
                        "auto_max_subtopics": 5
                    }
                },
                "researching": {
                    "max_iterations": 4,
                    "iteration_mode": "fixed"
                },
                "reporting": {
                    "min_section_length": 500
                }
            },
            "deep": {
                "description": "깊이 있는 연구",
                "planning": {
                    "decompose": {
                        "mode": "manual",
                        "initial_subtopics": 8,
                        "auto_max_subtopics": 8
                    }
                },
                "researching": {
                    "max_iterations": 7,
                    "iteration_mode": "fixed"
                },
                "reporting": {
                    "min_section_length": 800
                }
            },
            "auto": {
                "description": "자율 결정 (에이전트가 최적 깊이 선택)",
                "planning": {
                    "decompose": {
                        "mode": "auto",
                        "auto_max_subtopics": 8
                    }
                },
                "researching": {
                    "max_iterations": 6,
                    "iteration_mode": "flexible"
                },
                "reporting": {
                    "min_section_length": 500
                }
            }
        }
        
        # 사용자 설정으로 기본값 병합
        for preset_name, default_config in default_presets.items():
            if preset_name in presets:
                user_config = presets[preset_name]
                if isinstance(user_config, dict):
                    # 딥 머지
                    merged = self._deep_merge(default_config, user_config)
                    default_presets[preset_name] = merged
                elif hasattr(user_config, "planning"):  # ResearchPresetConfig 객체
                    default_presets[preset_name] = {
                        "description": user_config.description,
                        "planning": user_config.planning,
                        "researching": user_config.researching,
                        "reporting": user_config.reporting,
                    }
        
        # Cache the result before returning
        self._preset_configs = default_presets
        return default_presets
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """딥 머지."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def determine_depth(
        self,
        user_request: str,
        preset: Optional[ResearchPreset] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DepthConfig:
        """
        연구 깊이 결정.
        
        Args:
            user_request: 사용자 요청
            preset: 프리셋 모드 (None이면 AUTO 또는 config의 default_preset)
            context: 추가 컨텍스트 (optional)
        
        Returns:
            DepthConfig 객체
        """
        # 프리셋 결정
        if preset is None:
            # config에서 default_preset 가져오기
            if isinstance(self.config, dict):
                default_preset_str = self.config.get("default_preset", "auto")
            else:
                default_preset_str = getattr(self.config, "default_preset", "auto")
            
            try:
                preset = ResearchPreset(default_preset_str)
            except ValueError:
                preset = ResearchPreset.AUTO
        
        # AUTO 모드인 경우 복잡도 분석
        if preset == ResearchPreset.AUTO:
            complexity = self._analyze_complexity(user_request, context)
            preset = self._auto_determine_preset(complexity)
            logger.info(f"Auto mode: complexity={complexity:.2f}, selected preset={preset.value}")
        else:
            complexity = self._estimate_complexity_for_preset(preset)
        
        # 프리셋 설정 가져오기
        configs = self._load_preset_configs()
        preset_config = configs.get(preset.value, configs["medium"])
        
        return DepthConfig(
            preset=preset,
            planning=preset_config.get("planning", {}),
            researching=preset_config.get("researching", {}),
            reporting=preset_config.get("reporting", {}),
            complexity_score=complexity
        )
    
    def _analyze_complexity(self, request: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        요청 복잡도 분석.
        
        Args:
            request: 사용자 요청
            context: 추가 컨텍스트
        
        Returns:
            복잡도 점수 (0.0 ~ 1.0)
        """
        complexity = 0.0
        
        # 1. 키워드 분석
        request_lower = request.lower()
        
        # 복잡도 증가 키워드
        complex_keywords = [
            "comprehensive", "detailed", "thorough", "deep", "extensive",
            "analyze", "compare", "evaluate", "synthesize", "review",
            "multiple", "various", "different", "all", "every"
        ]
        for keyword in complex_keywords:
            if keyword in request_lower:
                complexity += 0.1
        
        # 단순도 증가 키워드
        simple_keywords = [
            "quick", "brief", "summary", "overview", "simple",
            "what is", "define", "explain briefly"
        ]
        for keyword in simple_keywords:
            if keyword in request_lower:
                complexity -= 0.1
        
        # 2. 질문 유형 분석
        question_words = ["what", "who", "when", "where", "why", "how"]
        question_count = sum(1 for word in question_words if word in request_lower)
        complexity += question_count * 0.05
        
        # 3. 도메인 수 추정 (키워드 기반)
        domain_keywords = [
            "machine learning", "deep learning", "neural network",
            "natural language", "computer vision", "reinforcement",
            "statistics", "mathematics", "physics", "biology", "chemistry"
        ]
        domain_count = sum(1 for domain in domain_keywords if domain in request_lower)
        complexity += min(domain_count * 0.1, 0.3)
        
        # 4. 컨텍스트에서 복잡도 힌트
        if context:
            if context.get("complexity_hint"):
                complexity = max(0.0, min(1.0, complexity + context["complexity_hint"]))
            if context.get("time_constraint") == "urgent":
                complexity = max(0.0, complexity - 0.2)  # 긴급하면 단순화
        
        # 5. 요청 길이 고려
        request_length = len(request.split())
        if request_length > 50:
            complexity += 0.1
        elif request_length < 10:
            complexity -= 0.1
        
        # 정규화 (0.0 ~ 1.0)
        complexity = max(0.0, min(1.0, complexity))
        
        return complexity
    
    def _auto_determine_preset(self, complexity: float) -> ResearchPreset:
        """
        복잡도 기반 자동 프리셋 결정.
        
        Args:
            complexity: 복잡도 점수 (0.0 ~ 1.0)
        
        Returns:
            ResearchPreset
        """
        if complexity < 0.3:
            return ResearchPreset.QUICK
        elif complexity < 0.7:
            return ResearchPreset.MEDIUM
        else:
            return ResearchPreset.DEEP
    
    def _estimate_complexity_for_preset(self, preset: ResearchPreset) -> float:
        """
        프리셋에 대한 예상 복잡도 점수.
        
        Args:
            preset: ResearchPreset
        
        Returns:
            예상 복잡도 점수
        """
        preset_complexity = {
            ResearchPreset.QUICK: 0.2,
            ResearchPreset.MEDIUM: 0.5,
            ResearchPreset.DEEP: 0.8,
            ResearchPreset.AUTO: 0.5,
        }
        return preset_complexity.get(preset, 0.5)
    
    def adjust_depth_progressively(
        self,
        current_depth: DepthConfig,
        progress: Dict[str, Any],
        goals_achieved: bool = False
    ) -> Optional[DepthConfig]:
        """
        Progressive Deepening: 연구 진행에 따라 깊이를 점진적으로 증가.
        
        Args:
            current_depth: 현재 깊이 설정
            progress: 연구 진행 상황
            goals_achieved: 목표 달성 여부
        
        Returns:
            조정된 DepthConfig (조정 불필요 시 None)
        """
        # 목표 달성 시 조기 종료
        if goals_achieved and current_depth.researching.get("iteration_mode") == "flexible":
            logger.info("Goals achieved, no depth adjustment needed")
            return None
        
        # 진행 상황 분석
        iteration_count = progress.get("iteration_count", 0)
        max_iterations = current_depth.researching.get("max_iterations", 5)
        completion_rate = progress.get("completion_rate", 0.0)
        
        # 깊이 증가 조건: 반복 횟수가 많고 완성도가 낮을 때
        if iteration_count >= max_iterations * 0.7 and completion_rate < 0.6:
            # 깊이 증가
            if current_depth.preset == ResearchPreset.QUICK:
                new_preset = ResearchPreset.MEDIUM
            elif current_depth.preset == ResearchPreset.MEDIUM:
                new_preset = ResearchPreset.DEEP
            else:
                return None  # 이미 최대 깊이
            
            logger.info(f"Progressive deepening: {current_depth.preset.value} -> {new_preset.value}")
            configs = self._load_preset_configs()
            preset_config = configs.get(new_preset.value, configs["medium"])
            
            return DepthConfig(
                preset=new_preset,
                planning=preset_config.get("planning", {}),
                researching=preset_config.get("researching", {}),
                reporting=preset_config.get("reporting", {}),
                complexity_score=current_depth.complexity_score + 0.2
            )
        
        return None

