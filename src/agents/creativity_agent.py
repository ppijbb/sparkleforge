#!/usr/bin/env python3
"""
Creativity Agent for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade 창의성 에이전트.
기존 아이디어들을 결합하여 새로운 해법을 제시하는
창의적 사고와 혁신적 접근을 제공합니다.

2025년 10월 최신 기술 스택:
- Gemini 2.5 Flash Lite for creative thinking
- Analogical reasoning patterns
- Cross-domain synthesis
- Production-grade creativity engine
"""

import asyncio
import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import itertools
from collections import defaultdict

# MCP integration
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.mcp_integration import execute_tool
from src.core.llm_manager import execute_llm_task, TaskType
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class CreativityType(Enum):
    """창의성 타입."""
    ANALOGICAL = "analogical"  # 유추적 추론
    CROSS_DOMAIN = "cross_domain"  # 교차 도메인 결합
    LATERAL = "lateral"  # 측면적 사고
    CONVERGENT = "convergent"  # 수렴적 사고
    DIVERGENT = "divergent"  # 발산적 사고


@dataclass
class CreativeInsight:
    """창의적 인사이트."""
    insight_id: str
    type: CreativityType
    title: str
    description: str
    related_concepts: List[str]
    confidence: float
    novelty_score: float
    applicability_score: float
    reasoning: str
    examples: List[str]
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


@dataclass
class IdeaCombination:
    """아이디어 조합."""
    combination_id: str
    base_ideas: List[str]
    combined_idea: str
    connection_strength: float
    novelty_level: str
    potential_impact: str
    implementation_difficulty: str
    reasoning: str


class CreativityAgent:
    """
    Production-grade 창의성 에이전트.
    
    Features:
    - 기존 아이디어들을 결합하여 새로운 해법 제시
    - 유추적 추론을 통한 창의적 접근
    - 교차 도메인 결합으로 혁신적 솔루션 생성
    - 지속적인 창의적 사고 프로세스
    """
    
    def __init__(self):
        """창의성 에이전트 초기화."""
        self.creativity_patterns = self._initialize_creativity_patterns()
        self.domain_knowledge = self._initialize_domain_knowledge()
        self.analogical_mappings = self._initialize_analogical_mappings()
        
        # 창의성 설정
        self.config = {
            'max_insights_per_request': 5,
            'min_confidence_threshold': 0.6,
            'novelty_weight': 0.4,
            'applicability_weight': 0.6,
            'cross_domain_threshold': 0.7
        }
        
        logger.info("CreativityAgent initialized with production-grade creativity engine")
    
    def _initialize_creativity_patterns(self) -> Dict[str, List[str]]:
        """창의성 패턴을 초기화합니다."""
        return {
            'analogical': [
                "How does this work in nature?",
                "What if we applied this to a completely different field?",
                "How do other industries solve similar problems?",
                "What can we learn from historical examples?",
                "How would this work in a different culture?"
            ],
            'cross_domain': [
                "Combine {domain1} principles with {domain2} methods",
                "Apply {domain1} thinking to {domain2} problems",
                "Merge {domain1} and {domain2} approaches",
                "Use {domain1} tools for {domain2} challenges",
                "Integrate {domain1} concepts into {domain2} solutions"
            ],
            'lateral': [
                "What if we did the opposite?",
                "How can we make this more absurd?",
                "What if we removed the main constraint?",
                "How would a child approach this?",
                "What if we had unlimited resources?"
            ],
            'convergent': [
                "What are the common elements?",
                "How can we unify these approaches?",
                "What's the core principle here?",
                "How do these connect?",
                "What's the underlying pattern?"
            ],
            'divergent': [
                "What are all possible variations?",
                "How many different ways can we approach this?",
                "What if we changed every aspect?",
                "How can we make this completely different?",
                "What are the wildest possibilities?"
            ]
        }
    
    def _initialize_domain_knowledge(self) -> Dict[str, List[str]]:
        """도메인 지식을 초기화합니다."""
        return {
            'technology': ['AI', 'blockchain', 'IoT', 'quantum computing', 'biotechnology'],
            'business': ['strategy', 'marketing', 'finance', 'operations', 'leadership'],
            'science': ['physics', 'chemistry', 'biology', 'mathematics', 'psychology'],
            'art': ['design', 'music', 'literature', 'visual arts', 'performance'],
            'nature': ['evolution', 'ecosystems', 'adaptation', 'symbiosis', 'emergence'],
            'society': ['culture', 'politics', 'economics', 'education', 'healthcare']
        }
    
    def _initialize_analogical_mappings(self) -> Dict[str, List[str]]:
        """유추적 매핑을 초기화합니다."""
        return {
            'biological': ['evolution', 'adaptation', 'ecosystem', 'symbiosis', 'growth'],
            'mechanical': ['engine', 'machine', 'system', 'mechanism', 'automation'],
            'social': ['community', 'network', 'collaboration', 'communication', 'culture'],
            'mathematical': ['algorithm', 'formula', 'pattern', 'sequence', 'optimization'],
            'artistic': ['composition', 'harmony', 'rhythm', 'balance', 'expression']
        }
    
    async def generate_creative_insights(
        self,
        context: str,
        current_ideas: List[str],
        creativity_types: Optional[List[CreativityType]] = None
    ) -> List[CreativeInsight]:
        """
        창의적 인사이트를 생성합니다.
        
        Args:
            context: 현재 상황/문제 맥락
            current_ideas: 기존 아이디어들
            creativity_types: 창의성 타입 (선택사항)
            
        Returns:
            List[CreativeInsight]: 창의적 인사이트들
        """
        try:
            if creativity_types is None:
                creativity_types = [
                    CreativityType.ANALOGICAL,
                    CreativityType.CROSS_DOMAIN,
                    CreativityType.LATERAL
                ]
            
            insights = []
            
            for creativity_type in creativity_types:
                type_insights = await self._generate_insights_by_type(
                    context, current_ideas, creativity_type
                )
                insights.extend(type_insights)
            
            # 인사이트 정렬 및 필터링
            filtered_insights = self._filter_and_rank_insights(insights)
            
            logger.info(f"Generated {len(filtered_insights)} creative insights")
            return filtered_insights
            
        except Exception as e:
            logger.error(f"Failed to generate creative insights: {e}")
            return []
    
    async def _generate_insights_by_type(
        self,
        context: str,
        current_ideas: List[str],
        creativity_type: CreativityType
    ) -> List[CreativeInsight]:
        """타입별 창의적 인사이트를 생성합니다."""
        try:
            if creativity_type == CreativityType.ANALOGICAL:
                return await self._generate_analogical_insights(context, current_ideas)
            elif creativity_type == CreativityType.CROSS_DOMAIN:
                return await self._generate_cross_domain_insights(context, current_ideas)
            elif creativity_type == CreativityType.LATERAL:
                return await self._generate_lateral_insights(context, current_ideas)
            elif creativity_type == CreativityType.CONVERGENT:
                return await self._generate_convergent_insights(context, current_ideas)
            elif creativity_type == CreativityType.DIVERGENT:
                return await self._generate_divergent_insights(context, current_ideas)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to generate {creativity_type.value} insights: {e}")
            return []
    
    async def _generate_analogical_insights(
        self,
        context: str,
        current_ideas: List[str]
    ) -> List[CreativeInsight]:
        """유추적 인사이트를 생성합니다."""
        try:
            # 유추적 추론 프롬프트
            analogical_prompt = f"""
            Apply analogical reasoning to generate creative insights for the following context:
            
            Context: {context}
            Current Ideas: {', '.join(current_ideas)}
            
            Use these analogical frameworks:
            - Biological systems (evolution, adaptation, ecosystems)
            - Mechanical systems (engines, machines, automation)
            - Social systems (communities, networks, collaboration)
            - Mathematical systems (algorithms, patterns, optimization)
            - Artistic systems (composition, harmony, expression)
            
            Generate 3-5 analogical insights that:
            1. Draw parallels from different domains
            2. Identify transferable principles
            3. Suggest novel applications
            4. Provide concrete examples
            
            Return JSON format:
            {{
                "insights": [
                    {{
                        "title": "Insight title",
                        "description": "Detailed description",
                        "analogical_source": "biological/mechanical/social/mathematical/artistic",
                        "related_concepts": ["concept1", "concept2"],
                        "reasoning": "Why this analogy works",
                        "examples": ["example1", "example2"],
                        "confidence": 0.0-1.0,
                        "novelty_score": 0.0-1.0,
                        "applicability_score": 0.0-1.0
                    }}
                ]
            }}
            """
            
            result = await execute_llm_task(
                prompt=analogical_prompt,
                task_type=TaskType.CREATIVE,
                system_message=self.config.prompts["analogical_reasoning"]["system_message"]
            )
            
            analysis = json.loads(result.content)
            insights = []
            
            for insight_data in analysis.get('insights', []):
                insight = CreativeInsight(
                    insight_id=f"analogical_{len(insights) + 1}",
                    type=CreativityType.ANALOGICAL,
                    title=insight_data['title'],
                    description=insight_data['description'],
                    related_concepts=insight_data['related_concepts'],
                    confidence=insight_data['confidence'],
                    novelty_score=insight_data['novelty_score'],
                    applicability_score=insight_data['applicability_score'],
                    reasoning=insight_data['reasoning'],
                    examples=insight_data['examples'],
                    metadata={
                        'analogical_source': insight_data['analogical_source'],
                        'generation_method': 'analogical_reasoning'
                    }
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate analogical insights: {e}")
            return []
    
    async def _generate_cross_domain_insights(
        self,
        context: str,
        current_ideas: List[str]
    ) -> List[CreativeInsight]:
        """교차 도메인 인사이트를 생성합니다."""
        try:
            # 도메인 조합 생성
            domains = list(self.domain_knowledge.keys())
            domain_combinations = list(itertools.combinations(domains, 2))
            
            # 랜덤하게 3개 조합 선택
            selected_combinations = random.sample(domain_combinations, min(3, len(domain_combinations)))
            
            insights = []
            
            for domain1, domain2 in selected_combinations:
                cross_domain_prompt = f"""
                Generate cross-domain insights by combining {domain1} and {domain2}:
                
                Context: {context}
                Current Ideas: {', '.join(current_ideas)}
                
                {domain1} concepts: {', '.join(self.domain_knowledge[domain1])}
                {domain2} concepts: {', '.join(self.domain_knowledge[domain2])}
                
                Create innovative insights that:
                1. Merge principles from both domains
                2. Identify unexpected connections
                3. Suggest novel applications
                4. Provide implementation ideas
                
                Return JSON format:
                {{
                    "insights": [
                        {{
                            "title": "Cross-domain insight title",
                            "description": "Detailed description",
                            "domain1_concepts": ["concept1", "concept2"],
                            "domain2_concepts": ["concept1", "concept2"],
                            "fusion_approach": "How the domains are combined",
                            "related_concepts": ["merged_concept1", "merged_concept2"],
                            "reasoning": "Why this combination works",
                            "examples": ["example1", "example2"],
                            "confidence": 0.0-1.0,
                            "novelty_score": 0.0-1.0,
                            "applicability_score": 0.0-1.0
                        }}
                    ]
                }}
                """
                
                result = await execute_llm_task(
                    prompt=cross_domain_prompt,
                    task_type=TaskType.CREATIVE,
                    system_message=f"You are a cross-domain innovation expert specializing in {domain1} and {domain2} integration."
                )
                
                analysis = json.loads(result.content)
                
                for insight_data in analysis.get('insights', []):
                    insight = CreativeInsight(
                        insight_id=f"cross_domain_{domain1}_{domain2}_{len(insights) + 1}",
                        type=CreativityType.CROSS_DOMAIN,
                        title=insight_data['title'],
                        description=insight_data['description'],
                        related_concepts=insight_data['related_concepts'],
                        confidence=insight_data['confidence'],
                        novelty_score=insight_data['novelty_score'],
                        applicability_score=insight_data['applicability_score'],
                        reasoning=insight_data['reasoning'],
                        examples=insight_data['examples'],
                        metadata={
                            'domain1': domain1,
                            'domain2': domain2,
                            'domain1_concepts': insight_data['domain1_concepts'],
                            'domain2_concepts': insight_data['domain2_concepts'],
                            'fusion_approach': insight_data['fusion_approach'],
                            'generation_method': 'cross_domain_synthesis'
                        }
                    )
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate cross-domain insights: {e}")
            return []
    
    async def _generate_lateral_insights(
        self,
        context: str,
        current_ideas: List[str]
    ) -> List[CreativeInsight]:
        """측면적 사고 인사이트를 생성합니다."""
        try:
            lateral_prompt = f"""
            Apply lateral thinking to generate unconventional insights:
            
            Context: {context}
            Current Ideas: {', '.join(current_ideas)}
            
            Use these lateral thinking techniques:
            - Challenge assumptions
            - Reverse thinking
            - Random word association
            - Provocative operations
            - Alternative perspectives
            
            Generate 3-5 lateral insights that:
            1. Challenge conventional approaches
            2. Offer unexpected perspectives
            3. Suggest unconventional solutions
            4. Break mental models
            
            Return JSON format:
            {{
                "insights": [
                    {{
                        "title": "Lateral insight title",
                        "description": "Unconventional description",
                        "lateral_technique": "assumption_challenge/reverse_thinking/random_association/provocative/alternative_perspective",
                        "challenged_assumption": "What assumption is being challenged",
                        "related_concepts": ["concept1", "concept2"],
                        "reasoning": "Why this lateral approach works",
                        "examples": ["example1", "example2"],
                        "confidence": 0.0-1.0,
                        "novelty_score": 0.0-1.0,
                        "applicability_score": 0.0-1.0
                    }}
                ]
            }}
            """
            
            result = await execute_llm_task(
                prompt=lateral_prompt,
                task_type=TaskType.CREATIVE,
                system_message=self.config.prompts["lateral_thinking"]["system_message"]
            )
            
            analysis = json.loads(result.content)
            insights = []
            
            for insight_data in analysis.get('insights', []):
                insight = CreativeInsight(
                    insight_id=f"lateral_{len(insights) + 1}",
                    type=CreativityType.LATERAL,
                    title=insight_data['title'],
                    description=insight_data['description'],
                    related_concepts=insight_data['related_concepts'],
                    confidence=insight_data['confidence'],
                    novelty_score=insight_data['novelty_score'],
                    applicability_score=insight_data['applicability_score'],
                    reasoning=insight_data['reasoning'],
                    examples=insight_data['examples'],
                    metadata={
                        'lateral_technique': insight_data['lateral_technique'],
                        'challenged_assumption': insight_data['challenged_assumption'],
                        'generation_method': 'lateral_thinking'
                    }
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate lateral insights: {e}")
            return []
    
    async def _generate_convergent_insights(
        self,
        context: str,
        current_ideas: List[str]
    ) -> List[CreativeInsight]:
        """수렴적 사고 인사이트를 생성합니다."""
        try:
            convergent_prompt = f"""
            Apply convergent thinking to find common patterns and unifying principles:
            
            Context: {context}
            Current Ideas: {', '.join(current_ideas)}
            
            Look for:
            - Common underlying principles
            - Unifying patterns
            - Core elements
            - Essential characteristics
            - Fundamental concepts
            
            Generate 3-5 convergent insights that:
            1. Identify common patterns
            2. Unify different approaches
            3. Find core principles
            4. Synthesize key elements
            
            Return JSON format:
            {{
                "insights": [
                    {{
                        "title": "Convergent insight title",
                        "description": "Unifying description",
                        "unified_elements": ["element1", "element2"],
                        "core_principle": "The fundamental principle",
                        "related_concepts": ["concept1", "concept2"],
                        "reasoning": "Why these elements converge",
                        "examples": ["example1", "example2"],
                        "confidence": 0.0-1.0,
                        "novelty_score": 0.0-1.0,
                        "applicability_score": 0.0-1.0
                    }}
                ]
            }}
            """
            
            result = await execute_llm_task(
                prompt=convergent_prompt,
                task_type=TaskType.CREATIVE,
                system_message=self.config.prompts["convergent_thinking"]["system_message"]
            )
            
            analysis = json.loads(result.content)
            insights = []
            
            for insight_data in analysis.get('insights', []):
                insight = CreativeInsight(
                    insight_id=f"convergent_{len(insights) + 1}",
                    type=CreativityType.CONVERGENT,
                    title=insight_data['title'],
                    description=insight_data['description'],
                    related_concepts=insight_data['related_concepts'],
                    confidence=insight_data['confidence'],
                    novelty_score=insight_data['novelty_score'],
                    applicability_score=insight_data['applicability_score'],
                    reasoning=insight_data['reasoning'],
                    examples=insight_data['examples'],
                    metadata={
                        'unified_elements': insight_data['unified_elements'],
                        'core_principle': insight_data['core_principle'],
                        'generation_method': 'convergent_thinking'
                    }
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate convergent insights: {e}")
            return []
    
    async def _generate_divergent_insights(
        self,
        context: str,
        current_ideas: List[str]
    ) -> List[CreativeInsight]:
        """발산적 사고 인사이트를 생성합니다."""
        try:
            divergent_prompt = f"""
            Apply divergent thinking to explore all possible variations and alternatives:
            
            Context: {context}
            Current Ideas: {', '.join(current_ideas)}
            
            Generate as many variations as possible:
            - Different approaches
            - Alternative methods
            - Various perspectives
            - Multiple solutions
            - Diverse applications
            
            Generate 5-7 divergent insights that:
            1. Explore all possible directions
            2. Generate multiple alternatives
            3. Consider various perspectives
            4. Suggest diverse applications
            
            Return JSON format:
            {{
                "insights": [
                    {{
                        "title": "Divergent insight title",
                        "description": "Alternative description",
                        "variation_type": "approach/method/perspective/solution/application",
                        "original_idea": "What this varies from",
                        "related_concepts": ["concept1", "concept2"],
                        "reasoning": "Why this variation is valuable",
                        "examples": ["example1", "example2"],
                        "confidence": 0.0-1.0,
                        "novelty_score": 0.0-1.0,
                        "applicability_score": 0.0-1.0
                    }}
                ]
            }}
            """
            
            result = await execute_llm_task(
                prompt=divergent_prompt,
                task_type=TaskType.CREATIVE,
                system_message=self.config.prompts["divergent_thinking"]["system_message"]
            )
            
            analysis = json.loads(result.content)
            insights = []
            
            for insight_data in analysis.get('insights', []):
                insight = CreativeInsight(
                    insight_id=f"divergent_{len(insights) + 1}",
                    type=CreativityType.DIVERGENT,
                    title=insight_data['title'],
                    description=insight_data['description'],
                    related_concepts=insight_data['related_concepts'],
                    confidence=insight_data['confidence'],
                    novelty_score=insight_data['novelty_score'],
                    applicability_score=insight_data['applicability_score'],
                    reasoning=insight_data['reasoning'],
                    examples=insight_data['examples'],
                    metadata={
                        'variation_type': insight_data['variation_type'],
                        'original_idea': insight_data['original_idea'],
                        'generation_method': 'divergent_thinking'
                    }
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate divergent insights: {e}")
            return []
    
    def _filter_and_rank_insights(self, insights: List[CreativeInsight]) -> List[CreativeInsight]:
        """인사이트를 필터링하고 순위를 매깁니다."""
        try:
            # 최소 임계값 필터링
            filtered_insights = [
                insight for insight in insights
                if insight.confidence >= self.config['min_confidence_threshold']
            ]
            
            # 종합 점수 계산
            for insight in filtered_insights:
                insight.metadata['composite_score'] = (
                    insight.novelty_score * self.config['novelty_weight'] +
                    insight.applicability_score * self.config['applicability_weight']
                )
            
            # 점수순 정렬
            filtered_insights.sort(
                key=lambda x: (x.metadata['composite_score'], x.confidence),
                reverse=True
            )
            
            # 최대 개수 제한
            return filtered_insights[:self.config['max_insights_per_request']]
            
        except Exception as e:
            logger.error(f"Failed to filter and rank insights: {e}")
            return insights
    
    async def combine_ideas(
        self,
        ideas: List[str],
        combination_strategy: str = "random"
    ) -> List[IdeaCombination]:
        """아이디어들을 조합합니다."""
        try:
            if len(ideas) < 2:
                return []
            
            combinations = []
            
            if combination_strategy == "random":
                # 랜덤 조합
                for _ in range(min(5, len(ideas) * 2)):
                    base_ideas = random.sample(ideas, min(3, len(ideas)))
                    combination = await self._create_idea_combination(base_ideas)
                    if combination:
                        combinations.append(combination)
            
            elif combination_strategy == "systematic":
                # 체계적 조합
                for r in range(2, min(4, len(ideas) + 1)):
                    for combo in itertools.combinations(ideas, r):
                        combination = await self._create_idea_combination(list(combo))
                        if combination:
                            combinations.append(combination)
            
            # 조합 정렬
            combinations.sort(key=lambda x: x.connection_strength, reverse=True)
            
            return combinations[:10]  # 최대 10개
            
        except Exception as e:
            logger.error(f"Failed to combine ideas: {e}")
            return []
    
    async def _create_idea_combination(self, base_ideas: List[str]) -> Optional[IdeaCombination]:
        """아이디어 조합을 생성합니다."""
        try:
            combination_prompt = f"""
            Combine these ideas into a novel solution:
            
            Ideas: {', '.join(base_ideas)}
            
            Create a combination that:
            1. Synthesizes the core elements
            2. Identifies unexpected connections
            3. Suggests practical applications
            4. Explains the reasoning
            
            Return JSON format:
            {{
                "combined_idea": "Synthesized idea description",
                "connection_strength": 0.0-1.0,
                "novelty_level": "low/medium/high",
                "potential_impact": "low/medium/high",
                "implementation_difficulty": "easy/medium/hard",
                "reasoning": "Why this combination works"
            }}
            """
            
            result = await execute_llm_task(
                prompt=combination_prompt,
                task_type=TaskType.CREATIVE,
                system_message=self.config.prompts["idea_combination"]["system_message"]
            )
            
            analysis = json.loads(result.content)
            
            return IdeaCombination(
                combination_id=f"combo_{hash('_'.join(base_ideas))}",
                base_ideas=base_ideas,
                combined_idea=analysis['combined_idea'],
                connection_strength=analysis['connection_strength'],
                novelty_level=analysis['novelty_level'],
                potential_impact=analysis['potential_impact'],
                implementation_difficulty=analysis['implementation_difficulty'],
                reasoning=analysis['reasoning']
            )
            
        except Exception as e:
            logger.error(f"Failed to create idea combination: {e}")
            return None
    
    def get_creativity_stats(self) -> Dict[str, Any]:
        """창의성 에이전트 통계를 반환합니다."""
        return {
            'creativity_patterns': len(self.creativity_patterns),
            'domain_knowledge': len(self.domain_knowledge),
            'analogical_mappings': len(self.analogical_mappings),
            'config': self.config
        }
