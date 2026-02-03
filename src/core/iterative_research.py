"""
Iterative Deep-Research Paradigm

DeepResearch (Alibaba-NLP) 영감을 받은 반복적 깊은 연구 시스템.
라운드 기반 Think/Report/Action 패턴으로 복잡한 주제를 점진적으로 탐색.

핵심 특징:
- Round-based research with workspace reconstruction
- Think/Report/Action 분리로 context bloat 방지
- Evolving Summary Report as central memory
- Quality threshold-based termination
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, TypedDict
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ResearchPhase(Enum):
    """연구 라운드의 현재 단계."""
    THINK = "think"
    REPORT = "report"
    ACTION = "action"
    RECONSTRUCT = "reconstruct"
    COMPLETE = "complete"


class QualityMetrics(BaseModel):
    """연구 품질 측정 지표."""
    completeness: float = Field(default=0.0, ge=0.0, le=1.0, description="주제 완성도")
    depth: float = Field(default=0.0, ge=0.0, le=1.0, description="분석 깊이")
    source_diversity: float = Field(default=0.0, ge=0.0, le=1.0, description="소스 다양성")
    factual_accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="사실적 정확성")
    coherence: float = Field(default=0.0, ge=0.0, le=1.0, description="일관성")
    
    @property
    def overall_score(self) -> float:
        """전체 품질 점수 (가중 평균)."""
        weights = {
            "completeness": 0.25,
            "depth": 0.25,
            "source_diversity": 0.15,
            "factual_accuracy": 0.20,
            "coherence": 0.15
        }
        return sum(
            getattr(self, k) * v 
            for k, v in weights.items()
        )


class ThinkOutput(BaseModel):
    """Think 단계의 출력."""
    current_understanding: str = Field(description="현재까지의 이해")
    knowledge_gaps: List[str] = Field(default_factory=list, description="지식 공백")
    next_research_directions: List[str] = Field(default_factory=list, description="다음 연구 방향")
    hypotheses: List[str] = Field(default_factory=list, description="검증할 가설들")
    confidence_level: float = Field(default=0.0, ge=0.0, le=1.0, description="현재 신뢰도")


class ReportOutput(BaseModel):
    """Report 단계의 출력 (Evolving Summary)."""
    round_number: int = Field(description="현재 라운드 번호")
    executive_summary: str = Field(description="핵심 요약")
    key_findings: List[Dict[str, Any]] = Field(default_factory=list, description="주요 발견사항")
    sources_used: List[Dict[str, str]] = Field(default_factory=list, description="사용된 소스")
    quality_metrics: QualityMetrics = Field(default_factory=QualityMetrics, description="품질 지표")
    remaining_questions: List[str] = Field(default_factory=list, description="남은 질문들")


class ActionOutput(BaseModel):
    """Action 단계의 출력."""
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list, description="수행한 액션들")
    new_information: List[Dict[str, Any]] = Field(default_factory=list, description="새로 획득한 정보")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="도구 호출 기록")
    errors_encountered: List[str] = Field(default_factory=list, description="발생한 오류들")


class RoundState(BaseModel):
    """개별 라운드의 상태."""
    round_number: int
    phase: ResearchPhase = ResearchPhase.THINK
    think_output: Optional[ThinkOutput] = None
    report_output: Optional[ReportOutput] = None
    action_output: Optional[ActionOutput] = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True


class IterativeResearchState(BaseModel):
    """반복 연구 전체 상태."""
    query: str = Field(description="원본 연구 질문")
    session_id: str = Field(description="세션 ID")
    current_round: int = Field(default=1, description="현재 라운드")
    max_rounds: int = Field(default=5, description="최대 라운드 수")
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="종료 품질 임계값")
    
    # Evolving Summary Report (중앙 메모리)
    evolving_summary: str = Field(default="", description="진화하는 요약 보고서")
    accumulated_findings: List[Dict[str, Any]] = Field(default_factory=list, description="누적 발견사항")
    all_sources: List[Dict[str, str]] = Field(default_factory=list, description="모든 사용 소스")
    
    # 라운드 히스토리 (Think 제외 - context bloat 방지)
    round_reports: List[ReportOutput] = Field(default_factory=list, description="라운드별 보고서")
    
    # 종료 조건
    is_complete: bool = Field(default=False, description="연구 완료 여부")
    termination_reason: Optional[str] = Field(default=None, description="종료 이유")
    
    # 현재 라운드 상태
    current_round_state: Optional[RoundState] = None
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class WorkspaceContext:
    """라운드 간 전달되는 lean workspace context."""
    evolving_summary: str
    last_report: Optional[ReportOutput]
    knowledge_gaps: List[str]
    remaining_questions: List[str]
    quality_score: float
    round_number: int


class WorkspaceReconstructor:
    """
    라운드 간 workspace 재구성기.
    
    Think 출력을 다음 라운드로 전달하지 않고,
    Report의 핵심 정보만 추출하여 lean context 구성.
    """
    
    def __init__(self, max_summary_tokens: int = 4000):
        self.max_summary_tokens = max_summary_tokens
    
    def reconstruct(
        self,
        state: IterativeResearchState,
        new_report: ReportOutput
    ) -> WorkspaceContext:
        """
        새 라운드를 위한 workspace 재구성.
        
        Args:
            state: 현재 연구 상태
            new_report: 새로 생성된 보고서
            
        Returns:
            다음 라운드를 위한 lean workspace context
        """
        # 진화하는 요약 업데이트
        updated_summary = self._merge_summaries(
            state.evolving_summary,
            new_report.executive_summary
        )
        
        # Knowledge gaps와 remaining questions 추출
        knowledge_gaps = new_report.remaining_questions[:5]  # 상위 5개만
        
        return WorkspaceContext(
            evolving_summary=updated_summary,
            last_report=new_report,
            knowledge_gaps=knowledge_gaps,
            remaining_questions=new_report.remaining_questions,
            quality_score=new_report.quality_metrics.overall_score,
            round_number=state.current_round
        )
    
    def _merge_summaries(self, existing: str, new: str) -> str:
        """기존 요약과 새 요약 병합."""
        if not existing:
            return new
        
        # 간단한 병합 (실제 구현에서는 LLM 사용)
        merged = f"{existing}\n\n[Round Update]\n{new}"
        
        # 토큰 제한 적용 (간단한 문자 기반)
        max_chars = self.max_summary_tokens * 4  # 대략적 추정
        if len(merged) > max_chars:
            # 오래된 부분 압축
            merged = merged[-max_chars:]
            
        return merged


class IterativeResearchEngine:
    """
    반복적 깊은 연구 엔진.
    
    Think → Report → Action → Reconstruct 사이클을 관리.
    """
    
    def __init__(
        self,
        max_rounds: int = 5,
        quality_threshold: float = 0.8,
        min_improvement_threshold: float = 0.05,
        workspace_reconstructor: Optional[WorkspaceReconstructor] = None
    ):
        self.max_rounds = max_rounds
        self.quality_threshold = quality_threshold
        self.min_improvement_threshold = min_improvement_threshold
        self.workspace_reconstructor = workspace_reconstructor or WorkspaceReconstructor()
        
        # Callbacks
        self.on_round_start: Optional[Callable] = None
        self.on_think_complete: Optional[Callable] = None
        self.on_report_complete: Optional[Callable] = None
        self.on_action_complete: Optional[Callable] = None
        self.on_round_complete: Optional[Callable] = None
        
        logger.info(
            f"IterativeResearchEngine initialized: "
            f"max_rounds={max_rounds}, quality_threshold={quality_threshold}"
        )
    
    async def run(
        self,
        query: str,
        session_id: str,
        think_fn: Callable,
        report_fn: Callable,
        action_fn: Callable,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> IterativeResearchState:
        """
        반복적 연구 실행.
        
        Args:
            query: 연구 질문
            session_id: 세션 ID
            think_fn: Think 단계 함수 (async)
            report_fn: Report 단계 함수 (async)
            action_fn: Action 단계 함수 (async)
            initial_context: 초기 컨텍스트
            
        Returns:
            최종 연구 상태
        """
        # 상태 초기화
        state = IterativeResearchState(
            query=query,
            session_id=session_id,
            max_rounds=self.max_rounds,
            quality_threshold=self.quality_threshold
        )
        
        previous_quality = 0.0
        stagnation_count = 0
        max_stagnation = 2  # 연속 2라운드 개선 없으면 종료
        
        logger.info(f"🔄 Starting iterative research for query: {query[:100]}...")
        
        while not state.is_complete and state.current_round <= self.max_rounds:
            round_start = datetime.now()
            
            # 라운드 시작 콜백
            if self.on_round_start:
                await self._safe_callback(self.on_round_start, state)
            
            logger.info(f"📍 Round {state.current_round}/{self.max_rounds} starting...")
            
            # 라운드 상태 초기화
            state.current_round_state = RoundState(round_number=state.current_round)
            
            # Workspace context 준비
            workspace = self._prepare_workspace(state)
            
            try:
                # 1. THINK 단계
                state.current_round_state.phase = ResearchPhase.THINK
                think_output = await think_fn(
                    query=query,
                    workspace=workspace,
                    round_number=state.current_round
                )
                state.current_round_state.think_output = think_output
                
                if self.on_think_complete:
                    await self._safe_callback(self.on_think_complete, state, think_output)
                
                logger.info(f"💭 Think complete: {len(think_output.knowledge_gaps)} gaps identified")
                
                # 2. ACTION 단계 (Think 기반으로 정보 수집)
                state.current_round_state.phase = ResearchPhase.ACTION
                action_output = await action_fn(
                    query=query,
                    think_output=think_output,
                    workspace=workspace
                )
                state.current_round_state.action_output = action_output
                
                if self.on_action_complete:
                    await self._safe_callback(self.on_action_complete, state, action_output)
                
                logger.info(f"⚡ Action complete: {len(action_output.new_information)} new items")
                
                # 3. REPORT 단계 (Think + Action 결과 종합)
                state.current_round_state.phase = ResearchPhase.REPORT
                report_output = await report_fn(
                    query=query,
                    think_output=think_output,
                    action_output=action_output,
                    workspace=workspace,
                    round_number=state.current_round
                )
                state.current_round_state.report_output = report_output
                
                if self.on_report_complete:
                    await self._safe_callback(self.on_report_complete, state, report_output)
                
                logger.info(
                    f"📊 Report complete: quality={report_output.quality_metrics.overall_score:.2f}"
                )
                
                # 4. RECONSTRUCT 단계
                state.current_round_state.phase = ResearchPhase.RECONSTRUCT
                new_workspace = self.workspace_reconstructor.reconstruct(state, report_output)
                
                # 상태 업데이트 (Think 출력은 저장하지 않음 - context bloat 방지)
                state.evolving_summary = new_workspace.evolving_summary
                state.round_reports.append(report_output)
                state.accumulated_findings.extend(report_output.key_findings)
                state.all_sources.extend(report_output.sources_used)
                
                # 종료 조건 확인
                current_quality = report_output.quality_metrics.overall_score
                
                # 품질 임계값 도달
                if current_quality >= self.quality_threshold:
                    state.is_complete = True
                    state.termination_reason = f"Quality threshold reached: {current_quality:.2f} >= {self.quality_threshold}"
                    logger.info(f"✅ {state.termination_reason}")
                
                # 개선 정체 확인
                improvement = current_quality - previous_quality
                if improvement < self.min_improvement_threshold:
                    stagnation_count += 1
                    if stagnation_count >= max_stagnation:
                        state.is_complete = True
                        state.termination_reason = f"Improvement stagnated for {stagnation_count} rounds"
                        logger.info(f"⚠️ {state.termination_reason}")
                else:
                    stagnation_count = 0
                
                previous_quality = current_quality
                
                # 라운드 완료
                state.current_round_state.phase = ResearchPhase.COMPLETE
                state.current_round_state.completed_at = datetime.now()
                
                if self.on_round_complete:
                    await self._safe_callback(self.on_round_complete, state)
                
                round_duration = (datetime.now() - round_start).total_seconds()
                logger.info(
                    f"🔄 Round {state.current_round} complete in {round_duration:.1f}s "
                    f"(quality: {current_quality:.2f})"
                )
                
                state.current_round += 1
                
            except Exception as e:
                logger.error(f"❌ Round {state.current_round} failed: {e}")
                state.termination_reason = f"Error in round {state.current_round}: {str(e)}"
                # 오류 발생해도 이전 결과는 유지하고 종료
                if state.round_reports:
                    state.is_complete = True
                else:
                    raise
        
        # 최대 라운드 도달
        if not state.is_complete:
            state.is_complete = True
            state.termination_reason = f"Max rounds ({self.max_rounds}) reached"
            logger.info(f"📍 {state.termination_reason}")
        
        return state
    
    def _prepare_workspace(self, state: IterativeResearchState) -> WorkspaceContext:
        """현재 상태에서 workspace context 준비."""
        last_report = state.round_reports[-1] if state.round_reports else None
        
        return WorkspaceContext(
            evolving_summary=state.evolving_summary,
            last_report=last_report,
            knowledge_gaps=last_report.remaining_questions[:5] if last_report else [],
            remaining_questions=last_report.remaining_questions if last_report else [],
            quality_score=last_report.quality_metrics.overall_score if last_report else 0.0,
            round_number=state.current_round
        )
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """안전한 콜백 실행."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Callback error (non-fatal): {e}")


class IterativeResearchNode:
    """
    LangGraph 통합을 위한 Iterative Research 노드.
    
    기존 agent_orchestrator.py의 워크플로우에 통합 가능.
    """
    
    def __init__(
        self,
        engine: Optional[IterativeResearchEngine] = None,
        llm_task_executor: Optional[Callable] = None
    ):
        self.engine = engine or IterativeResearchEngine()
        self.llm_task_executor = llm_task_executor
    
    async def think(
        self,
        query: str,
        workspace: WorkspaceContext,
        round_number: int
    ) -> ThinkOutput:
        """Think 단계 실행."""
        from src.core.llm_manager import execute_llm_task, TaskType
        
        prompt = self._build_think_prompt(query, workspace, round_number)
        
        result = await execute_llm_task(
            prompt=prompt,
            task_type=TaskType.DEEP_REASONING,
            temperature=0.7
        )
        
        return self._parse_think_output(result)
    
    async def report(
        self,
        query: str,
        think_output: ThinkOutput,
        action_output: ActionOutput,
        workspace: WorkspaceContext,
        round_number: int
    ) -> ReportOutput:
        """Report 단계 실행."""
        from src.core.llm_manager import execute_llm_task, TaskType
        
        prompt = self._build_report_prompt(
            query, think_output, action_output, workspace, round_number
        )
        
        result = await execute_llm_task(
            prompt=prompt,
            task_type=TaskType.SYNTHESIS,
            temperature=0.3
        )
        
        return self._parse_report_output(result, round_number)
    
    async def action(
        self,
        query: str,
        think_output: ThinkOutput,
        workspace: WorkspaceContext
    ) -> ActionOutput:
        """Action 단계 실행 (도구 호출 포함)."""
        actions_taken = []
        new_information = []
        tool_calls = []
        errors = []
        
        # Knowledge gaps 기반으로 연구 방향 결정
        for direction in think_output.next_research_directions[:3]:  # 상위 3개 방향
            try:
                # 여기서 실제 MCP 도구 호출 (예: search, fetch 등)
                # 실제 구현은 agent_orchestrator와 통합 시 MCP 도구 사용
                action_record = {
                    "direction": direction,
                    "type": "research",
                    "timestamp": datetime.now().isoformat()
                }
                actions_taken.append(action_record)
                
                # Placeholder for actual tool execution
                # 실제 구현에서는 MCP 도구를 통해 정보 수집
                
            except Exception as e:
                errors.append(f"Action failed for direction '{direction}': {str(e)}")
        
        return ActionOutput(
            actions_taken=actions_taken,
            new_information=new_information,
            tool_calls=tool_calls,
            errors_encountered=errors
        )
    
    def _build_think_prompt(
        self,
        query: str,
        workspace: WorkspaceContext,
        round_number: int
    ) -> str:
        """Think 단계 프롬프트 생성."""
        context_section = ""
        if workspace.evolving_summary:
            context_section = f"""
## Previous Research Summary
{workspace.evolving_summary}

## Known Knowledge Gaps
{chr(10).join(f"- {gap}" for gap in workspace.knowledge_gaps) if workspace.knowledge_gaps else "None identified yet"}

## Current Quality Score: {workspace.quality_score:.2f}
"""
        
        return f"""# Deep Research Think Phase (Round {round_number})

## Original Query
{query}

{context_section}

## Your Task
Analyze the current state of research and identify:
1. Your current understanding of the topic
2. Knowledge gaps that need to be filled
3. Next research directions to pursue
4. Hypotheses to verify
5. Your confidence level (0-1)

Respond in a structured format:

### Current Understanding
[Your synthesis of what is known]

### Knowledge Gaps
- [Gap 1]
- [Gap 2]
...

### Next Research Directions
- [Direction 1]
- [Direction 2]
...

### Hypotheses to Verify
- [Hypothesis 1]
- [Hypothesis 2]
...

### Confidence Level
[0.0 - 1.0]
"""
    
    def _build_report_prompt(
        self,
        query: str,
        think_output: ThinkOutput,
        action_output: ActionOutput,
        workspace: WorkspaceContext,
        round_number: int
    ) -> str:
        """Report 단계 프롬프트 생성."""
        new_info = "\n".join(
            f"- {info.get('content', str(info))}" 
            for info in action_output.new_information
        ) if action_output.new_information else "No new information gathered in this round."
        
        return f"""# Deep Research Report Phase (Round {round_number})

## Original Query
{query}

## Previous Summary
{workspace.evolving_summary if workspace.evolving_summary else "First round - no previous summary"}

## Current Understanding (from Think phase)
{think_output.current_understanding}

## New Information Gathered
{new_info}

## Your Task
Create an evolving summary report that:
1. Synthesizes all information gathered so far
2. Highlights key findings with evidence
3. Assesses quality metrics (completeness, depth, accuracy, etc.)
4. Identifies remaining questions

Respond in a structured format:

### Executive Summary
[Comprehensive summary of research findings]

### Key Findings
1. [Finding 1] - Evidence: [source]
2. [Finding 2] - Evidence: [source]
...

### Quality Assessment
- Completeness: [0.0-1.0]
- Depth: [0.0-1.0]
- Source Diversity: [0.0-1.0]
- Factual Accuracy: [0.0-1.0]
- Coherence: [0.0-1.0]

### Remaining Questions
- [Question 1]
- [Question 2]
...
"""
    
    def _parse_think_output(self, result: str) -> ThinkOutput:
        """Think 결과 파싱."""
        # 간단한 파싱 (실제 구현에서는 더 정교한 파싱 필요)
        lines = result.split('\n')
        
        current_understanding = ""
        knowledge_gaps = []
        directions = []
        hypotheses = []
        confidence = 0.5
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if "Current Understanding" in line:
                current_section = "understanding"
            elif "Knowledge Gaps" in line:
                current_section = "gaps"
            elif "Research Directions" in line or "Next Research" in line:
                current_section = "directions"
            elif "Hypotheses" in line:
                current_section = "hypotheses"
            elif "Confidence" in line:
                current_section = "confidence"
            elif line.startswith("- ") or line.startswith("* "):
                item = line[2:].strip()
                if current_section == "gaps":
                    knowledge_gaps.append(item)
                elif current_section == "directions":
                    directions.append(item)
                elif current_section == "hypotheses":
                    hypotheses.append(item)
            elif current_section == "understanding" and line:
                current_understanding += line + " "
            elif current_section == "confidence":
                try:
                    # 숫자 추출 시도
                    import re
                    match = re.search(r'([0-9.]+)', line)
                    if match:
                        confidence = min(1.0, max(0.0, float(match.group(1))))
                except:
                    pass
        
        return ThinkOutput(
            current_understanding=current_understanding.strip(),
            knowledge_gaps=knowledge_gaps,
            next_research_directions=directions,
            hypotheses=hypotheses,
            confidence_level=confidence
        )
    
    def _parse_report_output(self, result: str, round_number: int) -> ReportOutput:
        """Report 결과 파싱."""
        lines = result.split('\n')
        
        executive_summary = ""
        key_findings = []
        remaining_questions = []
        quality_metrics = QualityMetrics()
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if "Executive Summary" in line:
                current_section = "summary"
            elif "Key Findings" in line:
                current_section = "findings"
            elif "Quality Assessment" in line:
                current_section = "quality"
            elif "Remaining Questions" in line:
                current_section = "questions"
            elif current_section == "summary" and line and not line.startswith("#"):
                executive_summary += line + " "
            elif current_section == "findings" and (line.startswith("- ") or line.startswith("* ") or line[0:2].isdigit()):
                key_findings.append({"content": line.lstrip("0123456789.-* ").strip()})
            elif current_section == "questions" and (line.startswith("- ") or line.startswith("* ")):
                remaining_questions.append(line[2:].strip())
            elif current_section == "quality":
                try:
                    import re
                    if "Completeness" in line:
                        match = re.search(r'([0-9.]+)', line)
                        if match:
                            quality_metrics.completeness = float(match.group(1))
                    elif "Depth" in line:
                        match = re.search(r'([0-9.]+)', line)
                        if match:
                            quality_metrics.depth = float(match.group(1))
                    elif "Source Diversity" in line:
                        match = re.search(r'([0-9.]+)', line)
                        if match:
                            quality_metrics.source_diversity = float(match.group(1))
                    elif "Factual Accuracy" in line or "Accuracy" in line:
                        match = re.search(r'([0-9.]+)', line)
                        if match:
                            quality_metrics.factual_accuracy = float(match.group(1))
                    elif "Coherence" in line:
                        match = re.search(r'([0-9.]+)', line)
                        if match:
                            quality_metrics.coherence = float(match.group(1))
                except:
                    pass
        
        return ReportOutput(
            round_number=round_number,
            executive_summary=executive_summary.strip(),
            key_findings=key_findings,
            sources_used=[],
            quality_metrics=quality_metrics,
            remaining_questions=remaining_questions
        )


# Singleton instance
_iterative_research_engine: Optional[IterativeResearchEngine] = None


def get_iterative_research_engine(
    max_rounds: int = 5,
    quality_threshold: float = 0.8
) -> IterativeResearchEngine:
    """IterativeResearchEngine 싱글톤 인스턴스 반환."""
    global _iterative_research_engine
    
    if _iterative_research_engine is None:
        _iterative_research_engine = IterativeResearchEngine(
            max_rounds=max_rounds,
            quality_threshold=quality_threshold
        )
    
    return _iterative_research_engine
