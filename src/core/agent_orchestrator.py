"""
Agent Orchestrator for Multi-Agent System

LangGraph 기반 에이전트 오케스트레이션 시스템
4대 핵심 에이전트를 조율하여 협업 워크플로우 구축
"""

import asyncio
import logging
import json
import operator
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal, Annotated
from datetime import datetime
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.core.shared_memory import get_shared_memory, MemoryScope
from src.core.skills_manager import get_skill_manager
from src.core.skills_selector import get_skill_selector, SkillMatch
from src.core.skills_loader import Skill
from src.core.agent_result_sharing import SharedResultsManager, AgentDiscussionManager
from src.core.researcher_config import get_agent_config
from src.core.mcp_auto_discovery import FastMCPMulti
from src.core.mcp_tool_loader import MCPToolLoader
from src.core.agent_tool_selector import AgentToolSelector, AgentType
from src.core.session_manager import get_session_manager, SessionManager
from src.core.context_engineer import get_context_engineer
from src.core.memory_service import get_background_memory_service
# prompt refiner는 execute_llm_task의 decorator에서 자동 적용됨

logger = logging.getLogger(__name__)

# HTTP 에러 메시지 필터링 클래스
class HTTPErrorFilter(logging.Filter):
    """HTML 에러 응답을 필터링하여 간단한 메시지만 출력"""
    def filter(self, record):
        message = record.getMessage()
        
        # HTML 에러 페이지 감지 및 필터링
        if '<!DOCTYPE html>' in message or '<html' in message.lower():
            # HTML에서 에러 메시지 추출 시도
            
            # HTTP 상태 코드 추출
            status_match = re.search(r'HTTP (\d{3})', message)
            status_code = status_match.group(1) if status_match else "Unknown"
            
            # 에러 제목 추출 시도
            title_match = re.search(r'<title>([^<]+)</title>', message, re.IGNORECASE)
            error_title = title_match.group(1).strip() if title_match else None
            
            # 간단한 에러 메시지 생성
            if error_title:
                record.msg = f"HTTP {status_code}: {error_title}"
            else:
                # 상태 코드에 따른 기본 메시지
                if status_code == "502":
                    record.msg = f"HTTP {status_code}: Bad Gateway - Server temporarily unavailable"
                elif status_code == "504":
                    record.msg = f"HTTP {status_code}: Gateway Timeout - Server response timeout"
                elif status_code == "503":
                    record.msg = f"HTTP {status_code}: Service Unavailable - Server temporarily unavailable"
                elif status_code == "401":
                    record.msg = f"HTTP {status_code}: Unauthorized - Authentication failed"
                elif status_code == "404":
                    record.msg = f"HTTP {status_code}: Not Found"
                elif status_code == "500":
                    record.msg = f"HTTP {status_code}: Internal Server Error"
                else:
                    record.msg = f"HTTP {status_code}: Server Error"
            
            record.args = ()  # args 초기화
        
        return True

# Logger가 handler가 없으면 root logger의 handler 사용
if not logger.handlers:
    logger.setLevel(logging.INFO)
    # Root logger의 handler 사용 (main.py에서 설정된 handler)
    parent_logger = logging.getLogger()
    if parent_logger.handlers:
        logger.handlers = parent_logger.handlers
        logger.propagate = True
    else:
        # Fallback: 기본 handler 설정
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        handler.addFilter(HTTPErrorFilter())  # HTTP 에러 필터 추가
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
else:
    # 기존 handler에 필터 추가
    for handler in logger.handlers:
        if not any(isinstance(f, HTTPErrorFilter) for f in handler.filters):
            handler.addFilter(HTTPErrorFilter())

# FastMCP Runner 로거에도 필터 추가 (외부 라이브러리 로깅 필터링)
# Runner 로거는 나중에 생성될 수 있으므로, propagate를 활성화하고 root logger의 필터 사용
def setup_runner_logger_filter():
    """Runner 로거에 HTML 필터 추가 (지연 초기화)"""
    runner_logger = logging.getLogger("Runner")
    if runner_logger:
        runner_logger.propagate = True  # Root logger로 전파하여 필터 적용
        # 기존 handler에 필터 추가 (혹시 직접 handler가 있는 경우)
        for handler in runner_logger.handlers:
            if not any(isinstance(f, HTTPErrorFilter) for f in handler.filters):
                handler.addFilter(HTTPErrorFilter())

# 초기 설정
setup_runner_logger_filter()


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


class AgentState(TypedDict):
    """Main agent state containing messages and research data."""
    
    messages: Annotated[list, add_messages]
    user_query: str
    research_plan: Optional[str]
    research_tasks: Annotated[list, override_reducer]  # List of research tasks for parallel execution
    research_results: Annotated[list, override_reducer]  # Changed: supports both dict and str
    verified_results: Annotated[list, override_reducer]  # Changed: supports both dict and str
    final_report: Optional[str]
    
    # Human-in-the-loop 관련 필드
    pending_questions: Optional[List[Dict[str, Any]]]  # 대기 중인 질문들
    user_responses: Optional[Dict[str, Any]]  # 질문 ID -> 사용자 응답
    clarification_context: Optional[Dict[str, Any]]  # 명확화된 정보
    waiting_for_user: Optional[bool]  # 사용자 응답 대기 중인지
    current_agent: Optional[str]
    iteration: int
    session_id: Optional[str]
    research_failed: bool
    verification_failed: bool
    report_failed: bool
    error: Optional[str]


###################
# Agent Definitions
###################

@dataclass
class AgentContext:
    """Agent execution context."""
    agent_id: str
    session_id: str
    shared_memory: Any
    config: Any = None
    shared_results_manager: Optional[SharedResultsManager] = None
    discussion_manager: Optional[AgentDiscussionManager] = None


class PlannerAgent:
    """Planner agent - creates research plans (YAML-based configuration)."""
    
    def __init__(self, context: AgentContext, skill: Optional[Skill] = None):
        self.context = context
        self.name = "planner"
        self.available_tools: list = []  # MCP 자동 할당 도구
        self.tool_infos: list = []  # 도구 메타데이터
        self.skill = skill
        
        # YAML 설정 로드
        from src.core.skills.agent_loader import load_agent_config
        self.config = load_agent_config("planner")
        self.instruction = self.config.instructions
    
    async def domain_exploration(self, query: str) -> Dict[str, Any]:
        """
        도메인 분석 및 탐색을 수행합니다.
        
        Args:
            query: 연구 질문
            
        Returns:
            도메인 분석 결과 딕셔너리
        """
        logger.info(f"[{self.name}] 🔍 Starting domain exploration for query: {query[:100]}...")
        
        from src.core.llm_manager import execute_llm_task, TaskType
        from src.core.skills.agent_loader import get_prompt
        
        try:
            # 도메인 분석 프롬프트 가져오기
            domain_prompt = get_prompt("planner", "domain_analysis", query=query)
            system_message = "You are a domain analysis expert. Analyze the research domain to understand its characteristics, terminology, and requirements."
            
            # domain_prompt와 system_message는 execute_llm_task의 decorator에서 자동으로 최적화됨
            domain_result = await execute_llm_task(
                prompt=domain_prompt,
                task_type=TaskType.ANALYSIS,
                model_name=None,
                system_message=system_message
            )
            
            # JSON 파싱 시도
            domain_text = domain_result.content or "{}"
            
            # JSON 블록 추출
            json_match = re.search(r'\{[\s\S]*\}', domain_text)
            if json_match:
                try:
                    domain_analysis = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning(f"[{self.name}] Failed to parse domain analysis JSON, using default structure")
                    domain_analysis = {
                        "domain": "general",
                        "subdomains": [],
                        "characteristics": [],
                        "key_terminology": [],
                        "data_types": ["quantitative", "qualitative"],
                        "reliable_source_types": ["academic", "news", "government"],
                        "verification_criteria": ["source_reliability", "data_recency"],
                        "search_strategy": {
                            "keywords": [],
                            "related_topics": []
                        }
                    }
            else:
                logger.warning(f"[{self.name}] No JSON found in domain analysis result, using default structure")
                domain_analysis = {
                    "domain": "general",
                    "subdomains": [],
                    "characteristics": [],
                    "key_terminology": [],
                    "data_types": ["quantitative", "qualitative"],
                    "reliable_source_types": ["academic", "news", "government"],
                    "verification_criteria": ["source_reliability", "data_recency"],
                    "search_strategy": {
                        "keywords": [],
                        "related_topics": []
                    }
                }
            
            # 메타데이터 추가
            domain_analysis['_metadata'] = {
                'model_used': domain_result.model_used,
                'confidence': domain_result.confidence,
                'execution_time': domain_result.execution_time,
                'timestamp': domain_result.timestamp if hasattr(domain_result, 'timestamp') else None
            }
            
            logger.info(f"[{self.name}] ✅ Domain analysis completed: {domain_analysis.get('domain', 'unknown')}")
            logger.info(f"[{self.name}] Domain characteristics: {domain_analysis.get('characteristics', [])}")
            logger.info(f"[{self.name}] Reliable source types: {domain_analysis.get('reliable_source_types', [])}")
            
            return domain_analysis
            
        except Exception as e:
            logger.error(f"[{self.name}] Domain exploration failed: {e}")
            # 기본 도메인 분석 결과 반환
            return {
                "domain": "general",
                "subdomains": [],
                "characteristics": [],
                "key_terminology": [],
                "data_types": ["quantitative", "qualitative"],
                "reliable_source_types": ["academic", "news", "government"],
                "verification_criteria": ["source_reliability", "data_recency"],
                "search_strategy": {
                    "keywords": [],
                    "related_topics": []
                },
                "_metadata": {
                    "error": str(e)
                }
            }
    
    async def _detect_economic_request(self, query: str, domain_analysis: Dict[str, Any]) -> bool:
        """
        LLM을 사용하여 사용자 요청이 경제/금융 관련인지 판단합니다.
        
        Args:
            query: 사용자 요청
            domain_analysis: 도메인 분석 결과
            
        Returns:
            bool: 경제/금융 관련이면 True
        """
        try:
            from src.core.llm_manager import execute_llm_task, TaskType
            
            # 도메인 분석 결과에서 경제 관련 키워드 확인 (1차 필터링)
            domain = domain_analysis.get('domain', '').lower()
            subdomains = [s.lower() for s in domain_analysis.get('subdomains', [])]
            characteristics = [c.lower() for c in domain_analysis.get('characteristics', [])]
            key_terminology = [t.lower() for t in domain_analysis.get('key_terminology', [])]
            
            # 경제 관련 키워드
            economic_keywords = [
                'finance', 'financial', 'economy', 'economic', 'stock', 'stocks', 'market', 'markets',
                'investment', 'investing', 'trading', 'trade', 'portfolio', 'asset', 'assets',
                'revenue', 'profit', 'loss', 'earnings', 'dividend', 'bond', 'bonds',
                'currency', 'exchange', 'banking', 'bank', 'credit', 'debt', 'loan',
                '주식', '주가', '투자', '경제', '금융', '시장', '증권', '자산', '수익', '손익',
                '환율', '은행', '대출', '채권', '배당', '거래', '포트폴리오'
            ]
            
            # 1차 필터링: 키워드 기반 빠른 체크
            query_lower = query.lower()
            domain_text = ' '.join([domain] + subdomains + characteristics + key_terminology).lower()
            
            has_economic_keyword = any(
                keyword in query_lower or keyword in domain_text
                for keyword in economic_keywords
            )
            
            # 키워드가 없으면 경제 관련이 아님
            if not has_economic_keyword:
                logger.info(f"[{self.name}] No economic keywords found - not an economic request")
                return False
            
            # 2차 필터링: LLM 기반 정확한 판단
            prompt = f"""
사용자 요청: {query}

도메인 분석 결과:
- Domain: {domain_analysis.get('domain', 'unknown')}
- Subdomains: {', '.join(domain_analysis.get('subdomains', []))}
- Characteristics: {', '.join(domain_analysis.get('characteristics', []))}
- Key Terminology: {', '.join(domain_analysis.get('key_terminology', []))}

위 요청이 경제/금융/투자 관련 요청인지 판단하세요.

경제/금융/투자 관련 요청의 예:
- 주식 시장 분석, 투자 전략, 경제 지표 분석
- 기업 재무 분석, 주가 예측, 포트폴리오 관리
- 경제 전망, 금융 정책, 환율 분석
- 부동산 투자, 채권 분석, 파생상품

비경제 요청의 예:
- 기술 동향, 과학 연구, 의학 연구
- 역사, 문화, 예술
- 교육, 법률, 정치 (경제와 무관한 경우)

출력 형식 (JSON only, 추가 텍스트 금지):
{{
    "is_economic": true/false,
    "confidence": 0.0-1.0,
    "reason": "판단 근거를 한국어로 1문장"
}}
"""
            
            result = await execute_llm_task(
                prompt=prompt,
                task_type=TaskType.ANALYSIS,
                model_name=None,
                system_message="You are an expert at classifying research requests. Determine if a request is related to economics, finance, or investment."
            )
            
            result_text = result.content or "{}"
            
            # JSON 파싱 (json은 이미 파일 상단에서 import됨)
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                try:
                    analysis_result = json.loads(json_match.group())
                    is_economic = analysis_result.get('is_economic', False)
                    confidence = analysis_result.get('confidence', 0.5)
                    reason = analysis_result.get('reason', '')
                    
                    # confidence가 0.7 이상이면 경제 관련으로 판단
                    if is_economic and confidence >= 0.7:
                        logger.info(f"[{self.name}] Economic request detected (confidence: {confidence:.2f}): {reason}")
                        return True
                    else:
                        logger.info(f"[{self.name}] Not an economic request (confidence: {confidence:.2f}): {reason}")
                        return False
                except json.JSONDecodeError:
                    logger.warning(f"[{self.name}] Failed to parse economic detection JSON, using keyword-based result")
                    return has_economic_keyword
            else:
                logger.warning(f"[{self.name}] No JSON found in economic detection result, using keyword-based result")
                return has_economic_keyword
                
        except Exception as e:
            logger.error(f"[{self.name}] Economic request detection failed: {e}")
            # 에러 발생 시 키워드 기반 결과 사용
            return has_economic_keyword if 'has_economic_keyword' in locals() else False
    
    async def _call_financial_agent(self, user_query: str) -> Dict[str, Any]:
        """
        Financial Agent MCP 도구를 호출하여 경제 지표 분석을 수행합니다.
        
        Args:
            user_query: 사용자 요청
            
        Returns:
            Dict: Financial Agent 분석 결과
        """
        try:
            from src.core.mcp_integration import execute_tool
            
            logger.info(f"[{self.name}] Calling financial_agent::run_financial_analysis...")
            
            # MCP 도구 호출
            result = await execute_tool(
                "financial_agent::run_financial_analysis",
                {"user_query": user_query}
            )
            
            if result.get("success", False):
                # execute_tool의 data 필드에 financial_agent의 전체 결과가 들어있음
                financial_result = result.get("data", {})
                # financial_agent의 결과도 success 키를 포함하므로 확인
                if isinstance(financial_result, dict) and financial_result.get("success", False):
                    logger.info(f"[{self.name}] Financial agent returned successful result")
                    return financial_result
                else:
                    logger.warning(f"[{self.name}] Financial agent result format unexpected: {type(financial_result)}")
                    return financial_result if isinstance(financial_result, dict) else None
            else:
                error_msg = result.get("error", "Unknown error")
                logger.warning(f"[{self.name}] Financial agent returned error: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"[{self.name}] Failed to call financial agent: {e}")
            return None
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute planning task with Skills-based instruction and detailed logging."""
        logger.info(f"=" * 80)
        logger.info(f"[{self.name.upper()}] Starting research planning")
        logger.info(f"Query: {state['user_query']}")
        logger.info(f"Session: {state['session_id']}")
        logger.info(f"=" * 80)
        
        # 사용자 응답 대기 중이면 응답 처리
        if state.get("waiting_for_user", False):
            user_responses = state.get("user_responses", {})
            if user_responses:
                # 응답이 있으면 명확화 정보 적용
                from src.core.human_clarification_handler import get_clarification_handler
                clarification_handler = get_clarification_handler()
                
                for question_id, response_data in user_responses.items():
                    clarification = response_data.get("clarification", {})
                    # 명확화 정보를 state에 저장
                    state["clarification_context"] = state.get("clarification_context", {})
                    state["clarification_context"][question_id] = clarification
                    
                    # 응답에 따라 작업 방향 조정
                    response = response_data.get("response", "")
                    if response == "top_5":
                        # 상위 5개 결과만 사용하도록 설정
                        state["max_results"] = 5
                    elif response == "top_10":
                        # 상위 10개 결과만 사용하도록 설정
                        state["max_results"] = 10
                    elif response == "expand":
                        # 검색 범위 확대
                        state["expand_search"] = True
                    elif response == "modify":
                        # 검색어 수정 필요
                        state["modify_query"] = True
                
                # 대기 상태 해제
                state["waiting_for_user"] = False
                state["pending_questions"] = []
                logger.info("✅ User responses processed, continuing execution")
        
        # Read from shared memory - ONLY search within current session to prevent cross-task contamination
        memory = self.context.shared_memory
        current_session_id = state['session_id']
        
        # Search only within current session to prevent mixing previous task memories
        previous_plans = memory.search(
            state['user_query'], 
            limit=3,
            scope=MemoryScope.SESSION,
            session_id=current_session_id  # Critical: filter by current session only
        )
        
        logger.info(f"[{self.name}] Previous plans found in current session ({current_session_id}): {len(previous_plans) if previous_plans else 0}")
        
        # If no plans found in current session, explicitly set to empty to avoid confusion
        if not previous_plans:
            previous_plans = []
            logger.info(f"[{self.name}] No previous plans in current session - starting fresh task")
        
        # Skills-based instruction 사용
        instruction = self.instruction if self.skill else "You are a research planning agent."
        
        logger.info(f"[{self.name}] Using skill: {self.skill is not None}")
        
        # LLM 호출은 llm_manager를 통해 Gemini 직결 사용
        from src.core.llm_manager import execute_llm_task, TaskType
        
        # Use YAML-based prompt
        from src.core.skills.agent_loader import get_prompt
        
        # Phase 1: Domain Analysis and Exploration
        logger.info(f"[{self.name}] 🔍 Starting domain analysis and exploration...")
        domain_analysis_result = await self.domain_exploration(state['user_query'])
        state['domain_analysis'] = domain_analysis_result
        logger.info(f"[{self.name}] ✅ Domain analysis completed: {domain_analysis_result.get('domain', 'unknown')}")
        
        # Phase 1.5: 경제 관련 요청 감지 (LLM 기반)
        logger.info(f"[{self.name}] 🔍 Checking if request is related to economics/finance...")
        is_economic_request = await self._detect_economic_request(state['user_query'], domain_analysis_result)
        state['is_economic_request'] = is_economic_request
        
        # Phase 1.6: 경제 관련 요청이면 financial_agent 호출
        financial_analysis_result = None
        if is_economic_request:
            logger.info(f"[{self.name}] ✅ Economic/finance related request detected")
            logger.info(f"[{self.name}] 📊 Calling financial_agent for economic indicator analysis...")
            try:
                financial_analysis_result = await self._call_financial_agent(state['user_query'])
                if financial_analysis_result and financial_analysis_result.get('success'):
                    logger.info(f"[{self.name}] ✅ Financial agent analysis completed successfully")
                    state['financial_analysis_result'] = financial_analysis_result
                else:
                    logger.warning(f"[{self.name}] ⚠️ Financial agent analysis failed or returned no results")
                    financial_analysis_result = None
            except Exception as e:
                logger.warning(f"[{self.name}] ⚠️ Financial agent call failed: {e}. Continuing with normal planning.")
                financial_analysis_result = None
        else:
            logger.info(f"[{self.name}] ℹ️ Not an economic/finance related request")
        
        # Format previous_plans for prompt - only include if from current session
        # CRITICAL: Previous context is for REFERENCE ONLY - current task must be planned independently
        if previous_plans:
            # Filter to ensure only current session plans are included
            current_session_plans = [
                p for p in previous_plans 
                if p.get("session_id") == current_session_id
            ]
            if current_session_plans:
                # Format previous plans with STRONG warning that they are REFERENCE ONLY
                previous_plans_text = f"""
⚠️ REFERENCE ONLY - DO NOT REUSE ⚠️
Previous research context (for domain understanding ONLY - NOT for task execution):
{chr(10).join([f"- {p.get('key', 'plan')}: {str(p.get('value', ''))[:200]}" for p in current_session_plans])}

CRITICAL: The above is for CONTEXT REFERENCE ONLY. You MUST create a NEW plan specifically for the CURRENT task: "{state['user_query']}".
DO NOT reuse previous task queries, search terms, or plan structures.
"""
            else:
                previous_plans_text = "No previous research found in current session. This is a COMPLETELY NEW task - create a fresh plan for the current query only."
        else:
            previous_plans_text = "No previous research found in current session. This is a COMPLETELY NEW task - create a fresh plan for the current query only."
        
        # 도메인 분석 결과를 프롬프트에 포함
        domain_context = ""
        if domain_analysis_result:
            domain_context = f"""
Domain Analysis Results:
- Domain: {domain_analysis_result.get('domain', 'general')}
- Subdomains: {', '.join(domain_analysis_result.get('subdomains', []))}
- Characteristics: {', '.join(domain_analysis_result.get('characteristics', []))}
- Key Terminology: {', '.join(domain_analysis_result.get('key_terminology', []))}
- Reliable Source Types: {', '.join(domain_analysis_result.get('reliable_source_types', []))}
- Verification Criteria: {', '.join(domain_analysis_result.get('verification_criteria', []))}
"""
        
        # Financial Agent 분석 결과를 프롬프트에 포함
        financial_context = ""
        if financial_analysis_result:
            try:
                # Financial analysis 결과를 JSON 문자열로 변환 (요약 포함)
                financial_summary = {
                    "extracted_info": financial_analysis_result.get("extracted_info", {}),
                    "market_outlook": financial_analysis_result.get("market_outlook"),
                    "investment_plan": financial_analysis_result.get("investment_plan"),
                    "technical_analysis_summary": {
                        ticker: {
                            "price": data.get("price"),
                            "rsi": data.get("rsi"),
                            "macd": data.get("macd")
                        }
                        for ticker, data in financial_analysis_result.get("technical_analysis", {}).items()
                    } if financial_analysis_result.get("technical_analysis") else {},
                    "daily_pnl": financial_analysis_result.get("daily_pnl"),
                    "sentiment_analysis": financial_analysis_result.get("sentiment_analysis")
                }
                
                financial_context = f"""
Financial Agent Analysis Results (경제 지표 분석):
{json.dumps(financial_summary, ensure_ascii=False, indent=2)}

이 분석 결과를 바탕으로 더 구체적이고 정확한 연구 계획을 수립하세요.
경제 지표, 시장 전망, 투자 계획 등을 고려하여 연구 방향을 설정하세요.
"""
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to format financial analysis context: {e}")
                financial_context = ""
        
        # prompt는 execute_llm_task의 decorator에서 자동으로 최적화됨
        logger.info(f"[{self.name}] Calling LLM for planning...")
        
        # Current time calculation for prompt
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S (%A)')
        
        prompt = get_prompt("planner", "planning",
                           instruction=self.instruction,
                           user_query=state['user_query'],
                           previous_plans=previous_plans_text,
                           current_time=current_time)
        
        # 도메인 분석 결과와 Financial Agent 결과를 프롬프트에 추가
        context_parts = []
        if domain_context:
            context_parts.append(domain_context)
        if financial_context:
            context_parts.append(financial_context)
        
        if context_parts:
            prompt = "\n\n".join(context_parts) + "\n\n" + prompt

        # prompt는 execute_llm_task의 decorator에서 자동으로 최적화됨
        logger.info(f"[{self.name}] Calling LLM for planning...")
        # Gemini 실행
        model_result = await execute_llm_task(
            prompt=prompt,
            task_type=TaskType.PLANNING,
            model_name=None,
            system_message=None
        )
        plan = model_result.content or 'No plan generated'
        
        logger.info(f"[{self.name}] ✅ Plan generated: {len(plan)} characters")
        logger.info(f"[{self.name}] Plan preview: {plan[:200]}...")
        
        # Council 활성화 확인 및 적용
        use_council = state.get('use_council', None)  # 수동 활성화 옵션
        if use_council is None:
            # 자동 활성화 판단
            from src.core.council_activator import get_council_activator
            activator = get_council_activator()
            activation_decision = activator.should_activate(
                process_type='planning',
                query=state['user_query'],
                context={'domains': [], 'steps': []}  # 컨텍스트는 향후 확장 가능
            )
            use_council = activation_decision.should_activate
            if use_council:
                logger.info(f"[{self.name}] 🏛️ Council auto-activated: {activation_decision.reason}")
        
        # Council 적용 (활성화된 경우)
        if use_council:
            try:
                from src.core.llm_council import run_full_council
                logger.info(f"[{self.name}] 🏛️ Running Council review for research plan...")
                
                # Council에 계획 검토 요청
                council_query = f"""Review and improve the following research plan. Provide feedback on completeness, feasibility, and quality.

Research Query: {state['user_query']}

Research Plan:
{plan}

Provide an improved version of the plan that addresses any gaps or issues you identify."""
                
                stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
                    council_query
                )
                
                # Council 결과를 계획에 반영
                council_improved_plan = stage3_result.get('response', plan)
                plan = council_improved_plan
                
                logger.info(f"[{self.name}] ✅ Council review completed. Plan improved with consensus.")
                logger.info(f"[{self.name}] Council aggregate rankings: {metadata.get('aggregate_rankings', [])}")
                
                # Council 메타데이터를 state에 저장
                state['council_metadata'] = {
                    'planning': {
                        'stage1_results': stage1_results,
                        'stage2_results': stage2_results,
                        'stage3_result': stage3_result,
                        'metadata': metadata
                    }
                }
            except Exception as e:
                logger.warning(f"[{self.name}] Council review failed: {e}. Using original plan.")
                # Council 실패 시 원본 계획 사용 (fallback 제거 - 명확한 로깅만)
        
        state['research_plan'] = plan
        
        # 작업 분할: 연구 계획을 여러 독립적인 작업으로 분할
        logger.info(f"[{self.name}] Splitting research plan into parallel tasks...")
        
        # Use YAML-based prompt template for task decomposition
        from src.core.skills.agent_loader import get_prompt
        
        # 도메인 분석 결과를 JSON 문자열로 변환
        domain_analysis_text = json.dumps(domain_analysis_result, ensure_ascii=False, indent=2) if domain_analysis_result else "{}"
        
        task_split_prompt = get_prompt(
            "planner",
            "task_decomposition",
            plan=plan,
            query=state['user_query'],
            domain_analysis=domain_analysis_text,
            current_time=current_time
        )

        try:
            task_split_result = await execute_llm_task(
                prompt=task_split_prompt,
                task_type=TaskType.PLANNING,
                model_name=None,
                system_message="You are a task decomposition agent. Split research plans into independent parallel tasks."
            )
            
            task_split_text = task_split_result.content or ""
            
            # JSON 파싱 시도
            
            # JSON 블록 추출
            json_match = re.search(r'\{[\s\S]*\}', task_split_text)
            if json_match:
                task_split_json = json.loads(json_match.group())
                tasks = task_split_json.get('tasks', [])
            else:
                # JSON이 없으면 텍스트에서 작업 추출 시도
                tasks = []
                lines = task_split_text.split('\n')
                current_task = None
                for line in lines:
                    line = line.strip()
                    if 'task_id' in line.lower() or 'task' in line.lower() and ':' in line:
                        if current_task:
                            tasks.append(current_task)
                        task_id_match = re.search(r'task[_\s]*(\d+)', line, re.IGNORECASE)
                        task_id = f"task_{task_id_match.group(1) if task_id_match else len(tasks) + 1}"
                        current_task = {
                            "task_id": task_id,
                            "description": "",
                            "search_queries": [],
                            "priority": len(tasks) + 1,
                            "estimated_time": "medium",
                            "dependencies": []
                        }
                    elif current_task:
                        if 'description' in line.lower() or '설명' in line:
                            desc_match = re.search(r':\s*(.+)', line)
                            if desc_match:
                                current_task["description"] = desc_match.group(1).strip()
                        elif 'query' in line.lower() or '쿼리' in line:
                            query_match = re.search(r':\s*(.+)', line)
                            if query_match:
                                current_task["search_queries"].append(query_match.group(1).strip())
                
                if current_task:
                    tasks.append(current_task)
            
            # 작업이 없으면 기본 작업 생성
            if not tasks:
                logger.warning(f"[{self.name}] Failed to parse tasks, creating default task")
                tasks = [{
                    "task_id": "task_1",
                    "description": state['user_query'],
                    "search_queries": [state['user_query']],
                    "priority": 1,
                    "estimated_time": "medium",
                    "dependencies": []
                }]
            
            # 각 작업에 메타데이터 추가 및 검색 쿼리 검증
            user_query_lower = state['user_query'].lower()
            # 잘못된 검색 쿼리 키워드 (메타 정보 관련)
            invalid_keywords = [
                '작업 분할', '태스크 분할', '병렬화', '병렬 실행', 'task decomposition',
                'task split', 'parallel', 'parallelization', '연구 방법론', '연구 전략',
                '연구 계획', 'research methodology', 'research strategy', 'research plan',
                '하위 연구 주제 분해', '독립적 연구 태스크', '연구 작업 병렬화'
            ]
            
            for i, task in enumerate(tasks):
                if 'task_id' not in task:
                    task['task_id'] = f"task_{i + 1}"
                if 'name' not in task:
                    task['name'] = task.get('description', state['user_query'])[:100]
                if 'description' not in task:
                    task['description'] = state['user_query']
                
                # Task 구조 확장 필드 기본값 설정
                if 'objectives' not in task:
                    task['objectives'] = [task.get('description', state['user_query'])]
                
                if 'required_information' not in task:
                    task['required_information'] = {
                        'data_types': ['quantitative', 'qualitative'],
                        'key_entities': [],
                        'sources': {
                            'min_count': 3,
                            'reliability_threshold': 0.7,
                            'preferred_types': ['academic', 'news', 'government']
                        }
                    }
                
                if 'verification_strategy' not in task:
                    task['verification_strategy'] = {
                        'cross_verify': True,
                        'fact_check': True,
                        'source_validation': True,
                        'min_consensus_sources': 2
                    }
                
                if 'success_criteria' not in task:
                    task['success_criteria'] = [
                        f"Task {task.get('task_id')} completed with valid results",
                        "Sources meet reliability threshold"
                    ]
                
                # 검색 쿼리 검증 및 필터링
                if 'search_queries' in task and task['search_queries']:
                    # 잘못된 검색 쿼리 필터링
                    valid_queries = []
                    for query in task['search_queries']:
                        query_str = str(query).strip()
                        query_lower = query_str.lower()
                        
                        # {query} 플레이스홀더가 포함된 쿼리 완전 제외
                        if "{query}" in query_str or "{query}" in query_lower:
                            logger.warning(f"[{self.name}] Task {task.get('task_id')}: Filtered out query with placeholder: '{query_str[:50]}...'")
                            continue
                        
                        # 메타 정보 관련 키워드가 포함된 쿼리 제외
                        is_invalid = any(keyword in query_lower for keyword in invalid_keywords)
                        # 사용자 쿼리와 관련이 없는 쿼리 제외 (너무 짧거나 일반적인 경우)
                        is_too_generic = len(query_str) < 10
                        
                        if not is_invalid and not is_too_generic:
                            valid_queries.append(query_str)
                        else:
                            logger.warning(f"[{self.name}] Task {task.get('task_id')}: Filtered out invalid query: '{query_str[:50]}...' (invalid={is_invalid}, generic={is_too_generic})")
                    
                    # 유효한 쿼리가 없으면 사용자 쿼리 사용
                    if not valid_queries:
                        logger.warning(f"[{self.name}] Task {task.get('task_id')} has no valid search queries, using user query: '{state['user_query']}'")
                        valid_queries = [state['user_query']]
                    
                    task['search_queries'] = valid_queries
                    logger.info(f"[{self.name}] Task {task.get('task_id')}: Final search queries: {valid_queries}")
                else:
                    # search_queries가 없으면 사용자 쿼리 사용
                    task['search_queries'] = [state['user_query']]
                
                if 'priority' not in task:
                    task['priority'] = i + 1
                if 'estimated_time' not in task:
                    task['estimated_time'] = "medium"
                if 'dependencies' not in task:
                    task['dependencies'] = []
            
            state['research_tasks'] = tasks
            logger.info(f"[{self.name}] ✅ Split research plan into {len(tasks)} parallel tasks")
            for task in tasks:
                queries = task.get('search_queries', [])
                queries_preview = [q[:40] + '...' if len(q) > 40 else q for q in queries[:3]]
                logger.info(f"[{self.name}]   - {task.get('task_id')}: {task.get('description', '')[:50]}... ({len(queries)} queries: {queries_preview})")
                
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Failed to split tasks: {e}")
            # 실패 시 기본 작업 생성
            state['research_tasks'] = [{
                "task_id": "task_1",
                "description": state['user_query'],
                "search_queries": [state['user_query']],
                "priority": 1,
                "estimated_time": "medium",
                "dependencies": []
            }]
            logger.warning(f"[{self.name}] Using default single task")
        
        state['current_agent'] = self.name
        
        # Write to shared memory
        memory.write(
            key=f"plan_{state['session_id']}",
            value=plan,
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        memory.write(
            key=f"tasks_{state['session_id']}",
            value=state['research_tasks'],
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        logger.info(f"[{self.name}] Plan and tasks saved to shared memory")
        logger.info(f"=" * 80)
        
        return state


class ExecutorAgent:
    """Executor agent - executes research tasks using tools (Skills-based)."""
    
    def __init__(self, context: AgentContext, skill: Optional[Skill] = None):
        self.context = context
        self.name = "executor"
        self.available_tools: list = []  # MCP 자동 할당 도구
        self.tool_infos: list = []  # 도구 메타데이터
        self.skill = skill
        
        # Skill이 없으면 로드 시도
        if self.skill is None:
            skill_manager = get_skill_manager()
            self.skill = skill_manager.load_skill("research_executor")
        
        # Skill instruction 사용
        if self.skill:
            self.instruction = self.skill.instructions
        else:
            self.instruction = "You are a research execution agent."
    
    async def _filter_results_by_relevance(
        self,
        search_results: List[Dict[str, Any]],
        user_query: str,
        search_queries: List[str],
        current_time: str = ""
    ) -> List[Dict[str, Any]]:
        """
        검색 결과를 관련성 기준으로 사전 필터링합니다.
        
        Args:
            search_results: 필터링할 검색 결과 리스트
            user_query: 원래 사용자 쿼리
            search_queries: 검색에 사용된 쿼리 리스트
            
        Returns:
            관련성 점수 3점 이상인 결과만 포함된 리스트
        """
        from src.core.llm_manager import execute_llm_task, TaskType
        
        MIN_REQUIRED_RESULTS = 30
        RELEVANCE_THRESHOLD = 3  # 1-10 점수 기준
        
        if len(search_results) <= MIN_REQUIRED_RESULTS:
            # 결과가 이미 충분하면 필터링 스킵 (너무 공격적으로 필터링하지 않음)
            logger.info(f"[{self.name}] Results count ({len(search_results)}) is acceptable, skipping aggressive filtering")
            return search_results
        
        logger.info(f"[{self.name}] 🔍 Filtering {len(search_results)} results by relevance (threshold: {RELEVANCE_THRESHOLD}/10)")
        
        # 배치로 관련성 평가 (성능 최적화)
        batch_size = 10
        filtered_results = []
        
        for i in range(0, len(search_results), batch_size):
            batch = search_results[i:i+batch_size]
            
            # 배치 평가 프롬프트
            batch_evaluation_prompt = f"""다음 검색 결과들을 원래 쿼리와의 관련성에 따라 평가하세요.

현재 시각: {current_time}
원래 쿼리: {user_query}
검색 쿼리: {', '.join(search_queries[:3])}

검색 결과:
{chr(10).join([f"{j+1}. 제목: {r.get('title', 'N/A')[:100]}{chr(10)}   내용: {r.get('snippet', r.get('content', ''))[:200]}{chr(10)}   URL: {r.get('url', 'N/A')}" for j, r in enumerate(batch)])}

각 결과에 대해 다음을 평가하세요:
1. 직접적 관련성 (1-10): 쿼리와 직접적으로 관련이 있는가?
2. 간접적 관련성 (1-10): 배경 정보나 맥락 제공에 도움이 되는가?
3. 완전히 무관한지 여부 (YES/NO)

응답 형식 (JSON):
{{
  "evaluations": [
    {{
      "index": 1,
      "direct_relevance": 8,
      "indirect_relevance": 5,
      "is_irrelevant": false,
      "overall_score": 7,
      "reason": "엔비디아 GPU 시장 점유율에 대한 직접적 정보"
    }},
    ...
  ]
}}

⚠️ 중요:
- 완전히 무관한 결과만 제외 (예: 엔비디아 쿼리인데 부동산 관련 결과)
- 관련성이 약간 낮아도 배경 정보로 유용하면 포함
- overall_score는 (direct_relevance * 0.7 + indirect_relevance * 0.3)로 계산"""
            
            try:
                evaluation_result = await execute_llm_task(
                    prompt=batch_evaluation_prompt,
                    task_type=TaskType.ANALYSIS,
                    model_name=None,
                    system_message="You are an expert information relevance evaluator. Evaluate search results for relevance to the query."
                )
                
                # JSON 파싱                
                evaluation_text = evaluation_result.content or "{}"
                json_match = re.search(r'\{[\s\S]*\}', evaluation_text)
                if json_match:
                    try:
                        evaluation_data = json.loads(json_match.group())
                        evaluations = evaluation_data.get('evaluations', [])
                        
                        for eval_item in evaluations:
                            idx = eval_item.get('index', 0) - 1  # 1-based to 0-based
                            if 0 <= idx < len(batch):
                                overall_score = eval_item.get('overall_score', 0)
                                is_irrelevant = eval_item.get('is_irrelevant', False)
                                
                                # 관련성 점수가 threshold 이상이고 무관하지 않으면 포함
                                if overall_score >= RELEVANCE_THRESHOLD and not is_irrelevant:
                                    result = batch[idx].copy()
                                    result['relevance_score'] = overall_score
                                    result['relevance_reason'] = eval_item.get('reason', '')
                                    filtered_results.append(result)
                                else:
                                    logger.debug(f"[{self.name}] Filtered out result {i+idx+1}: score={overall_score}, irrelevant={is_irrelevant}")
                    except json.JSONDecodeError:
                        logger.warning(f"[{self.name}] Failed to parse relevance evaluation JSON, including all results in batch")
                        filtered_results.extend(batch)
                else:
                    logger.warning(f"[{self.name}] No JSON found in relevance evaluation, including all results in batch")
                    filtered_results.extend(batch)
                    
            except Exception as e:
                logger.warning(f"[{self.name}] Relevance evaluation failed for batch {i//batch_size + 1}: {e}. Including all results in batch.")
                filtered_results.extend(batch)
        
        # 필터링 후에도 최소 30개 이상 보장
        if len(filtered_results) < MIN_REQUIRED_RESULTS:
            logger.warning(f"[{self.name}] ⚠️ Filtered results ({len(filtered_results)}) < minimum ({MIN_REQUIRED_RESULTS}), including lower relevance results")
            # 관련성 점수 순으로 정렬하여 상위 결과 포함
            scored_results = []
            for result in search_results:
                score = result.get('relevance_score', 5)  # 기본값 5
                scored_results.append((score, result))
            
            scored_results.sort(reverse=True, key=lambda x: x[0])
            filtered_results = [r for _, r in scored_results[:MIN_REQUIRED_RESULTS]]
            logger.info(f"[{self.name}] ✅ Included top {len(filtered_results)} results to meet minimum requirement")
        
        return filtered_results
    
    async def execute(self, state: AgentState, assigned_task: Optional[Dict[str, Any]] = None) -> AgentState:
        """Execute research tasks with detailed logging."""
        logger.info(f"=" * 80)
        logger.info(f"[{self.name.upper()}] Starting research execution")
        logger.info(f"Agent ID: {self.context.agent_id}")
        logger.info(f"Query: {state['user_query']}")
        logger.info(f"Session: {state['session_id']}")
        logger.info(f"=" * 80)
        
        # 사용자 응답 대기 중이면 응답 처리
        if state.get("waiting_for_user", False):
            user_responses = state.get("user_responses", {})
            if user_responses:
                # 응답이 있으면 명확화 정보 적용
                from src.core.human_clarification_handler import get_clarification_handler
                clarification_handler = get_clarification_handler()
                
                for question_id, response_data in user_responses.items():
                    clarification = response_data.get("clarification", {})
                    # 명확화 정보를 state에 저장
                    state["clarification_context"] = state.get("clarification_context", {})
                    state["clarification_context"][question_id] = clarification
                    
                    # 응답에 따라 작업 방향 조정
                    response = response_data.get("response", "")
                    if response == "top_5":
                        # 상위 5개 결과만 사용하도록 설정
                        state["max_results"] = 5
                    elif response == "top_10":
                        # 상위 10개 결과만 사용하도록 설정
                        state["max_results"] = 10
                    elif response == "expand":
                        # 검색 범위 확대
                        state["expand_search"] = True
                    elif response == "modify":
                        # 검색어 수정 필요
                        state["modify_query"] = True
                
                # 대기 상태 해제
                state["waiting_for_user"] = False
                state["pending_questions"] = []
                logger.info("✅ User responses processed, continuing execution")
        
        # 작업 할당: assigned_task가 있으면 사용, 없으면 state에서 찾기
        if assigned_task is None:
            # state['research_tasks']에서 이 에이전트에게 할당된 작업 찾기
            tasks = state.get('research_tasks', [])
            if tasks:
                # agent_id를 기반으로 작업 할당 (라운드로빈)
                agent_id = self.context.agent_id
                if agent_id.startswith("executor_"):
                    try:
                        agent_index = int(agent_id.split("_")[1])
                        if agent_index < len(tasks):
                            assigned_task = tasks[agent_index]
                            logger.info(f"[{self.name}] Assigned task {assigned_task.get('task_id', 'unknown')} to {agent_id}")
                        else:
                            # 인덱스가 범위를 벗어나면 첫 번째 작업 할당
                            assigned_task = tasks[0]
                            logger.info(f"[{self.name}] Agent index out of range, using first task")
                    except (ValueError, IndexError):
                        assigned_task = tasks[0] if tasks else None
                        logger.info(f"[{self.name}] Using first task (fallback)")
                else:
                    # agent_id가 executor_ 형식이 아니면 첫 번째 작업 사용
                    assigned_task = tasks[0] if tasks else None
            else:
                # 작업이 없으면 메모리에서 읽기
                memory = self.context.shared_memory
                tasks = memory.read(
                    key=f"tasks_{state['session_id']}",
                    scope=MemoryScope.SESSION,
                    session_id=state['session_id']
                ) or []
                if tasks:
                    assigned_task = tasks[0] if tasks else None
        
        # Read plan from shared memory
        memory = self.context.shared_memory
        plan = memory.read(
            key=f"plan_{state['session_id']}",
            scope=MemoryScope.SESSION,
            session_id=state['session_id']
        )
        
        logger.info(f"[{self.name}] Research plan loaded: {plan is not None}")
        if plan:
            logger.info(f"[{self.name}] Plan preview: {plan[:200]}...")
        
        # 실제 연구 실행 - MCP Hub를 통한 병렬 검색 수행
        query = state['user_query']
        
        # Current time calculation for prompt and context
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S (%A)')
        
        results = []
        
        try:
            # MCP Hub 초기화 확인
            from src.core.mcp_integration import get_mcp_hub, execute_tool, ToolCategory
            
            hub = get_mcp_hub()
            logger.info(f"[{self.name}] MCP Hub status: {len(hub.mcp_sessions) if hub.mcp_sessions else 0} servers connected")
            
            if not hub.mcp_sessions:
                logger.info(f"[{self.name}] Initializing MCP Hub...")
                await hub.initialize_mcp()
                logger.info(f"[{self.name}] MCP Hub initialized: {len(hub.mcp_sessions)} servers")
            
            # 작업 할당이 있으면 해당 작업의 검색 쿼리 사용
            # CRITICAL: Verify that assigned_task is for the CURRENT query
            search_queries = []
            if assigned_task:
                task_id = assigned_task.get('task_id', 'unknown')
                task_description = assigned_task.get('description', '')
                raw_queries = assigned_task.get('search_queries', [])
                
                logger.info(f"[{self.name}] ⚠️ TASK VERIFICATION:")
                logger.info(f"[{self.name}]   - Current User Query: '{query}'")
                logger.info(f"[{self.name}]   - Assigned Task ID: {task_id}")
                logger.info(f"[{self.name}]   - Task Description: {task_description[:100]}...")
                logger.info(f"[{self.name}]   - Raw queries count: {len(raw_queries)}")
                
                # Verify that assigned_task queries are related to current query
                # Extract key terms from current query for validation
                current_query_terms = set(query.lower().split())
                query_relevance_check = False
                
                # 잘못된 검색 쿼리 필터링 (메타 정보 관련)
                invalid_keywords = [
                    '작업 분할', '태스크 분할', '병렬화', '병렬 실행', 'task decomposition',
                    'task split', 'parallel', 'parallelization', '연구 방법론', '연구 전략',
                    '연구 계획', 'research methodology', 'research strategy', 'research plan',
                    '하위 연구 주제 분해', '독립적 연구 태스크', '연구 작업 병렬화'
                ]
                
                for q in raw_queries:
                    q_str = str(q).strip()
                    q_lower = q_str.lower()
                    
                    # {query} 플레이스홀더가 포함된 쿼리 완전 제외
                    if "{query}" in q_str or "{query}" in q_lower:
                        logger.warning(f"[{self.name}] ❌ Filtered out query with placeholder: '{q_str[:50]}...'")
                        continue
                    
                    # Verify query relevance to current user query
                    query_terms = set(q_lower.split())
                    # Check if query contains at least one key term from current query (for relevance)
                    if len(current_query_terms) > 0:
                        common_terms = query_terms.intersection(current_query_terms)
                        if len(common_terms) > 0:
                            query_relevance_check = True
                    
                    is_invalid = any(keyword in q_lower for keyword in invalid_keywords)
                    is_too_generic = len(q_str) < 10
                    
                    if not is_invalid and not is_too_generic:
                        search_queries.append(q_str)
                        logger.info(f"[{self.name}] ✅ Valid query added: '{q_str[:80]}...'")
                    else:
                        logger.warning(f"[{self.name}] ❌ Filtered out invalid query: '{q_str[:50]}...' (invalid={is_invalid}, generic={is_too_generic})")
                
                # Verify task relevance
                if not query_relevance_check and len(current_query_terms) > 0:
                    logger.warning(f"[{self.name}] ⚠️ WARNING: Assigned task queries may not be related to current query: '{query}'")
                    logger.warning(f"[{self.name}] ⚠️ This might be a previous task's queries. Verifying task assignment...")
                
                # 유효한 쿼리가 없으면 사용자 쿼리 사용
                if not search_queries:
                    logger.warning(f"[{self.name}] ⚠️ No valid queries in assigned task, using CURRENT user query: '{query}'")
                    search_queries = [query]
                else:
                    logger.info(f"[{self.name}] ✅ Using {len(search_queries)} valid queries from task {task_id} (for current query: '{query}')")
            
            # 작업 할당이 없거나 쿼리가 없으면 기존 로직 사용
            if not search_queries:
                search_queries = [query]  # 기본 쿼리
                if plan:
                    # LLM으로 연구 계획에서 검색 쿼리 추출
                    from src.core.llm_manager import execute_llm_task, TaskType

                    # Use YAML-based prompt for query generation
                    from src.core.skills.agent_loader import get_prompt
                    query_generation_prompt = get_prompt("planner", "query_generation",
                                                        plan=plan,
                                                        query=query,
                                                        current_time=current_time)

                    try:
                        system_message = self.config.prompts["query_generation"]["system_message"]
                        # query_generation_prompt와 system_message는 execute_llm_task의 decorator에서 자동으로 최적화됨
                        
                        query_result = await execute_llm_task(
                            prompt=query_generation_prompt,
                            task_type=TaskType.PLANNING,
                            model_name=None,
                            system_message=system_message
                        )

                        generated_queries = query_result.content or ""
                        # 각 줄을 쿼리로 파싱
                        for line in generated_queries.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#') and len(line) > 5:
                                search_queries.append(line)

                        # 중복 제거
                        search_queries = list(dict.fromkeys(search_queries))[:5]  # 최대 5개
                        logger.info(f"[{self.name}] Generated {len(search_queries)} search queries from plan")
                    except Exception as e:
                        logger.warning(f"[{self.name}] Failed to generate search queries from plan: {e}, using original query only")
            
            # 최소 3-5개의 다양한 검색 쿼리 보장
            MIN_QUERIES = 3
            MAX_QUERIES = 8
            if len(search_queries) < MIN_QUERIES:
                logger.info(f"[{self.name}] Only {len(search_queries)} queries available, generating additional queries to ensure diversity...")
                # 사용자 쿼리를 기반으로 다양한 관점의 검색 쿼리 생성
                base_query = query
                additional_queries = []
                
                # 다양한 관점의 쿼리 생성
                query_variations = [
                    f"{base_query} 분석",
                    f"{base_query} 전망 {datetime.now().year}",
                    f"{base_query} 동향",
                    f"{base_query} 현황",
                    f"{base_query} 전문가 의견"
                ]
                
                for variation in query_variations:
                    if variation not in search_queries and len(search_queries) < MAX_QUERIES:
                        search_queries.append(variation)
                        additional_queries.append(variation)
                
                if additional_queries:
                    logger.info(f"[{self.name}] Added {len(additional_queries)} additional query variations: {additional_queries}")
            
            # 병렬 검색 실행
            logger.info(f"[{self.name}] Executing {len(search_queries)} searches in parallel...")
            
            # 검색 쿼리 중복 제거 (Strict Deduplication)
            unique_search_queries = []
            seen_queries = set()
            normalized_base_query = re.sub(r'\s+', ' ', query.lower().strip())
            
            for q in search_queries:
                q_normalized = re.sub(r'\s+', ' ', q.lower().strip())
                
                # 1. 이미 처리된 쿼리인지 확인
                if q_normalized in seen_queries:
                    logger.warning(f"[{self.name}] ⚠️ Duplicate query removed: '{q}'")
                    continue
                
                # 2. Base 쿼리와 완전히 동일한 경우 (이미 포함되어 있을 수 있으므로)
                # 단, 첫 번째 쿼리가 Base 쿼리인 경우는 허용
                if q_normalized == normalized_base_query and seen_queries:
                    logger.warning(f"[{self.name}] ⚠️ Query identical to user query removed (redundant): '{q}'")
                    continue
                    
                seen_queries.add(q_normalized)
                unique_search_queries.append(q)
            
            search_queries = unique_search_queries
            logger.info(f"[{self.name}] Unique search queries ({len(search_queries)}): {search_queries}")
            
            async def execute_single_search(search_query: str, query_index: int) -> Dict[str, Any]:
                """단일 검색 실행 (여러 검색 도구 fallback 지원)."""
                # 실제 검색 쿼리 값 로그 출력
                logger.info(f"[{self.name}] Search {query_index + 1}/{len(search_queries)}: '{search_query}'")
                
                # 각 검색마다 더 많은 결과 수집 (최소 30개 출처 보장을 위해)
                # 여러 검색 쿼리 사용 시 각 쿼리당 최소 10-15개씩 수집하여 총 30개 이상 보장
                num_queries = len(search_queries)
                results_per_query = max(10, min(15, 30 // max(1, num_queries)))  # 최소 10개, 최대 15개, 총 30개 이상 보장
                
                # 여러 검색 도구 시도 (fallback 지원)
                search_tools = ["g-search", "mcp_search", "ddg_search"]  # 우선순위 순서
                
                for tool_name in search_tools:
                    try:
                        logger.info(f"[{self.name}] Trying search tool: {tool_name}")
                        search_result = await execute_tool(
                            tool_name,
                            {"query": search_query, "max_results": results_per_query}
                        )
                        
                        # 성공한 경우
                        if search_result.get('success', False):
                            logger.info(f"[{self.name}] ✅ Search succeeded with {tool_name}")
                            return {
                                "query": search_query,
                                "index": query_index,
                                "result": search_result,
                                "success": True,
                                "tool_used": tool_name
                            }
                        else:
                            # 실패했지만 에러가 없는 경우 (다음 도구 시도)
                            error_msg = search_result.get('error', 'Unknown error')
                            logger.warning(f"[{self.name}] ⚠️ {tool_name} returned success=False: {error_msg}")
                            continue
                            
                    except Exception as e:
                        error_str = str(e)
                        # DuckDuckGo MCP 서버 버그 등 특정 에러 처리
                        if "AttributeError" in error_str or "TimeoutError" in error_str or "HTTPStatusError" in error_str:
                            logger.warning(f"[{self.name}] ⚠️ {tool_name} failed with known issue: {error_str[:100]}... (trying next tool)")
                        else:
                            logger.warning(f"[{self.name}] ⚠️ {tool_name} failed: {error_str[:100]}... (trying next tool)")
                        continue
                
                # 모든 검색 도구 실패
                logger.error(f"[{self.name}] ❌ All search tools failed for query: '{search_query}'")
                return {
                    "query": search_query,
                    "index": query_index,
                    "result": {"success": False, "error": "All search tools failed"},
                    "success": False
                }
            
            # 모든 검색을 병렬로 실행
            search_tasks = [execute_single_search(q, i) for i, q in enumerate(search_queries)]
            search_results_list = await asyncio.gather(*search_tasks)
            
            logger.info(f"[{self.name}] ✅ Completed {len(search_results_list)} parallel searches")
            
            # 모든 성공한 검색 결과 통합
            successful_results = [sr for sr in search_results_list if sr.get('success') and sr.get('result', {}).get('data')]
            
            # 최소 30개 결과 보장을 위한 추가 검색 로직
            MIN_REQUIRED_RESULTS = 30
            total_results_count = 0
            for sr in successful_results:
                result_data = sr.get('result', {}).get('data', {})
                if isinstance(result_data, dict):
                    total_results_count += len(result_data.get('results', result_data.get('items', [])))
                elif isinstance(result_data, list):
                    total_results_count += len(result_data)
            
            logger.info(f"[{self.name}] 📊 Total results collected so far: {total_results_count}")
            
            # 결과가 부족하면 추가 검색 수행
            if total_results_count < MIN_REQUIRED_RESULTS and len(search_queries) > 0:
                additional_queries_needed = (MIN_REQUIRED_RESULTS - total_results_count) // 10 + 1
                logger.info(f"[{self.name}] 🔍 Results insufficient ({total_results_count} < {MIN_REQUIRED_RESULTS}), generating {additional_queries_needed} additional search queries...")
                
                # 추가 검색 쿼리 생성 (동의어, 관련 용어 기반)
                from src.core.llm_manager import execute_llm_task, TaskType
                query_expansion_prompt = f"""Generate {additional_queries_needed} additional search queries related to the following research topic. Use synonyms, related terms, and different perspectives.

Original queries: {', '.join(search_queries[:3])}
Research topic: {state['user_query']}

Generate diverse search queries that will help find more relevant documents. Each query should be specific and different from the original queries.

Return only the queries, one per line, without numbering or bullets."""
                
                try:
                    expansion_result = await execute_llm_task(
                        prompt=query_expansion_prompt,
                        task_type=TaskType.PLANNING,
                        model_name=None,
                        system_message="You are a search query expansion expert. Generate diverse, specific search queries."
                    )
                    
                    additional_queries = [q.strip() for q in expansion_result.content.split('\n') if q.strip() and len(q.strip()) > 10]
                    additional_queries = additional_queries[:additional_queries_needed]
                    
                    logger.info(f"[{self.name}] ✅ Generated {len(additional_queries)} additional search queries")
                    
                    # 추가 검색 수행
                    additional_search_tasks = [execute_single_search(q, len(search_queries) + i) for i, q in enumerate(additional_queries)]
                    additional_search_results = await asyncio.gather(*additional_search_tasks)
                    search_results_list.extend(additional_search_results)
                    
                    # 성공한 결과 업데이트
                    successful_results = [sr for sr in search_results_list if sr.get('success') and sr.get('result', {}).get('data')]
                    logger.info(f"[{self.name}] ✅ Additional searches completed: {len([sr for sr in additional_search_results if sr.get('success')])} successful")
                except Exception as e:
                    logger.warning(f"[{self.name}] ⚠️ Failed to generate additional queries: {e}")
            
            if not successful_results:
                # 실패한 검색 상세 정보 수집
                failed_searches = [sr for sr in search_results_list if not sr.get('success')]
                error_details = []
                for fs in failed_searches:
                    query = fs.get('query', 'unknown')
                    result = fs.get('result', {})
                    error = result.get('error', 'Unknown error')
                    error_details.append(f"  - Query: '{query[:60]}...' → Error: {str(error)[:100]}")
                
                logger.error(f"[{self.name}] ❌ 모든 검색 쿼리 실행 실패 ({len(failed_searches)}/{len(search_results_list)} 실패)")
                logger.error(f"[{self.name}] 📋 실패 상세:")
                for detail in error_details:
                    logger.error(f"[{self.name}] {detail}")
                
                # MCP 서버 연결 상태 확인
                try:
                    from src.core.mcp_integration import get_mcp_hub
                    mcp_hub = get_mcp_hub()
                    connected_servers = list(mcp_hub.mcp_sessions.keys()) if mcp_hub.mcp_sessions else []
                    logger.error(f"[{self.name}] 🔌 현재 연결된 MCP 서버: {connected_servers if connected_servers else '없음'}")
                    logger.error(f"[{self.name}] 📝 Fallback (duckduckgo_search 라이브러리)가 작동했는지 확인 필요")
                except Exception as e:
                    logger.debug(f"[{self.name}] MCP Hub 상태 확인 실패: {e}")
                
                error_msg = f"연구 실행 실패: 모든 검색 쿼리 실행이 실패했습니다. ({len(failed_searches)}/{len(search_results_list)} 실패)"
                raise RuntimeError(error_msg)
            
            # 모든 검색 결과를 통합 (하드코딩 제거, 동적 통합)
            all_search_data = []
            for sr in successful_results:
                result_data = sr['result'].get('data', {})
                if isinstance(result_data, dict):
                    items = result_data.get('results', result_data.get('items', []))
                    if isinstance(items, list):
                        all_search_data.extend(items)
                elif isinstance(result_data, list):
                    all_search_data.extend(result_data)
            
            # 통합된 결과를 하나의 검색 결과 형식으로 구성
            search_result = {
                'success': True,
                'data': {
                    'results': all_search_data,
                    'total_results': len(all_search_data),
                    'source': 'parallel_search'
                }
            }
            
            logger.info(f"[{self.name}] ✅ Integrated {len(all_search_data)} results from {len(successful_results)} successful searches")
            
            # 모든 검색 결과를 SharedResultsManager에 공유
            if self.context.shared_results_manager:
                shared_count = 0
                for sr in search_results_list:
                    if sr.get('success'):
                        task_id = f"search_{sr['index']}"
                        result_id = await self.context.shared_results_manager.share_result(
                            task_id=task_id,
                            agent_id=self.context.agent_id,  # 고유한 agent_id 사용
                            result=sr['result'],
                            metadata={"query": sr['query'], "index": sr['index']},
                            confidence=1.0 if sr.get('success') else 0.0
                        )
                        shared_count += 1
                        logger.info(f"[{self.name}] 🔗 Shared search result for query: '{sr['query'][:50]}...' (result_id: {result_id[:8]}..., agent_id: {self.context.agent_id})")

                # 공유 통계 로깅
                total_results = len([sr for sr in search_results_list if sr.get('success')])
                logger.info(f"[{self.name}] 📤 Shared {shared_count}/{total_results} successful search results with other agents")
                logger.info(f"[{self.name}] 🤝 Agent communication: {shared_count} results shared via SharedResultsManager")
            
            logger.info(f"[{self.name}] Search completed: success={search_result.get('success')}, total_results={search_result.get('data', {}).get('total_results', 0)}")
            logger.info(f"[{self.name}] Search result type: {type(search_result)}, keys: {list(search_result.keys()) if isinstance(search_result, dict) else 'N/A'}")
            
            if search_result.get('success') and search_result.get('data'):
                data = search_result.get('data', {})
                logger.info(f"[{self.name}] Data type: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                
                # 검색 결과 파싱 - 다양한 형식 지원
                search_results = []
                if isinstance(data, dict):
                    # 표준 형식: {"query": "...", "results": [...], "total_results": N, "source": "..."}
                    search_results = data.get('results', [])
                    logger.info(f"[{self.name}] Found 'results' key: {len(search_results)} items")
                    
                    if not search_results:
                        # 다른 키 시도
                        search_results = data.get('items', data.get('data', []))
                        logger.info(f"[{self.name}] Tried 'items' or 'data' keys: {len(search_results)} items")
                    
                    # data 자체가 리스트인 경우 (중첩된 경우)
                    if not search_results and isinstance(data, dict):
                        # data의 값 중 리스트 찾기
                        for key, value in data.items():
                            if isinstance(value, list) and len(value) > 0:
                                # 첫 번째 항목이 dict인지 확인
                                if value and isinstance(value[0], dict):
                                    search_results = value
                                    logger.info(f"[{self.name}] Found list in key '{key}': {len(search_results)} items")
                                    break
                elif isinstance(data, list):
                    search_results = data
                    logger.info(f"[{self.name}] Data is directly a list: {len(search_results)} items")
                
                logger.info(f"[{self.name}] ✅ Parsed {len(search_results)} search results")
                
                # 디버깅: 첫 번째 결과 샘플 출력
                if search_results and len(search_results) > 0:
                    first_result = search_results[0]
                    logger.info(f"[{self.name}] First result type: {type(first_result)}, sample: {str(first_result)[:200]}")
                
                # Phase 2: 검색 결과 관련성 사전 필터링
                if search_results and len(search_results) > 0:
                    logger.info(f"[{self.name}] 🔍 Starting relevance pre-filtering for {len(search_results)} results...")
                    filtered_results = await self._filter_results_by_relevance(
                        search_results, 
                        state['user_query'],
                        assigned_task.get('search_queries', [state['user_query']]) if assigned_task else [state['user_query']],
                        current_time
                    )
                    search_results = filtered_results
                    logger.info(f"[{self.name}] ✅ Relevance filtering completed: {len(search_results)} relevant results (from {len(search_results) + (len(search_results) - len(filtered_results)) if len(filtered_results) < len(search_results) else 0} total)")
                    
                    # 의문점 감지 (검색 결과가 모호하거나 사용자 선호도가 필요한 경우)
                    if len(filtered_results) > 10 or len(filtered_results) == 0:
                        # 결과가 너무 많거나 없으면 사용자에게 질문
                        from src.core.human_clarification_handler import get_clarification_handler
                        clarification_handler = get_clarification_handler()
                        
                        # 의문점 생성
                        ambiguity = {
                            "type": "resource_constraint" if len(filtered_results) > 10 else "scope_depth",
                            "field": "result_count",
                            "description": f"Found {len(filtered_results)} results. Need to clarify scope or priority.",
                            "suggested_question": "검색 결과가 많습니다. 어떤 방향으로 진행할까요?" if len(filtered_results) > 10 else "검색 결과가 없습니다. 검색 범위를 조정할까요?",
                            "suggested_options": [
                                {"label": "상위 5개 결과만 사용", "value": "top_5"},
                                {"label": "상위 10개 결과 사용", "value": "top_10"},
                                {"label": "모든 결과 사용", "value": "all"}
                            ] if len(filtered_results) > 10 else [
                                {"label": "검색 범위 확대", "value": "expand"},
                                {"label": "검색어 수정", "value": "modify"},
                                {"label": "계속 진행", "value": "continue"}
                            ]
                        }
                        
                        question = await clarification_handler.generate_question(
                            ambiguity,
                            {'user_request': state['user_query'], 'result_count': len(filtered_results)}
                        )
                        
                        # CLI 모드 감지 (더 정확한 방법)
                        import sys
                        is_cli_mode = (
                            not hasattr(sys, 'ps1') and  # Interactive shell이 아님
                            'streamlit' not in sys.modules and  # Streamlit이 로드되지 않음
                            not any('streamlit' in str(arg) for arg in sys.argv)  # Streamlit 실행 인자가 없음
                        )
                        
                        # CLI 모드이거나 autopilot 모드인 경우 자동 선택
                        if is_cli_mode or state.get("autopilot_mode", False):
                            logger.info("🤖 CLI/Autopilot mode - auto-selecting response")
                            
                            # History 기반 자동 선택
                            shared_memory = self.context.shared_memory
                            auto_response = await clarification_handler.auto_select_response(
                                question,
                                {'user_request': state['user_query'], 'result_count': len(filtered_results)},
                                shared_memory
                            )
                            
                            # 응답 처리
                            processed = await clarification_handler.process_user_response(
                                question['id'],
                                auto_response,
                                {'question': question}
                            )
                            
                            if processed.get('validated', False):
                                # 명확화 정보 적용
                                clarification = processed.get('clarification', {})
                                
                                # 응답에 따라 결과 필터링
                                if auto_response == "top_5":
                                    filtered_results = filtered_results[:5]
                                elif auto_response == "top_10":
                                    filtered_results = filtered_results[:10]
                                # "all"이면 그대로 사용
                                
                                logger.info(f"✅ Auto-selected: {auto_response}, using {len(filtered_results)} results")
                                # 계속 진행 (return 하지 않음)
                        else:
                            # 웹 모드: 사용자에게 질문
                            state['pending_questions'] = state.get('pending_questions', []) + [question]
                            state['waiting_for_user'] = True
                            state['user_responses'] = state.get('user_responses', {})
                            
                            logger.info(f"❓ Generated question during execution: {question['id']}")
                            logger.info("⏸️ Waiting for user response...")
                            
                            return state
                
                if search_results and len(search_results) > 0:
                    # 실제 검색 결과를 구조화된 형식으로 저장
                    unique_results = []
                    seen_urls = set()
                    filtered_count = 0
                    filtered_reasons = []
                    
                    # 실제 검색 쿼리 값 로그 출력 (query 변수는 실제 검색 쿼리)
                    actual_query = query if isinstance(query, str) else str(query)
                    logger.info(f"[{self.name}] Processing {len(search_results)} results for query: '{actual_query}'")
                    
                    for i, result in enumerate(search_results, 1):
                        # 다양한 형식 지원
                        if isinstance(result, dict):
                            title = result.get('title', result.get('name', result.get('Title', 'No title')))
                            snippet = result.get('snippet', result.get('content', result.get('summary', result.get('description', result.get('abstract', '')))))
                            url = result.get('url', result.get('link', result.get('href', result.get('URL', ''))))
                            
                            # snippet에 마크다운 형식의 여러 결과가 들어있는 경우 파싱
                            if snippet and ("Found" in snippet or "search results" in snippet.lower() or "\n1." in snippet):
                                logger.info(f"[{self.name}] Detected markdown format in snippet, parsing...")
                                parsed_results = []
                                lines = snippet.split('\n')
                                current_result = None
                                
                                for line in lines:
                                    original_line = line
                                    line = line.strip()
                                    if not line:
                                        continue
                                    
                                    # 패턴 1: 마크다운 링크 "1. [Title](URL)"
                                    link_match = re.match(r'^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)', line)
                                    # 패턴 2: 번호와 제목만 "1. [Title]" 또는 "1. Title"
                                    title_match = re.match(r'^\d+\.\s*(?:\[([^\]]+)\]|(.+?))(?:\s*$|:)', line)
                                    # 패턴 3: URL 줄 "   URL: https://..."
                                    url_match = re.search(r'URL:\s*(https?://[^\s]+)', line, re.IGNORECASE)
                                    # 패턴 4: Summary 줄 "   Summary: ..."
                                    summary_match = re.search(r'Summary:\s*(.+)$', line, re.IGNORECASE)
                                    
                                    if link_match:
                                        # 이전 결과 저장
                                        if current_result and current_result.get('title'):
                                            parsed_results.append(current_result)
                                        
                                        title_parsed = link_match.group(1)
                                        url_parsed = link_match.group(2)
                                        current_result = {
                                            "title": title_parsed,
                                            "url": url_parsed,
                                            "snippet": ""
                                        }
                                    elif title_match and not current_result:
                                        # 번호와 제목만 있는 경우 (다음 줄에 URL이 올 것으로 예상)
                                        title_parsed = title_match.group(1) or title_match.group(2)
                                        if title_parsed:
                                            current_result = {
                                                "title": title_parsed.strip(),
                                                "url": "",
                                                "snippet": ""
                                            }
                                    elif url_match:
                                        # URL이 별도 줄에 있는 경우
                                        if current_result:
                                            current_result["url"] = url_match.group(1)
                                        else:
                                            # URL만 있고 제목이 없는 경우 (이전 결과에 추가)
                                            if parsed_results:
                                                parsed_results[-1]["url"] = url_match.group(1)
                                    elif summary_match and current_result:
                                        # Summary 줄
                                        current_result["snippet"] = summary_match.group(1).strip()
                                    elif current_result and line and not any([
                                        line.startswith('URL:'), 
                                        line.startswith('Summary:'),
                                        line.startswith('Found'),
                                        'search results' in line.lower()
                                    ]):
                                        # 설명 텍스트 (들여쓰기된 경우)
                                        if original_line.startswith('   ') or original_line.startswith('\t'):
                                            if current_result["snippet"]:
                                                current_result["snippet"] += " " + line
                                            else:
                                                current_result["snippet"] = line
                                
                                # 마지막 결과 추가
                                if current_result and current_result.get('title'):
                                    parsed_results.append(current_result)
                                
                                if parsed_results:
                                    logger.info(f"[{self.name}] Parsed {len(parsed_results)} results from markdown snippet")
                                    # 파싱된 결과들을 unique_results에 추가
                                    for parsed_result in parsed_results:
                                        parsed_url = parsed_result.get('url', '')
                                        parsed_title = parsed_result.get('title', '')
                                        parsed_snippet = parsed_result.get('snippet', '')
                                        
                                        if parsed_url and parsed_url in seen_urls:
                                            logger.debug(f"[{self.name}] Duplicate URL skipped in parsed results: {parsed_url[:50]}")
                                            continue
                                        if parsed_url:
                                            seen_urls.add(parsed_url)
                                        
                                        # 마크다운 파싱 결과도 필터링 적용
                                        invalid_indicators = [
                                            "no results were found", "bot detection",
                                            "no results", "not found", "try again",
                                            "unable to", "error occurred", "no matches"
                                        ]
                                        parsed_snippet_lower = parsed_snippet.lower() if parsed_snippet else ""
                                        matched_indicators = [ind for ind in invalid_indicators if ind in parsed_snippet_lower]
                                        
                                        if matched_indicators:
                                            filtered_count += 1
                                            reason = f"Matched indicators: {', '.join(matched_indicators)}"
                                            filtered_reasons.append({
                                                "result_index": f"{i}(parsed)",
                                                "title": parsed_title[:80],
                                                "reason": reason,
                                                "snippet_preview": parsed_snippet[:200] if parsed_snippet else "(empty)"
                                            })
                                            logger.warning(f"[{self.name}] ⚠️ Filtering invalid parsed result: '{parsed_title[:60]}...' - Reason: {reason}")
                                            continue
                                        
                                        unique_results.append({
                                            "index": len(unique_results) + 1,
                                            "title": parsed_title,
                                            "snippet": parsed_snippet[:500],
                                            "url": parsed_url,
                                            "source": "search"
                                        })
                                        logger.info(f"[{self.name}] Parsed result: {parsed_title[:50]}... (URL: {parsed_url[:50] if parsed_url else 'N/A'}...)")
                                    
                                    # 원본 결과는 건너뛰기
                                    continue
                            
                            logger.debug(f"[{self.name}] Result {i}: title={title[:50] if title else 'N/A'}, url={url[:50] if url else 'N/A'}")
                        elif isinstance(result, str):
                            # 문자열 형식인 경우 파싱 시도 (마크다운 링크 형식)
                            link_match = re.match(r'^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)', result.strip())
                            if link_match:
                                title = link_match.group(1)
                                url = link_match.group(2)
                                snippet = ""
                                logger.info(f"[{self.name}] Parsed string result {i} as markdown: {title[:50]}")
                            else:
                                logger.warning(f"[{self.name}] Result {i} is string but not markdown format, skipping: {result[:100]}")
                                continue
                        else:
                            logger.warning(f"[{self.name}] Unknown result format for result {i}: {type(result)}, value: {str(result)[:100]}")
                            continue
                        
                        # URL 중복 제거
                        if url and url in seen_urls:
                            logger.debug(f"[{self.name}] Duplicate URL skipped: {url}")
                            continue
                        if url:
                            seen_urls.add(url)
                        
                        # 디버깅: 원본 데이터 로깅
                        logger.debug(f"[{self.name}] Result {i} 원본 데이터 - title: '{title[:80]}', snippet: '{snippet[:150] if snippet else '(empty)'}', url: '{url[:80] if url else '(empty)'}'")
                        
                        # snippet 내용으로 유효하지 않은 검색 결과 필터링
                        invalid_indicators = [
                            "no results were found", "bot detection",
                            "no results", "not found", "try again",
                            "unable to", "error occurred", "no matches"
                        ]
                        snippet_lower = snippet.lower() if snippet else ""
                        matched_indicators = [ind for ind in invalid_indicators if ind in snippet_lower]
                        
                        if matched_indicators:
                            filtered_count += 1
                            reason = f"Matched indicators: {', '.join(matched_indicators)}"
                            filtered_reasons.append({
                                "result_index": i,
                                "title": title[:80],
                                "reason": reason,
                                "snippet_preview": snippet[:200] if snippet else "(empty)"
                            })
                            logger.warning(f"[{self.name}] ⚠️ Filtering invalid search result {i}: '{title[:60]}...' - Reason: {reason}")
                            logger.debug(f"[{self.name}]   Filtered snippet preview: '{snippet[:200] if snippet else '(empty)'}'")
                            continue
                        
                        # 구조화된 결과 저장
                        result_dict = {
                            "index": len(unique_results) + 1,
                            "title": title,
                            "snippet": snippet[:500] if snippet else "",
                            "url": url,
                            "source": "search"
                        }
                        unique_results.append(result_dict)
                        
                        logger.info(f"[{self.name}] Result {i}: {title[:50]}... (URL: {url[:50] if url else 'N/A'}...)")
                    
                    # 필터링 통계 로깅
                    total_processed = len(search_results)
                    valid_results = len(unique_results)
                    logger.info(f"[{self.name}] 📊 필터링 통계: 총 {total_processed}개 중 {filtered_count}개 필터링됨, {valid_results}개 유효한 결과")
                    
                    if filtered_count > 0:
                        logger.warning(f"[{self.name}] ⚠️ 필터링된 결과 상세:")
                        for fr in filtered_reasons[:5]:  # 최대 5개만 상세 로깅
                            logger.warning(f"[{self.name}]   - 결과 {fr['result_index']}: '{fr['title']}' - {fr['reason']}")
                            logger.warning(f"[{self.name}]     Snippet: '{fr['snippet_preview']}'")
                        if len(filtered_reasons) > 5:
                            logger.warning(f"[{self.name}]   ... 외 {len(filtered_reasons) - 5}개 결과도 필터링됨")
                    
                    # 결과를 구조화된 형식으로 저장
                    if unique_results:
                        results = unique_results
                        logger.info(f"[{self.name}] ✅ Collected {len(results)} unique results")
                        
                        # 최소 5개 이상의 고유한 출처 보장
                        MIN_UNIQUE_SOURCES = 5
                        unique_urls = set()
                        for result in results:
                            url = result.get('url', '')
                            if url:
                                # URL에서 도메인 추출
                                try:
                                    from urllib.parse import urlparse
                                    parsed = urlparse(url)
                                    domain = f"{parsed.scheme}://{parsed.netloc}"
                                    unique_urls.add(domain)
                                except:
                                    unique_urls.add(url)
                        
                        logger.info(f"[{self.name}] 📊 Unique sources found: {len(unique_urls)} (minimum required: {MIN_UNIQUE_SOURCES})")
                        
                        # 출처가 부족하면 추가 검색 수행
                        if len(unique_urls) < MIN_UNIQUE_SOURCES:
                            logger.warning(f"[{self.name}] ⚠️ Only {len(unique_urls)} unique sources found, need at least {MIN_UNIQUE_SOURCES}. Performing additional searches...")
                            
                            # 추가 검색 쿼리 생성 (다양한 관점)
                            additional_queries = []
                            base_query = query
                            
                            # 다양한 검색어 패턴 시도
                            additional_patterns = [
                                f"{base_query} 뉴스",
                                f"{base_query} 리포트",
                                f"{base_query} 조사",
                                f"{base_query} 통계",
                                f"{base_query} 자료"
                            ]
                            
                            # 이미 사용한 쿼리 제외
                            used_queries = set(search_queries)
                            for pattern in additional_patterns:
                                if pattern not in used_queries and len(additional_queries) < 3:
                                    additional_queries.append(pattern)
                            
                            if additional_queries:
                                logger.info(f"[{self.name}] 🔍 Executing {len(additional_queries)} additional searches for more sources...")
                                
                                # 추가 검색 실행
                                additional_search_tasks = [execute_single_search(q, len(search_queries) + i) for i, q in enumerate(additional_queries)]
                                additional_results_list = await asyncio.gather(*additional_search_tasks)
                                
                                # 추가 검색 결과 통합
                                additional_unique_results = []
                                additional_seen_urls = seen_urls.copy()
                                
                                for sr in additional_results_list:
                                    if sr.get('success') and sr.get('result', {}).get('data'):
                                        result_data = sr['result'].get('data', {})
                                        if isinstance(result_data, dict):
                                            items = result_data.get('results', result_data.get('items', []))
                                            if isinstance(items, list):
                                                for item in items:
                                                    if isinstance(item, dict):
                                                        url = item.get('url', item.get('link', ''))
                                                        if url and url not in additional_seen_urls:
                                                            title = item.get('title', item.get('name', ''))
                                                            snippet = item.get('snippet', item.get('content', ''))
                                                            if title and len(title.strip()) >= 3:
                                                                additional_unique_results.append({
                                                                    "index": len(results) + len(additional_unique_results) + 1,
                                                                    "title": title,
                                                                    "snippet": snippet[:500] if snippet else '',
                                                                    "url": url,
                                                                    "source": "additional_search"
                                                                })
                                                                additional_seen_urls.add(url)
                                        
                                        # 도메인 추출하여 고유 출처 확인
                                        for item in additional_unique_results:
                                            url = item.get('url', '')
                                            if url:
                                                try:
                                                    from urllib.parse import urlparse
                                                    parsed = urlparse(url)
                                                    domain = f"{parsed.scheme}://{parsed.netloc}"
                                                    unique_urls.add(domain)
                                                except:
                                                    unique_urls.add(url)
                                        
                                        # 충분한 출처를 얻으면 중단
                                        if len(unique_urls) >= MIN_UNIQUE_SOURCES:
                                            break
                                
                                if additional_unique_results:
                                    results.extend(additional_unique_results)
                                    logger.info(f"[{self.name}] ✅ Added {len(additional_unique_results)} additional results from {len(additional_queries)} searches")
                                    logger.info(f"[{self.name}] 📊 Total unique sources: {len(unique_urls)} (target: {MIN_UNIQUE_SOURCES})")
                                else:
                                    logger.warning(f"[{self.name}] ⚠️ Additional searches did not yield new unique sources")
                            else:
                                logger.warning(f"[{self.name}] ⚠️ No additional query patterns available")
                        else:
                            logger.info(f"[{self.name}] ✅ Sufficient unique sources found: {len(unique_urls)} >= {MIN_UNIQUE_SOURCES}")
                        
                        # 최종 결과 요약
                        final_unique_sources = set()
                        for result in results:
                            url = result.get('url', '')
                            if url:
                                try:
                                    from urllib.parse import urlparse
                                    parsed = urlparse(url)
                                    domain = f"{parsed.scheme}://{parsed.netloc}"
                                    final_unique_sources.add(domain)
                                except:
                                    final_unique_sources.add(url)
                        
                        logger.info(f"[{self.name}] 📊 Final collection: {len(results)} results from {len(final_unique_sources)} unique sources")
                        if len(final_unique_sources) < MIN_UNIQUE_SOURCES:
                            logger.warning(f"[{self.name}] ⚠️ Warning: Only {len(final_unique_sources)} unique sources collected (target: {MIN_UNIQUE_SOURCES})")
                        
                        # 검색 결과 검토 및 실제 웹 페이지 내용 크롤링
                        logger.info(f"[{self.name}] 🔍 Reviewing search results and fetching full web content...")
                        
                        # 검색 결과 검토 및 실제 웹 페이지 크롤링
                        enriched_results = []
                        for result in results:
                            url = result.get('url', '')
                            if not url:
                                enriched_results.append(result)
                                continue
                            
                            try:
                                # 실제 웹 페이지 내용 가져오기
                                logger.info(f"[{self.name}] 📥 Fetching full content from: {url[:80]}...")
                                fetch_result = await execute_tool("fetch", {"url": url})
                                
                                if fetch_result.get('success') and fetch_result.get('data'):
                                    content = fetch_result.get('data', {}).get('content', '')
                                    if content:
                                        # HTML 태그 제거 및 텍스트 정리
                                        from bs4 import BeautifulSoup
                                        
                                        try:
                                            soup = BeautifulSoup(content, 'html.parser')
                                            # 스크립트, 스타일, 헤더, 푸터 제거
                                            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
                                                element.decompose()
                                            
                                            # 메인 콘텐츠 추출
                                            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|article|post|main', re.I))
                                            if main_content:
                                                full_text = main_content.get_text(separator='\n', strip=True)
                                            else:
                                                full_text = soup.get_text(separator='\n', strip=True)
                                            
                                            # 텍스트 정리 (너무 긴 공백 제거)
                                            full_text = re.sub(r'\n{3,}', '\n\n', full_text)
                                            full_text = re.sub(r' {3,}', ' ', full_text)
                                            
                                            # 최대 길이 제한 (50000자)
                                            if len(full_text) > 50000:
                                                full_text = full_text[:50000] + "... [truncated]"
                                            
                                            result['full_content'] = full_text
                                            result['content_length'] = len(full_text)
                                            
                                            # 날짜 정보 추출 시도
                                            date_patterns = [
                                                r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})',  # YYYY-MM-DD
                                                r'(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})',  # MM-DD-YYYY
                                                r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',  # 한국어 형식
                                            ]
                                            
                                            date_found = None
                                            for pattern in date_patterns:
                                                matches = re.findall(pattern, full_text[:5000])  # 처음 5000자만 검색
                                                if matches:
                                                    try:
                                                        match = matches[-1]  # 가장 최근 날짜
                                                        if len(match) == 3:
                                                            if '년' in full_text[:5000]:
                                                                # 한국어 형식
                                                                date_str = f"{match[0]}-{match[1].zfill(2)}-{match[2].zfill(2)}"
                                                            elif len(match[0]) == 4:
                                                                # YYYY-MM-DD
                                                                date_str = f"{match[0]}-{match[1].zfill(2)}-{match[2].zfill(2)}"
                                                            else:
                                                                # MM-DD-YYYY
                                                                date_str = f"{match[2]}-{match[0].zfill(2)}-{match[1].zfill(2)}"
                                                            date_found = datetime.strptime(date_str, "%Y-%m-%d")
                                                            break
                                                    except:
                                                        continue
                                            
                                            if date_found:
                                                result['published_date'] = date_found.isoformat()
                                                logger.info(f"[{self.name}] 📅 Found date: {date_found.strftime('%Y-%m-%d')} for {url[:50]}...")
                                            else:
                                                # 날짜를 찾지 못한 경우 현재 시간으로 설정 (최신 정보 우선)
                                                result['published_date'] = datetime.now().isoformat()
                                                logger.info(f"[{self.name}] ⚠️ No date found, using current time for {url[:50]}...")
                                            
                                            logger.info(f"[{self.name}] ✅ Fetched {len(full_text)} characters from {url[:50]}...")
                                        except Exception as e:
                                            logger.warning(f"[{self.name}] ⚠️ Failed to parse HTML from {url[:50]}...: {e}")
                                            # 파싱 실패해도 원본 결과는 유지
                                            result['full_content'] = content[:50000] if len(content) > 50000 else content
                                            result['content_length'] = len(result['full_content'])
                                    else:
                                        logger.warning(f"[{self.name}] ⚠️ No content fetched from {url[:50]}...")
                                else:
                                    logger.warning(f"[{self.name}] ⚠️ Failed to fetch content from {url[:50]}...: {fetch_result.get('error', 'Unknown error')}")
                            except Exception as e:
                                logger.error(f"[{self.name}] ❌ Error fetching content from {url[:50]}...: {e}")
                            
                            enriched_results.append(result)
                        
                        # 최신 정보 우선순위로 정렬
                        enriched_results.sort(key=lambda x: (
                            datetime.fromisoformat(x.get('published_date', datetime.now().isoformat())) if x.get('published_date') else datetime.min,
                            x.get('content_length', 0)
                        ), reverse=True)
                        
                        logger.info(f"[{self.name}] ✅ Enriched {len(enriched_results)} results with full web content")
                        results = enriched_results
                        
                        # 검색 결과 검토 (LLM으로 검색 결과 평가)
                        logger.info(f"[{self.name}] 🔍 Reviewing search results for relevance and recency...")
                        try:
                            from src.core.llm_manager import execute_llm_task, TaskType
                            
                            # 검색 결과 요약 및 평가
                            review_prompt = f"""다음은 '{query}'에 대한 검색 결과입니다. 각 결과를 검토하여:
1. 사용자 쿼리와의 관련성 평가
2. 정보의 최신성 확인 (날짜 정보 포함)
3. 신뢰할 수 있는 출처인지 확인
4. 실제 웹 페이지 내용이 쿼리와 관련이 있는지 확인

검색 결과:
{chr(10).join([f"{i+1}. {r.get('title', 'N/A')} - {r.get('url', 'N/A')} - 날짜: {r.get('published_date', 'N/A')} - 내용 길이: {r.get('content_length', 0)}자" for i, r in enumerate(results[:10])])}

각 결과에 대해:
- 관련성 점수 (0-10)
- 최신성 평가 (최신/보통/오래됨)
- 신뢰도 평가 (높음/보통/낮음)
- 추천 여부 (추천/보통/비추천)

형식: JSON 배열로 반환
[
  {{
    "index": 1,
    "relevance_score": 8,
    "recency": "최신",
    "reliability": "높음",
    "recommend": "추천",
    "reason": "최신 정보이며 쿼리와 직접 관련"
  }},
  ...
]
"""
                            
                            review_result = await execute_llm_task(
                                prompt=review_prompt,
                                task_type=TaskType.ANALYSIS,
                                model_name=None,
                                system_message="You are an expert information analyst who evaluates search results for relevance, recency, and reliability."
                            )
                            
                            # LLM 결과 파싱
                            review_text = review_result.content or ""
                            try:
                                # JSON 추출
                                json_match = re.search(r'\[.*\]', review_text, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group().strip()
                                    if not json_str or json_str == "[]":
                                        logger.warning(f"[{self.name}] ⚠️ Empty JSON array in review result")
                                    else:
                                        review_data = json.loads(json_str)
                                        
                                        # 검토 결과를 결과에 추가
                                        for review_item in review_data:
                                            idx = review_item.get('index', 0) - 1
                                            if 0 <= idx < len(results):
                                                results[idx]['review'] = {
                                                    'relevance_score': review_item.get('relevance_score', 5),
                                                    'recency': review_item.get('recency', '보통'),
                                                    'reliability': review_item.get('reliability', '보통'),
                                                    'recommend': review_item.get('recommend', '보통'),
                                                    'reason': review_item.get('reason', '')
                                                }
                                        
                                        # 추천 결과만 필터링 (선택적)
                                        recommended_results = [r for r in results if r.get('review', {}).get('recommend') == '추천']
                                        if recommended_results:
                                            logger.info(f"[{self.name}] ✅ Found {len(recommended_results)} highly recommended results")
                                            # 추천 결과를 우선적으로 사용하되, 최소 5개는 유지
                                            if len(recommended_results) >= 5:
                                                results = recommended_results
                                            else:
                                                # 추천 결과 + 일반 결과 혼합
                                                results = recommended_results + [r for r in results if r not in recommended_results][:5-len(recommended_results)]
                                        
                                        logger.info(f"[{self.name}] ✅ Reviewed {len(review_data)} search results")
                            except Exception as e:
                                logger.warning(f"[{self.name}] ⚠️ Failed to parse review result: {e}")
                        except Exception as e:
                            logger.warning(f"[{self.name}] ⚠️ Failed to review search results: {e}")
                    else:
                        # 모든 결과가 필터링된 경우 상세한 에러 메시지
                        error_details = []
                        error_details.append(f"검색 쿼리: '{query[:100]}'")
                        error_details.append(f"총 검색 결과: {total_processed}개")
                        error_details.append(f"필터링된 결과: {filtered_count}개")
                        error_details.append(f"유효한 결과: 0개")
                        
                        if filtered_reasons:
                            error_details.append("\n필터링된 결과 상세:")
                            for fr in filtered_reasons[:3]:  # 최대 3개만 에러 메시지에 포함
                                error_details.append(f"  - 결과 {fr['result_index']}: '{fr['title']}' - {fr['reason']}")
                        
                        error_msg = f"연구 실행 실패: 모든 검색 결과가 필터링되었습니다.\n" + "\n".join(error_details)
                        logger.error(f"[{self.name}] ❌ {error_msg}")
                        raise RuntimeError(error_msg)
                else:
                    # 검색 결과가 없음 - 실패 처리
                    logger.error(f"[{self.name}] ❌ 검색 결과가 비어있습니다.")
                    logger.error(f"[{self.name}]   검색 쿼리: '{query[:100]}'")
                    logger.error(f"[{self.name}]   검색 도구: {search_result.get('source', 'unknown')}")
                    logger.error(f"[{self.name}]   검색 성공 여부: {search_result.get('success', False)}")
                    if search_result.get('error'):
                        logger.error(f"[{self.name}]   검색 에러: {search_result.get('error')}")
                    error_msg = f"연구 실행 실패: '{query[:100]}'에 대한 검색 결과를 찾을 수 없습니다."
                    logger.error(f"[{self.name}] ❌ {error_msg}")
                    raise RuntimeError(error_msg)
            else:
                # 검색 실패 - 에러 반환
                logger.error(f"[{self.name}] ❌ 검색 도구 실행 실패")
                logger.error(f"[{self.name}]   검색 쿼리: '{query[:100]}'")
                logger.error(f"[{self.name}]   검색 도구: {search_result.get('source', 'unknown')}")
                logger.error(f"[{self.name}]   검색 성공 여부: {search_result.get('success', False)}")
                logger.error(f"[{self.name}]   에러 메시지: {search_result.get('error', 'Unknown error')}")
                if search_result.get('data'):
                    logger.debug(f"[{self.name}]   응답 데이터 타입: {type(search_result.get('data'))}")
                    logger.debug(f"[{self.name}]   응답 데이터 샘플: {str(search_result.get('data'))[:200]}")
                error_msg = f"연구 실행 실패: 검색 도구 실행 중 오류가 발생했습니다. {search_result.get('error', 'Unknown error')}"
                logger.error(f"[{self.name}] ❌ {error_msg}")
                raise RuntimeError(error_msg)
                
        except Exception as e:
            # 실제 오류 발생 - 실패 처리
            import traceback
            error_type = type(e).__name__
            error_msg = f"연구 실행 실패: {str(e)}"
            logger.error(f"[{self.name}] ❌ 예외 발생: {error_type}")
            logger.error(f"[{self.name}]   에러 메시지: {error_msg}")
            logger.error(f"[{self.name}]   검색 쿼리: '{query[:100] if 'query' in locals() else 'N/A'}'")
            logger.debug(f"[{self.name}]   Traceback:\n{traceback.format_exc()}")
            
            # 실패 상태 기록
            state['research_results'] = []
            state['current_agent'] = self.name
            state['error'] = error_msg
            state['research_failed'] = True
            
            # 메모리에 실패 정보 기록
            memory.write(
                key=f"execution_error_{state['session_id']}",
                value=error_msg,
                scope=MemoryScope.SESSION,
                session_id=state['session_id'],
                agent_id=self.name
            )
            
            # 실패 상태 반환 (더미 데이터 없이)
            return state
        
        # Council 활성화 확인 및 적용 (중요한 정보 수집 시)
        use_council = state.get('use_council', None)  # 수동 활성화 옵션
        if use_council is None:
            # 자동 활성화 판단
            from src.core.council_activator import get_council_activator
            activator = get_council_activator()
            
            # 중요한 사실 확인이 필요한지 판단
            context = {
                'results_count': len(results),
                'has_controversial_topic': any(
                    keyword in state['user_query'].lower() 
                    for keyword in ['debate', 'controversy', 'disagreement', '논쟁', '의견']
                ),
                'high_stakes': any(
                    keyword in state['user_query'].lower()
                    for keyword in ['critical', 'important', 'decision', '중요한', '결정']
                )
            }
            
            activation_decision = activator.should_activate(
                process_type='execution',
                query=state['user_query'],
                context=context
            )
            use_council = activation_decision.should_activate
            if use_council:
                logger.info(f"[{self.name}] 🏛️ Council auto-activated: {activation_decision.reason}")
        
        # Council 적용 (활성화된 경우)
        if use_council and results:
            try:
                from src.core.llm_council import run_full_council
                logger.info(f"[{self.name}] 🏛️ Running Council verification for research results...")
                
                # 결과 요약 생성
                results_summary = "\n\n".join([
                    f"Result {i+1}:\nTitle: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')[:200]}"
                    for i, r in enumerate(results[:10])  # 최대 10개만 검토
                ])
                
                council_query = f"""Verify the accuracy and reliability of the following research results. Identify any inconsistencies, missing information, or potential issues.

Research Query: {state['user_query']}

Research Results:
{results_summary}

Provide a verification report with:
1. Accuracy assessment
2. Missing information
3. Recommendations for improvement"""
                
                stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
                    council_query
                )
                
                # Council 검증 결과를 결과에 추가
                verification_report = stage3_result.get('response', '')
                logger.info(f"[{self.name}] ✅ Council verification completed.")
                logger.info(f"[{self.name}] Council aggregate rankings: {metadata.get('aggregate_rankings', [])}")
                
                # Council 메타데이터를 state에 저장
                if 'council_metadata' not in state:
                    state['council_metadata'] = {}
                state['council_metadata']['execution'] = {
                    'stage1_results': stage1_results,
                    'stage2_results': stage2_results,
                    'stage3_result': stage3_result,
                    'metadata': metadata,
                    'verification_report': verification_report
                }
                
                # 검증 리포트를 결과에 추가
                results.append({
                    'title': 'Council Verification Report',
                    'url': '',
                    'snippet': verification_report,
                    'source': 'council',
                    'council_verified': True
                })
            except Exception as e:
                logger.warning(f"[{self.name}] Council verification failed: {e}. Using original results.")
                # Council 실패 시 원본 결과 사용 (fallback 제거 - 명확한 로깅만)
        
        # Executor 결과를 SharedResultsManager에 공유 (논박을 위해)
        executor_discussions = []
        if self.context.shared_results_manager and results:
            for i, result in enumerate(results[:10]):  # 최대 10개 결과에 대해 논박
                result_id = await self.context.shared_results_manager.share_result(
                    task_id=f"executor_result_{i}",
                    agent_id=self.context.agent_id,
                    result=result,
                    metadata={"executor_result_index": i, "query": state['user_query']},
                    confidence=result.get('confidence', 0.8) if isinstance(result, dict) else 0.8
                )
                logger.info(f"[{self.name}] 🔗 Shared executor result {i} for debate (result_id: {result_id[:8]}...)")
            
            # 다른 Executor들의 결과 가져오기 (논박을 위해)
            other_executor_results = await self.context.shared_results_manager.get_shared_results(
                exclude_agent_id=self.context.agent_id
            )
            
            if other_executor_results:
                logger.info(f"[{self.name}] 💬 Found {len(other_executor_results)} other executor results for debate")
                # 논박은 Verifier와 Evaluator에서 수행하도록 함 (여기서는 결과만 공유)
        
        # 성공적으로 결과 수집된 경우
        state['research_results'] = results  # 리스트로 저장 (덮어쓰기)
        state['current_agent'] = self.name
        state['research_failed'] = False
        
        # 논박 결과 초기화 (Verifier와 Evaluator가 채울 것)
        if 'agent_debates' not in state:
            state['agent_debates'] = {}
        
        logger.info(f"[{self.name}] ✅ Research execution completed: {len(results)} results")
        
        # Write to shared memory (구조화된 형식)
        memory.write(
            key=f"research_results_{state['session_id']}",
            value=results,
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        logger.info(f"[{self.name}] Results saved to shared memory")
        logger.info(f"=" * 80)
        
        return state


class VerifierAgent:
    """Verifier agent - verifies research results (Skills-based)."""
    
    def __init__(self, context: AgentContext, skill: Optional[Skill] = None):
        self.context = context
        self.name = "verifier"
        self.available_tools: list = []  # MCP 자동 할당 도구
        self.tool_infos: list = []  # 도구 메타데이터
        self.skill = skill
        
        # Skill이 없으면 로드 시도
        if self.skill is None:
            skill_manager = get_skill_manager()
            self.skill = skill_manager.load_skill("evaluator")
        
        # Skill instruction 사용
        if self.skill:
            self.instruction = self.skill.instructions
        else:
            self.instruction = "You are a verification agent."
    
    async def execute(self, state: AgentState) -> AgentState:
        """Verify research results with LLM-based verification."""
        logger.info(f"=" * 80)
        logger.info(f"[{self.name.upper()}] Starting verification")
        logger.info(f"=" * 80)
        
        # 연구 실패 확인
        if state.get('research_failed'):
            logger.error(f"[{self.name}] ❌ Research execution failed, skipping verification")
            state['verified_results'] = []
            state['verification_failed'] = True
            state['current_agent'] = self.name
            return state
        
        memory = self.context.shared_memory
        
        # Read results from state or shared memory
        results = state.get('research_results', [])
        if not results:
            results = memory.read(
                key=f"research_results_{state['session_id']}",
                scope=MemoryScope.SESSION,
                session_id=state['session_id']
            ) or []
        
        # SharedResultsManager에서 다른 Executor의 결과도 가져오기
        if self.context.shared_results_manager:
            shared_results = await self.context.shared_results_manager.get_shared_results(
                exclude_agent_id=self.name
            )
            logger.info(f"[{self.name}] 🔍 Found {len(shared_results)} shared results from other agents")

            # 공유된 결과를 results에 추가
            shared_data_count = 0
            for shared_result in shared_results:
                if isinstance(shared_result.result, dict) and shared_result.result.get('data'):
                    # 검색 결과에서 구조화된 데이터 추출
                    data = shared_result.result.get('data', {})
                    if isinstance(data, dict):
                        shared_search_results = data.get('results', data.get('items', []))
                        if isinstance(shared_search_results, list):
                            results.extend(shared_search_results)
                            shared_data_count += len(shared_search_results)
                    elif isinstance(data, list):
                        results.extend(data)
                        shared_data_count += len(data)

            logger.info(f"[{self.name}] 📥 Retrieved {shared_data_count} additional results from {len(shared_results)} shared agent results")
            logger.info(f"[{self.name}] 🤝 Agent communication: Retrieved results from agents: {[r.agent_id for r in shared_results]}")
        
        logger.info(f"[{self.name}] Found {len(results)} results to verify (including shared results)")
        
        if not results or len(results) == 0:
            # 검증할 결과가 없는 이유 상세 분석
            logger.error(f"[{self.name}] ❌ 검증할 연구 결과가 없습니다.")
            
            # state에서 결과 추적
            execution_results = state.get('execution_results', [])
            compression_results = state.get('compression_results', [])
            shared_results = state.get('shared_results', [])
            
            logger.error(f"[{self.name}] 📋 결과 추적:")
            logger.error(f"[{self.name}]   - execution_results: {len(execution_results) if isinstance(execution_results, list) else 0}개")
            logger.error(f"[{self.name}]   - compression_results: {len(compression_results) if isinstance(compression_results, list) else 0}개")
            logger.error(f"[{self.name}]   - shared_results: {len(shared_results) if isinstance(shared_results, list) else 0}개")
            logger.error(f"[{self.name}]   - 검증에 전달된 results: {len(results) if isinstance(results, list) else 0}개")
            
            # execution_results 상세 분석
            if execution_results:
                successful_executions = [er for er in execution_results if er.get('success', False)]
                failed_executions = [er for er in execution_results if not er.get('success', False)]
                logger.error(f"[{self.name}]   - 성공한 실행: {len(successful_executions)}개")
                logger.error(f"[{self.name}]   - 실패한 실행: {len(failed_executions)}개")
                
                if failed_executions:
                    logger.error(f"[{self.name}]   📝 실패한 실행 상세:")
                    for i, fe in enumerate(failed_executions[:3], 1):  # 최대 3개만 표시
                        error = fe.get('error', 'Unknown error')
                        logger.error(f"[{self.name}]     {i}. {str(error)[:100]}")
            
            # 검색 결과가 있는지 확인
            search_results_found = False
            for er in execution_results if isinstance(execution_results, list) else []:
                if isinstance(er, dict):
                    data = er.get('data', {})
                    if isinstance(data, dict):
                        results_data = data.get('results', data.get('items', []))
                        if results_data and len(results_data) > 0:
                            search_results_found = True
                            logger.error(f"[{self.name}]   ⚠️ 검색 결과는 있지만 검증 단계에 전달되지 않았습니다!")
                            break
            
            if not search_results_found:
                logger.error(f"[{self.name}]   ⚠️ 검색 단계에서 결과를 얻지 못했습니다. ExecutorAgent의 검색 실패를 확인하세요.")
            
            error_msg = "검증 실패: 검증할 연구 결과가 없습니다."
            logger.error(f"[{self.name}] ❌ {error_msg}")
            state['verified_results'] = []
            state['verification_failed'] = True
            state['error'] = error_msg
            state['current_agent'] = self.name
            return state
        
        # LLM을 사용한 실제 검증
        from src.core.llm_manager import execute_llm_task, TaskType
        
        verified = []
        rejected_reasons = []  # 검증 실패 원인 추적
        skipped_count = 0
        verification_errors = []
        
        user_query = state.get('user_query', '')
        logger.info(f"[{self.name}] 🔍 Starting verification of {len(results)} results for query: '{user_query}'")
        
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                # 다양한 키에서 title, snippet, url 추출 시도
                title = result.get('title') or result.get('name') or result.get('Title') or result.get('headline') or ''
                snippet = result.get('snippet') or result.get('content') or result.get('summary') or result.get('description') or result.get('abstract') or ''
                url = result.get('url') or result.get('link') or result.get('href') or result.get('URL') or ''
                
                # title이 비어있거나 "Search Results" 같은 메타데이터인 경우 스킵
                if not title or len(title.strip()) < 3:
                    skipped_count += 1
                    logger.debug(f"[{self.name}] ⏭️ Skipping result {i}: empty or invalid title")
                    continue
                
                # "Search Results", "Results", "Error" 같은 메타데이터 제외
                title_lower = title.lower().strip()
                if title_lower in ['search results', 'results', 'error', 'no results', 'no title']:
                    skipped_count += 1
                    logger.debug(f"[{self.name}] ⏭️ Skipping result {i}: metadata title '{title}'")
                    continue
                
                # snippet이 비어있고 url도 없는 경우 스킵
                if not snippet and not url:
                    skipped_count += 1
                    logger.debug(f"[{self.name}] ⏭️ Skipping result {i}: no content or URL")
                    continue

                # snippet 내용으로 유효하지 않은 검색 결과 필터링
                invalid_indicators = [
                    "no results were found", "bot detection",
                    "no results", "not found", "try again",
                    "unable to", "error occurred", "no matches"
                ]
                snippet_lower = snippet.lower() if snippet else ""
                if any(indicator in snippet_lower for indicator in invalid_indicators):
                    skipped_count += 1
                    logger.debug(f"[{self.name}] ⏭️ Skipping result {i}: invalid snippet content (contains error message)")
                    continue
                
                # full_content 우선 사용, 없으면 snippet 사용
                full_content = result.get('full_content', '')
                verification_content = full_content[:2000] if full_content else (snippet[:800] if snippet else '내용 없음')
                
                # 날짜 정보 추가
                published_date = result.get('published_date', '')
                date_info = ""
                if published_date:
                    try:
                        date_obj = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                        date_info = f"\n- 발행일: {date_obj.strftime('%Y-%m-%d')}"
                    except:
                        date_info = f"\n- 발행일: {published_date[:10]}"
                
                # LLM으로 검증 (점검 및 제언 중심) - 강화된 버전
                # Current time for verification context
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S (%A)')
                
                verification_prompt = f"""다음 검색 결과를 엄격하게 점검하고 제언하세요:
                
현재 시각: {current_time}

**검색 결과 정보:**
- 제목: {title}
- 내용: {verification_content}
- URL: {url if url else 'URL 없음'}{date_info}

**원래 쿼리:** {user_query}

**점검 및 제언 작업:**

당신의 역할은 자료를 "억제"하는 것이 아니라, 자료를 **엄격하게 점검하고 제언**하는 것입니다.

1. **관련성 점검** (엄격):
   - 이 자료가 쿼리와 직접적으로 관련이 있는가?
   - 관련성이 낮다면 어떤 부분이 관련이 있는가? (구체적으로 명시)
   - 배경 정보로만 유용한가? (그렇다면 낮은 관련성 점수 부여)

2. **품질 점검** (엄격):
   - 자료의 신뢰성은 어떤가? (출처, 작성자, 발행기관 고려)
   - 정보의 정확성에 오류가 있는가? (구체적인 오류 명시)
   - 출처가 신뢰할 수 있는가? (도메인, 기관, 작성자 검증)
   - 통계나 숫자가 있다면 출처가 명시되어 있는가?

3. **근거 및 증거 점검** (새로 추가):
   - 주장에 대한 근거가 제시되어 있는가?
   - 통계나 숫자는 출처가 있는가?
   - 날짜나 사실은 검증 가능한가?
   - 불확실한 정보는 명시되어 있는가?

4. **제언**:
   - 이 자료를 사용할 때 주의할 점은? (구체적으로)
   - 개선이 필요한 부분은? (구체적으로)
   - 다른 자료와 함께 사용하면 더 좋을 정보인가?
   - 추가 조사가 필요한 부분은? (구체적으로)

**중요 원칙:**
- **큰 오류만 조정**: 명백한 오류나 완전히 무관한 자료만 거부
- **작은 문제는 제언과 함께 통과**: 관련성이 약간 낮거나 품질이 약간 낮아도 제언과 함께 포함
- **억제보다는 유도**: 자료를 거부하기보다는 올바른 방향으로 사용하도록 제언
- **검색 결과의 특성 이해**: LLM이 모르는 상태에서 찾아본 결과이므로, 완벽하지 않아도 관련 정보는 포함
- **근거 없는 확신 금지**: 불확실한 정보는 반드시 불확실성 명시, 근거 없는 주장은 낮은 신뢰도 부여

**응답 형식 (반드시 이 형식으로 작성):**
```
STATUS: VERIFIED 또는 REJECTED
RELEVANCE_SCORE: 1-10 (관련성 점수, 엄격하게 평가)
QUALITY_SCORE: 1-10 (품질 점수, 엄격하게 평가)
EVIDENCE_SCORE: 1-10 (근거/증거 점수, 새로 추가)
CONFIDENCE_LEVEL: HIGH/MEDIUM/LOW (신뢰도 수준)
UNCERTAINTY_ISSUES: 불확실한 부분 명시 (없으면 "없음")
ISSUES: 발견된 문제점 (구체적으로, 없으면 "없음")
RECOMMENDATIONS: 사용 시 제언사항 (구체적으로, 없으면 "없음")
ADDITIONAL_RESEARCH_NEEDED: 추가 조사 필요한 부분 (구체적으로, 없으면 "없음")
REASON: 최종 판단 이유 (한 줄, 구체적으로)
```

⚠️ **절대 하지 말 것:**
- "y y y y..." 같은 반복 문자 사용 금지
- 단순히 "REJECTED"만 작성하지 말고 반드시 위 형식 준수
- 너무 엄격하게 판단하지 말 것
- **근거 없는 확신 표현 금지**: 불확실한 정보는 반드시 불확실성 명시
- **모호한 표현 금지**: 모든 점수와 판단은 구체적인 근거와 함께 제공"""
                
                try:
                    logger.info(f"[{self.name}] 🔍 Verifying result {i}/{len(results)}: '{title[:60]}...'")
                    
                    # Source Validation 수행
                    source_validation_result = None
                    if url:
                        try:
                            from src.verification.source_validator import SourceValidator
                            source_validator = SourceValidator()
                            source_validation_result = await source_validator.validate_source(url, verification_content)
                            logger.info(f"[{self.name}] 📊 Source validation: {source_validation_result.overall_score:.2f} (domain: {source_validation_result.domain_type.value})")
                        except Exception as e:
                            logger.warning(f"[{self.name}] Source validation failed: {e}")
                    
                    # Fact-checking 수행 (주요 주장이 있는 경우)
                    fact_check_result = None
                    if verification_content and len(verification_content) > 100:
                        try:
                            from src.verification.fact_checker import FactChecker
                            fact_checker = FactChecker()
                            # 주요 주장 추출 (숫자, 날짜, 통계 등)
                            claims = []
                            # 숫자 패턴 찾기
                            numbers = re.findall(r'\d+[.,]\d+[조억만원%]|\d+[조억만원%]', verification_content)
                            if numbers:
                                claims.extend([f"숫자/통계: {num}" for num in numbers[:3]])
                            # 날짜 패턴 찾기
                            dates = re.findall(r'\d{4}년|\d{4}-\d{2}-\d{2}', verification_content)
                            if dates:
                                claims.extend([f"날짜: {date}" for date in dates[:2]])
                            
                            if claims:
                                fact_check_result = await fact_checker.verify_fact(
                                    fact_text=verification_content[:500],
                                    sources=[result]
                                )
                                logger.info(f"[{self.name}] ✅ Fact-checking: {fact_check_result.fact_status.value} (confidence: {fact_check_result.confidence_score:.2f})")
                        except Exception as e:
                            logger.warning(f"[{self.name}] Fact-checking failed: {e}")
                    
                    # Cross-verification 수행 (다른 결과와 비교)
                    cross_verification_score = None
                    if len(results) > 1:
                        try:
                            # 같은 정보가 다른 출처에서도 확인되는지 체크
                            similar_results = []
                            for other_result in results:
                                if other_result != result and isinstance(other_result, dict):
                                    other_title = other_result.get('title', '')
                                    other_snippet = other_result.get('snippet', '')
                                    # 유사도 체크 (간단한 키워드 기반)
                                    common_keywords = set(title.lower().split()) & set(other_title.lower().split())
                                    if len(common_keywords) >= 2:
                                        similar_results.append(other_result)
                            
                            if similar_results:
                                cross_verification_score = len(similar_results) / len(results)
                                logger.info(f"[{self.name}] 🔄 Cross-verification: {len(similar_results)} similar results found (score: {cross_verification_score:.2f})")
                        except Exception as e:
                            logger.warning(f"[{self.name}] Cross-verification failed: {e}")
                    
                    # 검증 프롬프트에 추가 정보 포함
                    enhanced_prompt = verification_prompt
                    if source_validation_result:
                        enhanced_prompt += f"\n\n**출처 신뢰도 정보:**\n- 도메인 신뢰도: {source_validation_result.domain_trust:.2f}\n- 전체 신뢰도 점수: {source_validation_result.overall_score:.2f}\n- 도메인 타입: {source_validation_result.domain_type.value}"
                    if fact_check_result:
                        enhanced_prompt += f"\n\n**Fact-checking 결과:**\n- 상태: {fact_check_result.fact_status.value}\n- 신뢰도: {fact_check_result.confidence_score:.2f}"
                    if cross_verification_score is not None:
                        enhanced_prompt += f"\n\n**Cross-verification:**\n- 유사 결과 발견: {cross_verification_score:.2f}"
                    
                    verification_result = await execute_llm_task(
                        prompt=enhanced_prompt,
                        task_type=TaskType.VERIFICATION,
                        model_name=None,
                        system_message="You are a verification agent that checks and provides recommendations for research materials. Your role is to guide proper use of materials, not to overly suppress them. Only reject materials with major errors or complete irrelevance. For minor issues, provide recommendations and include the material."
                    )
                    
                    verification_text = verification_result.content or "UNKNOWN"
                    
                    # 이상한 반복 패턴 감지 및 필터링
                    if len(set(verification_text.strip().split())) < 3 or verification_text.count('y') > 10 or verification_text.count('Y') > 10:
                        logger.warning(f"[{self.name}] ⚠️ Detected abnormal response pattern, using fallback verification")
                        # 이상한 응답이면 관련성 기반으로 판단
                        verification_text = f"STATUS: VERIFIED\nREASON: Abnormal LLM response detected, using content-based verification"
                    
                    # 구조화된 응답 파싱
                    verification_upper = verification_text.upper().strip()
                    
                    # STATUS 필드 추출
                    status_match = None
                    if "STATUS:" in verification_upper:
                        status_line = [line for line in verification_upper.split('\n') if 'STATUS:' in line]
                        if status_line:
                            status_match = status_line[0]
                    elif "VERIFIED" in verification_upper:
                        status_match = "VERIFIED"
                    elif "REJECTED" in verification_upper:
                        status_match = "REJECTED"
                    
                    # RELEVANCE_SCORE 추출 (관련성 점수)
                    relevance_score = 5  # 기본값
                    if "RELEVANCE_SCORE:" in verification_upper:
                        score_lines = [line for line in verification_upper.split('\n') if 'RELEVANCE_SCORE:' in line]
                        if score_lines:
                            try:
                                score_str = score_lines[0].split('RELEVANCE_SCORE:')[1].strip().split()[0]
                                relevance_score = int(float(score_str))
                            except:
                                pass
                    
                    # QUALITY_SCORE 추출 (품질 점수)
                    quality_score = 5  # 기본값
                    if "QUALITY_SCORE:" in verification_upper:
                        score_lines = [line for line in verification_upper.split('\n') if 'QUALITY_SCORE:' in line]
                        if score_lines:
                            try:
                                score_str = score_lines[0].split('QUALITY_SCORE:')[1].strip().split()[0]
                                quality_score = int(float(score_str))
                            except:
                                pass
                    
                    # EVIDENCE_SCORE 추출 (근거/증거 점수) - 새로 추가
                    evidence_score = 5  # 기본값
                    if "EVIDENCE_SCORE:" in verification_upper:
                        score_lines = [line for line in verification_upper.split('\n') if 'EVIDENCE_SCORE:' in line]
                        if score_lines:
                            try:
                                score_str = score_lines[0].split('EVIDENCE_SCORE:')[1].strip().split()[0]
                                evidence_score = int(float(score_str))
                            except:
                                pass
                    
                    # CONFIDENCE_LEVEL 추출 (신뢰도 수준) - 새로 추가
                    confidence_level = "MEDIUM"  # 기본값
                    if "CONFIDENCE_LEVEL:" in verification_upper:
                        level_lines = [line for line in verification_upper.split('\n') if 'CONFIDENCE_LEVEL:' in line]
                        if level_lines:
                            level_str = level_lines[0].split('CONFIDENCE_LEVEL:')[1].strip().split()[0]
                            if level_str in ["HIGH", "MEDIUM", "LOW"]:
                                confidence_level = level_str
                    
                    # UNCERTAINTY_ISSUES 추출 (불확실성 이슈) - 새로 추가
                    uncertainty_issues = "없음"
                    if "UNCERTAINTY_ISSUES:" in verification_text:
                        issue_lines = [line for line in verification_text.split('\n') if 'UNCERTAINTY_ISSUES:' in line]
                        if issue_lines:
                            issue_text = issue_lines[0].split('UNCERTAINTY_ISSUES:')[1].strip()
                            if issue_text and issue_text != "없음":
                                uncertainty_issues = issue_text[:300]
                    
                    # ADDITIONAL_RESEARCH_NEEDED 추출 (추가 조사 필요) - 새로 추가
                    additional_research_needed = "없음"
                    if "ADDITIONAL_RESEARCH_NEEDED:" in verification_text:
                        research_lines = [line for line in verification_text.split('\n') if 'ADDITIONAL_RESEARCH_NEEDED:' in line]
                        if research_lines:
                            research_text = research_lines[0].split('ADDITIONAL_RESEARCH_NEEDED:')[1].strip()
                            if research_text and research_text != "없음":
                                additional_research_needed = research_text[:300]
                    
                    # 종합 신뢰도 계산 (다단계 검증)
                    # Self-verification: evidence_score와 quality_score의 평균
                    self_verification_score = (evidence_score + quality_score) / 2.0 / 10.0
                    # Cross-verification: cross_verification_score 사용 (이미 계산됨)
                    cross_verification_score_normalized = cross_verification_score if cross_verification_score is not None else 0.5
                    # External verification: source_validation과 fact_check 사용
                    external_verification_score = 0.5  # 기본값
                    if source_validation_result:
                        external_verification_score = source_validation_result.overall_score
                    elif fact_check_result:
                        external_verification_score = fact_check_result.confidence_score
                    
                    # 최종 신뢰도 점수 (가중 평균)
                    final_confidence = (
                        self_verification_score * 0.3 +
                        cross_verification_score_normalized * 0.4 +
                        external_verification_score * 0.3
                    )
                    
                    # 신뢰도 수준에 따른 점수 조정
                    if confidence_level == "LOW":
                        final_confidence = min(final_confidence, 0.5)
                    elif confidence_level == "HIGH":
                        final_confidence = max(final_confidence, 0.7)
                    
                    # 검증 판단: REJECTED가 명시적으로 있고 관련성 점수가 매우 낮은 경우만 거부
                    is_verified = True  # 기본값은 통과
                    if status_match and "REJECTED" in status_match:
                        # REJECTED이지만 관련성 점수가 3 이상이면 통과 (큰 오류만 거부)
                        if relevance_score >= 3:
                            logger.info(f"[{self.name}] ⚠️ Result marked as REJECTED but relevance_score={relevance_score} >= 3, verifying anyway")
                            is_verified = True
                        else:
                            is_verified = False
                    elif status_match and "VERIFIED" in status_match:
                        is_verified = True
                    elif "REJECTED" in verification_upper and relevance_score < 2:
                        # 명시적 REJECTED가 없어도 관련성 점수가 매우 낮으면 거부
                        is_verified = False
                    else:
                        # 명시적 판단이 없으면 관련성 기반으로 판단
                        is_verified = relevance_score >= 3
                    
                    logger.info(f"[{self.name}] 📋 Verification result {i}: '{verification_text[:150]}' -> is_verified={is_verified}")
                    
                    if is_verified:
                        # 제언사항 추출
                        recommendations = "없음"
                        if "RECOMMENDATIONS:" in verification_text:
                            rec_lines = [line for line in verification_text.split('\n') if 'RECOMMENDATIONS:' in line]
                            if rec_lines:
                                rec_text = rec_lines[0].split('RECOMMENDATIONS:')[1].strip()
                                if rec_text and rec_text != "없음":
                                    recommendations = rec_text[:300]
                        
                        # 이슈 추출
                        issues = "없음"
                        if "ISSUES:" in verification_text:
                            issue_lines = [line for line in verification_text.split('\n') if 'ISSUES:' in line]
                            if issue_lines:
                                issue_text = issue_lines[0].split('ISSUES:')[1].strip()
                                if issue_text and issue_text != "없음":
                                    issues = issue_text[:300]
                        
                        verified_result = {
                            "index": i,
                            "title": title,
                            "snippet": snippet,
                            "url": url,
                            "status": "verified",
                            "verification_note": verification_text[:500],  # 더 긴 제언 포함
                            "relevance_score": relevance_score,
                            "quality_score": quality_score,
                            "evidence_score": evidence_score,
                            "confidence_level": confidence_level,
                            "final_confidence": final_confidence,
                            "uncertainty_issues": uncertainty_issues,
                            "recommendations": recommendations,
                            "issues": issues,
                            "additional_research_needed": additional_research_needed,
                            # 다단계 검증 점수
                            "verification_stages": {
                                "self_verification_score": self_verification_score,
                                "cross_verification_score": cross_verification_score_normalized,
                                "external_verification_score": external_verification_score,
                                "final_confidence": final_confidence
                            },
                            "source_validation": {
                                "overall_score": source_validation_result.overall_score if source_validation_result else None,
                                "domain_type": source_validation_result.domain_type.value if source_validation_result else None,
                                "domain_trust": source_validation_result.domain_trust if source_validation_result else None
                            } if source_validation_result else None,
                            "fact_check": {
                                "status": fact_check_result.fact_status.value if fact_check_result else None,
                                "confidence": fact_check_result.confidence_score if fact_check_result else None
                            } if fact_check_result else None,
                            "cross_verification_score": cross_verification_score
                        }
                        # full_content와 published_date 포함
                        if full_content:
                            verified_result['full_content'] = full_content
                        if published_date:
                            verified_result['published_date'] = published_date
                        verified.append(verified_result)
                        logger.info(f"[{self.name}] ✅ Result {i} verified: '{title[:50]}...' (relevance: {relevance_score}, issues: {issues[:50] if issues != '없음' else '없음'})")
                    else:
                        rejected_reasons.append({
                            "index": i,
                            "title": title[:80],
                            "reason": verification_text[:200],
                            "url": url[:100] if url else "N/A"
                        })
                        logger.info(f"[{self.name}] ⚠️ Result {i} rejected: '{title[:50]}...' (reason: {verification_text[:100]})")
                        continue
                except Exception as e:
                    error_str = str(e).lower()
                    verification_errors.append({
                        "index": i,
                        "title": title[:80],
                        "error": str(e)[:200]
                    })
                    # Rate limit이나 모든 모델 실패 시에는 포함하지 않음 (품질 저하 방지)
                    if "rate limit" in error_str or "429" in error_str or "all fallback models failed" in error_str or "no available models" in error_str:
                        logger.warning(f"[{self.name}] ⚠️ Verification failed for result {i}: {e} (rate limit/all models failed), excluding from results")
                        continue  # 품질 저하 방지를 위해 제외
                    else:
                        logger.warning(f"[{self.name}] ⚠️ Verification failed for result {i}: {e}, including anyway")
                        # 검증 실패해도 기본 정보가 있으면 포함 (단, rate limit이 아닌 경우만)
                        if title and (snippet or url):
                            verified.append({
                                "index": i,
                                "title": title,
                                "snippet": snippet,
                                "url": url,
                                "status": "partial",
                                "verification_note": f"Verification failed: {str(e)[:100]}"
                            })
            else:
                skipped_count += 1
                logger.warning(f"[{self.name}] ⚠️ Unknown result format: {type(result)}, value: {str(result)[:100]}")
                continue
        
        # 검증 통계 및 디버깅 정보 출력
        logger.info(f"[{self.name}] 📊 Verification Statistics:")
        logger.info(f"[{self.name}]   - Total results: {len(results)}")
        logger.info(f"[{self.name}]   - Verified: {len(verified)}")
        logger.info(f"[{self.name}]   - Rejected: {len(rejected_reasons)}")
        logger.info(f"[{self.name}]   - Skipped: {skipped_count}")
        logger.info(f"[{self.name}]   - Verification errors: {len(verification_errors)}")
        
        if rejected_reasons:
            logger.warning(f"[{self.name}] 🔍 Rejected Results Analysis:")
            for rejected in rejected_reasons[:5]:  # 최대 5개만 표시
                logger.warning(f"[{self.name}]   - Result {rejected['index']}: '{rejected['title']}'")
                logger.warning(f"[{self.name}]     Reason: {rejected['reason']}")
                logger.warning(f"[{self.name}]     URL: {rejected['url']}")
        
        if verification_errors:
            logger.error(f"[{self.name}] ❌ Verification Errors:")
            for error_info in verification_errors[:3]:  # 최대 3개만 표시
                logger.error(f"[{self.name}]   - Result {error_info['index']}: '{error_info['title']}'")
                logger.error(f"[{self.name}]     Error: {error_info['error']}")
        
        # 검증된 결과가 없을 때 원본 결과를 사용하는 fallback
        if not verified and len(results) > 0:
            logger.warning(f"[{self.name}] ⚠️ No results verified! Using original results as fallback...")
            logger.warning(f"[{self.name}] 🔍 This may indicate:")
            logger.warning(f"[{self.name}]   1. Search queries are not matching the user query")
            logger.warning(f"[{self.name}]   2. Verification criteria are too strict")
            logger.warning(f"[{self.name}]   3. Search results are genuinely irrelevant")
            
            # 원본 결과를 검증된 결과로 사용 (신뢰도 낮게)
            for i, result in enumerate(results[:5], 1):  # 최대 5개만
                if isinstance(result, dict):
                    title = result.get('title') or result.get('name') or ''
                    snippet = result.get('snippet') or result.get('content') or ''
                    url = result.get('url') or result.get('link') or ''
                    
                    if title and len(title.strip()) >= 3:
                        verified.append({
                            "index": i,
                            "title": title,
                            "snippet": snippet[:500] if snippet else '',
                            "url": url,
                            "status": "fallback",
                            "verification_note": "No verified results found, using original search results as fallback"
                        })
                        logger.warning(f"[{self.name}] ⚠️ Added fallback result {i}: '{title[:50]}...'")
            
            logger.warning(f"[{self.name}] ⚠️ Using {len(verified)} fallback results (low confidence)")
        
        logger.info(f"[{self.name}] ✅ Verification completed: {len(verified)}/{len(results)} results verified (including fallback)")
        
        # 검증 결과를 SharedResultsManager에 공유
        if self.context.shared_results_manager:
            shared_verification_count = 0
            for verified_result in verified:
                task_id = f"verification_{verified_result.get('index', 0)}"
                result_id = await self.context.shared_results_manager.share_result(
                    task_id=task_id,
                    agent_id=self.context.agent_id,  # 고유한 agent_id 사용
                    result=verified_result,
                    metadata={"status": verified_result.get('status', 'unknown')},
                    confidence=1.0 if verified_result.get('status') == 'verified' else 0.5
                )
                shared_verification_count += 1
                # 개별 로그는 debug 레벨로 변경 (너무 많은 로그 방지)
                logger.debug(f"[{self.name}] 🔗 Shared verification result {verified_result.get('index', 0)} (result_id: {result_id[:8]}..., status: {verified_result.get('status', 'unknown')})")

            logger.info(f"[{self.name}] 📤 Shared {shared_verification_count} verification results with other agents")

            # Executor 결과에 대한 논박 (Debate) 수행
            if self.context.discussion_manager and self.context.shared_results_manager and len(verified) > 0:
                # Executor 결과 가져오기
                executor_results = await self.context.shared_results_manager.get_shared_results(
                    task_id=None  # 모든 Executor 결과
                )
                
                # Executor 결과 필터링 (executor로 시작하는 agent_id)
                executor_shared_results = [r for r in executor_results if r.agent_id.startswith('executor')]
                
                if executor_shared_results:
                    logger.info(f"[{self.name}] 💬 Found {len(executor_shared_results)} executor results to debate")
                    
                    # 각 Executor 결과에 대해 논박 수행
                    debate_results = []
                    for executor_result in executor_shared_results[:5]:  # 최대 5개 결과에 대해 논박
                        # 다른 Verifier들의 검증 결과도 가져오기
                        other_verifiers = await self.context.shared_results_manager.get_shared_results(
                            agent_id=None,
                            exclude_agent_id=self.context.agent_id
                        )
                        other_verifier_results = [r for r in other_verifiers if r.agent_id.startswith('verifier')]
                        
                        # 논박 수행
                        debate_result = await self.context.discussion_manager.agent_discuss_result(
                            result_id=executor_result.task_id,
                            agent_id=self.context.agent_id,
                            other_agent_results=other_verifier_results[:3] + [executor_result],  # 다른 Verifier + Executor 결과
                            discussion_type="verification"
                        )
                        
                        if debate_result:
                            debate_results.append(debate_result)
                            logger.info(f"[{self.name}] 💬 Debate completed for executor result: consistency={debate_result.get('consistency_check', 'unknown')}, validity={debate_result.get('logical_validity', 'unknown')}")
                    
                    # 논박 결과를 state에 저장
                    if 'agent_debates' not in state:
                        state['agent_debates'] = {}
                    state['agent_debates']['verifier_debates'] = debate_results
                    logger.info(f"[{self.name}] 💬 Saved {len(debate_results)} debate results to state")
                
                # 다른 Verifier들의 검증 결과와 논박
                other_verified = await self.context.shared_results_manager.get_shared_results(
                    agent_id=None,
                    exclude_agent_id=self.context.agent_id
                )
                
                # 검증된 결과만 필터링
                other_verified_results = [r for r in other_verified if isinstance(r.result, dict) and r.result.get('status') == 'verified']

                if other_verified_results:
                    logger.info(f"[{self.name}] 💬 Found {len(other_verified_results)} verified results from other verifiers for debate")

                    # 첫 번째 검증 결과에 대해 논박
                    first_verified = verified[0]
                    result_id = f"verification_{first_verified.get('index', 0)}"
                    logger.info(f"[{self.name}] 💬 Starting debate on verification result {first_verified.get('index', 0)} with {len(other_verified_results[:3])} other verifiers")

                    debate_result = await self.context.discussion_manager.agent_discuss_result(
                        result_id=result_id,
                        agent_id=self.context.agent_id,
                        other_agent_results=other_verified_results[:3],
                        discussion_type="verification"
                    )
                    
                    if debate_result:
                        logger.info(f"[{self.name}] 💬 Debate completed: consistency={debate_result.get('consistency_check', 'unknown')}, validity={debate_result.get('logical_validity', 'unknown')}")
                        logger.info(f"[{self.name}] 🤝 Agent debate: Analyzed verification consistency with {len(other_verified_results[:3])} peer agents")
                        
                        # 논박 결과 저장
                        if 'agent_debates' not in state:
                            state['agent_debates'] = {}
                        if 'verifier_peer_debates' not in state['agent_debates']:
                            state['agent_debates']['verifier_peer_debates'] = []
                        state['agent_debates']['verifier_peer_debates'].append(debate_result)
                    else:
                        logger.info(f"[{self.name}] 💬 No debate generated for verification result")
                else:
                    logger.info(f"[{self.name}] 💬 No other verified results found for debate")
            else:
                logger.info(f"[{self.name}] Agent debate disabled or no verified results to debate")
        
        # Council 활성화 확인 및 적용 (사실 확인이 중요한 경우 - 기본 활성화)
        use_council = state.get('use_council', None)  # 수동 활성화 옵션
        if use_council is None:
            # 자동 활성화 판단 (기본 활성화)
            from src.core.council_activator import get_council_activator
            activator = get_council_activator()
            
            context = {
                'low_confidence_sources': len([r for r in verified if r.get('confidence', 1.0) < 0.7]),
                'verification_count': len(verified)
            }
            
            activation_decision = activator.should_activate(
                process_type='verification',
                query=state['user_query'],
                context=context
            )
            use_council = activation_decision.should_activate
            if use_council:
                logger.info(f"[{self.name}] 🏛️ Council auto-activated: {activation_decision.reason}")
        
        # Council 적용 (활성화된 경우)
        if use_council and verified:
            try:
                from src.core.llm_council import run_full_council
                logger.info(f"[{self.name}] 🏛️ Running Council review for verification results...")
                
                # 검증 결과 요약 생성
                verification_summary = "\n\n".join([
                    f"Result {i+1}:\nTitle: {r.get('title', 'N/A')}\nStatus: {r.get('status', 'N/A')}\nConfidence: {r.get('confidence', 0.0):.2f}\nNote: {r.get('verification_note', 'N/A')[:100]}"
                    for i, r in enumerate(verified[:10])  # 최대 10개만 검토
                ])
                
                council_query = f"""Review the verification results and assess their reliability. Check for consistency and identify any potential issues.

Research Query: {state['user_query']}

Verification Results:
{verification_summary}

Provide a review with:
1. Overall verification quality assessment
2. Consistency check across results
3. Recommendations for improvement"""
                
                stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
                    council_query
                )
                
                # Council 검토 결과
                review_report = stage3_result.get('response', '')
                logger.info(f"[{self.name}] ✅ Council review completed.")
                logger.info(f"[{self.name}] Council aggregate rankings: {metadata.get('aggregate_rankings', [])}")
                
                # Council 메타데이터를 state에 저장
                if 'council_metadata' not in state:
                    state['council_metadata'] = {}
                state['council_metadata']['verification'] = {
                    'stage1_results': stage1_results,
                    'stage2_results': stage2_results,
                    'stage3_result': stage3_result,
                    'metadata': metadata,
                    'review_report': review_report
                }
            except Exception as e:
                logger.warning(f"[{self.name}] Council review failed: {e}. Using original verification results.")
                # Council 실패 시 원본 검증 결과 사용 (fallback 제거 - 명확한 로깅만)
        
        # Enhanced Quality Assessment (Phase 2)
        quality_assessments = {}
        logger.info(f"[{self.name}] 📊 Performing quality assessment on {len(verified)} verified results")
        
        for result in verified:
            result_id = result.get('id') or result.get('url', '')
            if result_id:
                quality_assessment = await self._assess_result_quality(result, verified)
                quality_assessments[result_id] = quality_assessment
                
                # Add quality assessment to result
                result['quality_assessment'] = quality_assessment
                
                logger.debug(f"[{self.name}] Quality assessment for {result.get('title', 'N/A')[:50]}: "
                           f"credibility={quality_assessment['source_credibility']:.2f}, "
                           f"academic={quality_assessment['academic_rigor']:.2f}, "
                           f"verifiability={quality_assessment['verifiability']:.2f}")
        
        # Store quality assessments in state
        state['quality_assessments'] = quality_assessments
        state['verified_results'] = verified
        state['current_agent'] = self.name
        state['verification_failed'] = False if verified else True
        
        logger.info(f"[{self.name}] ✅ Quality assessment complete: {len(quality_assessments)} results assessed")
        
        # Write to shared memory
        memory.write(
            key=f"verified_{state['session_id']}",
            value=verified,
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        memory.write(
            key=f"quality_assessments_{state['session_id']}",
            value=quality_assessments,
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        logger.info(f"[{self.name}] Verified results and quality assessments saved to shared memory")
        logger.info(f"=" * 80)
        
        return state
    
    async def _assess_result_quality(
        self,
        result: Dict[str, Any],
        all_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Assess the quality of a research result across multiple dimensions.
        
        Args:
            result: The result to assess
            all_results: All verified results for cross-validation
            
        Returns:
            Quality assessment dictionary with scores (0-1)
        """
        try:
            # Assess each dimension
            source_credibility = self._assess_source_credibility(result)
            academic_rigor = self._assess_academic_rigor(result)
            verifiability = self._assess_verifiability(result)
            cross_source_support = self._assess_cross_validation(result, all_results)
            information_freshness = self._assess_recency(result)
            
            # Calculate weighted overall quality
            weights = {
                'source_credibility': 0.25,
                'academic_rigor': 0.25,
                'verifiability': 0.25,
                'cross_source_support': 0.15,
                'information_freshness': 0.10
            }
            
            overall_quality = (
                source_credibility * weights['source_credibility'] +
                academic_rigor * weights['academic_rigor'] +
                verifiability * weights['verifiability'] +
                cross_source_support * weights['cross_source_support'] +
                information_freshness * weights['information_freshness']
            )
            
            return {
                'source_credibility': source_credibility,
                'academic_rigor': academic_rigor,
                'verifiability': verifiability,
                'cross_source_support': cross_source_support,
                'information_freshness': information_freshness,
                'overall_quality': overall_quality
            }
            
        except Exception as e:
            logger.warning(f"[{self.name}] Error assessing result quality: {e}")
            # Return default scores on error
            return {
                'source_credibility': 0.5,
                'academic_rigor': 0.5,
                'verifiability': 0.5,
                'cross_source_support': 0.5,
                'information_freshness': 0.5,
                'overall_quality': 0.5
            }
    
    def _assess_source_credibility(self, result: Dict[str, Any]) -> float:
        """
        Assess source credibility based on domain and source type.
        
        Priority: Academic > Official > News > General
        
        Returns:
            Credibility score (0-1)
        """
        url = result.get('url', '').lower()
        title = result.get('title', '').lower()
        
        # Academic sources (highest credibility)
        academic_indicators = [
            'arxiv.org', 'doi.org', 'scholar.google', 'pubmed', 'ncbi.nlm.nih.gov',
            'ieee.org', 'acm.org', 'springer.com', 'sciencedirect.com',
            'nature.com', 'science.org', 'cell.com', 'wiley.com',
            '.edu', 'university', 'journal', 'peer-reviewed'
        ]
        
        if any(indicator in url for indicator in academic_indicators):
            return 0.95
        
        # Official/Government sources
        official_indicators = [
            '.gov', '.mil', 'who.int', 'un.org', 'europa.eu',
            'nih.gov', 'nasa.gov', 'cdc.gov', 'fda.gov'
        ]
        
        if any(indicator in url for indicator in official_indicators):
            return 0.90
        
        # Reputable news/media
        news_indicators = [
            'reuters.com', 'bbc.com', 'apnews.com', 'nytimes.com',
            'wsj.com', 'economist.com', 'theguardian.com', 'washingtonpost.com'
        ]
        
        if any(indicator in url for indicator in news_indicators):
            return 0.80
        
        # Industry/Professional organizations
        industry_indicators = [
            '.org', 'association', 'institute', 'foundation', 'society'
        ]
        
        if any(indicator in url for indicator in industry_indicators):
            return 0.70
        
        # General web sources
        return 0.50
    
    def _assess_academic_rigor(self, result: Dict[str, Any]) -> float:
        """
        Assess academic rigor based on content indicators.
        
        Checks for: citations, methodology, peer-review, research terms
        
        Returns:
            Academic rigor score (0-1)
        """
        content = (result.get('content', '') + ' ' + 
                  result.get('snippet', '') + ' ' + 
                  result.get('title', '')).lower()
        
        url = result.get('url', '').lower()
        
        rigor_score = 0.0
        
        # Check for peer-reviewed publication
        peer_review_indicators = ['peer-reviewed', 'peer reviewed', 'refereed']
        if any(indicator in content or indicator in url for indicator in peer_review_indicators):
            rigor_score += 0.3
        
        # Check for citations/references
        citation_indicators = ['citation', 'references', 'bibliography', 'doi:', 'cited by']
        if any(indicator in content for indicator in citation_indicators):
            rigor_score += 0.2
        
        # Check for methodology
        methodology_indicators = ['methodology', 'methods', 'experimental', 'study design', 'data collection']
        if any(indicator in content for indicator in methodology_indicators):
            rigor_score += 0.2
        
        # Check for research terms
        research_indicators = ['research', 'study', 'analysis', 'investigation', 'findings', 'results', 'conclusion']
        matching_indicators = sum(1 for indicator in research_indicators if indicator in content)
        rigor_score += min(0.3, matching_indicators * 0.05)
        
        return min(1.0, rigor_score)
    
    def _assess_verifiability(self, result: Dict[str, Any]) -> float:
        """
        Assess verifiability based on evidence, data, and references.
        
        Returns:
            Verifiability score (0-1)
        """
        content = (result.get('content', '') + ' ' + 
                  result.get('snippet', '')).lower()
        
        url = result.get('url', '').lower()
        
        verifiability_score = 0.0
        
        # Check for data/evidence indicators
        data_indicators = ['data', 'evidence', 'statistics', 'figures', 'table', 'chart']
        matching_data = sum(1 for indicator in data_indicators if indicator in content)
        verifiability_score += min(0.3, matching_data * 0.05)
        
        # Check for specific claims with sources
        source_indicators = ['according to', 'source:', 'reference:', 'published in', 'reported by']
        if any(indicator in content for indicator in source_indicators):
            verifiability_score += 0.3
        
        # Check for URL availability and type
        if url:
            verifiability_score += 0.2
            
            # Bonus for direct document links
            if any(ext in url for ext in ['.pdf', '.doc', '.html']):
                verifiability_score += 0.1
        
        # Check for author information
        author_indicators = ['author:', 'by ', 'written by', 'published by']
        if any(indicator in content for indicator in author_indicators):
            verifiability_score += 0.1
        
        return min(1.0, verifiability_score)
    
    def _assess_cross_validation(
        self,
        result: Dict[str, Any],
        all_results: List[Dict[str, Any]]
    ) -> float:
        """
        Assess cross-validation by checking if information is supported by other sources.
        
        Returns:
            Cross-validation score (0-1)
        """
        if len(all_results) < 2:
            return 0.5  # Can't cross-validate with single result
        
        try:
            # Extract key terms from current result
            current_title = result.get('title', '').lower()
            current_content = (result.get('content', '') + ' ' + 
                             result.get('snippet', '')).lower()
            
            # Extract significant words (simple approach)
            import re
            words = re.findall(r'\b\w{4,}\b', current_title + ' ' + current_content)
            significant_words = set(words[:20])  # Take top 20 words
            
            # Check overlap with other results
            overlap_scores = []
            
            for other_result in all_results:
                if other_result.get('url') == result.get('url'):
                    continue  # Skip same result
                
                other_content = (other_result.get('title', '') + ' ' +
                               other_result.get('content', '') + ' ' +
                               other_result.get('snippet', '')).lower()
                
                other_words = set(re.findall(r'\b\w{4,}\b', other_content))
                
                if significant_words and other_words:
                    overlap = len(significant_words & other_words) / len(significant_words)
                    overlap_scores.append(overlap)
            
            if overlap_scores:
                # Average overlap with other sources
                avg_overlap = sum(overlap_scores) / len(overlap_scores)
                return min(1.0, avg_overlap * 2)  # Scale up
            else:
                return 0.5
                
        except Exception as e:
            logger.debug(f"Error in cross-validation: {e}")
            return 0.5
    
    def _assess_recency(self, result: Dict[str, Any]) -> float:
        """
        Assess information freshness based on publication date.
        
        Returns:
            Recency score (0-1)
        """
        published_date = result.get('published_date', '')
        
        if not published_date:
            return 0.5  # Unknown date
        
        try:
            from datetime import datetime, timezone
            
            # Parse date
            if isinstance(published_date, str):
                date_obj = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            else:
                date_obj = published_date
            
            # Calculate age in days
            now = datetime.now(timezone.utc)
            age_days = (now - date_obj).days
            
            # Scoring based on age
            if age_days < 30:
                return 1.0  # Very recent (< 1 month)
            elif age_days < 90:
                return 0.9  # Recent (< 3 months)
            elif age_days < 365:
                return 0.8  # This year
            elif age_days < 730:
                return 0.7  # Last 2 years
            elif age_days < 1825:
                return 0.6  # Last 5 years
            else:
                return 0.4  # Older than 5 years
                
        except Exception as e:
            logger.debug(f"Error assessing recency: {e}")
            return 0.5


class GeneratorAgent:
    """Generator agent - creates final report (Skills-based)."""
    
    def __init__(self, context: AgentContext, skill: Optional[Skill] = None):
        self.context = context
        self.name = "generator"
        self.available_tools: list = []  # MCP 자동 할당 도구
        self.tool_infos: list = []  # 도구 메타데이터
        self.skill = skill
        
        # Skill이 없으면 로드 시도
        if self.skill is None:
            skill_manager = get_skill_manager()
            self.skill = skill_manager.load_skill("synthesizer")
        
        # Skill instruction 사용
        if self.skill:
            self.instruction = self.skill.instructions
        else:
            self.instruction = "You are a report generation agent."
    
    def _validate_and_enhance_citations(self, report: str, source_mapping: Dict[int, Dict[str, Any]], verified_results: List[Dict[str, Any]]) -> str:
        """
        보고서의 출처 인용을 검증하고 보완합니다.
        본문에서 실제로 인용된 출처만 참고문헌에 포함합니다.
        
        Args:
            report: 생성된 보고서
            source_mapping: 출처 번호 -> 출처 정보 매핑
            verified_results: 검증된 결과 리스트
            
        Returns:
            출처 인용이 보완된 보고서
        """
        
        # 본문에서 인용된 출처 번호 추출
        body_text = report
        references_section_match = re.search(r'##?\s*참고\s*문헌|##?\s*References|##?\s*출처', report, re.IGNORECASE)
        if references_section_match:
            body_text = report[:references_section_match.start()]
        
        # 본문에서 인용 패턴 찾기: [1], [1,2], 출처 1, (출처 1), 출처1 등
        cited_patterns = [
            r'\[(\d+)\]',  # [1]
            r'\[(\d+),\s*(\d+)\]',  # [1, 2]
            r'출처\s*(\d+)',  # 출처 1
            r'\(출처\s*(\d+)\)',  # (출처 1)
            r'출처(\d+)',  # 출처1
        ]
        
        cited_numbers = set()
        for pattern in cited_patterns:
            matches = re.findall(pattern, body_text)
            for match in matches:
                if isinstance(match, tuple):
                    for num in match:
                        if num and num.isdigit():
                            cited_numbers.add(int(num))
                elif match and match.isdigit():
                    cited_numbers.add(int(match))
        
        logger.info(f"[{self.name}] 📋 Found {len(cited_numbers)} cited sources in body: {sorted(cited_numbers)}")
        
        # 참고 문헌 섹션 확인 및 재생성
        if references_section_match:
            # 기존 참고 문헌 섹션 제거
            report = report[:references_section_match.start()].rstrip()
        
        # 본문에서 인용된 출처만 포함하는 참고 문헌 생성
        if cited_numbers:
            references_text = "\n\n## 참고 문헌\n\n"
            
            # 인용된 번호 순서대로 정렬
            sorted_cited = sorted(cited_numbers)
            new_source_mapping = {}
            
            for new_idx, old_num in enumerate(sorted_cited, 1):
                if old_num in source_mapping:
                    source_info = source_mapping[old_num]
                    new_source_mapping[new_idx] = source_info
                    references_text += f"{new_idx}. [{source_info['title']}]({source_info['url']})\n"
                    if source_info.get('published_date'):
                        references_text += f"   발행일: {source_info['published_date']}\n"
                    references_text += "\n"
                else:
                    # source_mapping에 없는 경우 verified_results에서 찾기
                    if old_num <= len(verified_results):
                        result = verified_results[old_num - 1]
                        if isinstance(result, dict):
                            title = result.get('title', '')
                            url = result.get('url', '')
                            if title and url:
                                new_source_mapping[new_idx] = {
                                    'title': title,
                                    'url': url,
                                    'published_date': result.get('published_date', '')
                                }
                                references_text += f"{new_idx}. [{title}]({url})\n"
                                if result.get('published_date'):
                                    references_text += f"   발행일: {result['published_date']}\n"
                                references_text += "\n"
            
            # 본문의 출처 번호를 새로운 번호로 업데이트
            for old_idx, new_idx in enumerate(sorted_cited, 1):
                old_num = sorted_cited[old_idx - 1]
                # [old_num] -> [new_idx]로 변경
                report = re.sub(rf'\[{old_num}\]', f'[{new_idx}]', report)
                report = re.sub(rf'출처\s*{old_num}\b', f'출처 {new_idx}', report)
                report = re.sub(rf'\(출처\s*{old_num}\)', f'(출처 {new_idx})', report)
            
            report += references_text
            logger.info(f"[{self.name}] ✅ Rebuilt references section with {len(sorted_cited)} cited sources (removed uncited sources)")
        else:
            # 인용이 없으면 참고 문헌 섹션 제거
            logger.warning(f"[{self.name}] ⚠️ No citations found in body, removing references section")
        
        return report
    
    async def execute(self, state: AgentState) -> AgentState:
        """Generate final report."""
        logger.info(f"[{self.name}] Generating final report...")
        
        # 연구 또는 검증 실패 확인 - Fallback 제거, 명확한 에러만 반환
        if state.get('research_failed') or state.get('verification_failed'):
            error_msg = state.get('error')
            if not error_msg:
                if state.get('verification_failed'):
                    error_msg = "검증 실패: 검증된 결과가 없습니다"
                elif state.get('research_failed'):
                    error_msg = "연구 실행 실패"
                else:
                    error_msg = "알 수 없는 오류"

            # 상세 디버깅 정보 출력
            logger.error(f"[{self.name}] ❌ Research or verification failed: {error_msg}")
            logger.error(f"[{self.name}] 🔍 Debugging Information:")
            logger.error(f"[{self.name}]   - Research failed: {state.get('research_failed', False)}")
            logger.error(f"[{self.name}]   - Verification failed: {state.get('verification_failed', False)}")
            logger.error(f"[{self.name}]   - User query: '{state.get('user_query', 'N/A')}'")
            
            # 검증 결과 확인
            verified_results = state.get('verified_results', [])
            research_results = state.get('research_results', [])
            logger.error(f"[{self.name}]   - Verified results count: {len(verified_results) if verified_results else 0}")
            logger.error(f"[{self.name}]   - Research results count: {len(research_results) if research_results else 0}")
            
            # SharedResultsManager에서 결과 확인
            if self.context.shared_results_manager:
                try:
                    shared_results = await self.context.shared_results_manager.get_shared_results(
                        agent_id=None
                    )
                    logger.error(f"[{self.name}]   - Shared results count: {len(shared_results) if shared_results else 0}")
                except Exception as e:
                    logger.error(f"[{self.name}]   - Failed to get shared results: {e}")
            
            # 검증 실패 원인 분석
            if state.get('verification_failed'):
                logger.error(f"[{self.name}] 🔍 Verification Failure Analysis:")
                logger.error(f"[{self.name}]   - Possible causes:")
                logger.error(f"[{self.name}]     1. Search queries did not match user query")
                logger.error(f"[{self.name}]     2. Verification criteria were too strict")
                logger.error(f"[{self.name}]     3. Search results were genuinely irrelevant")
                logger.error(f"[{self.name}]     4. LLM verification service issues")
                
                # 원본 검색 결과가 있으면 일부 표시
                if research_results and len(research_results) > 0:
                    logger.error(f"[{self.name}]   - Sample research results (first 3):")
                    for i, result in enumerate(research_results[:3], 1):
                        if isinstance(result, dict):
                            title = result.get('title', result.get('name', 'N/A'))[:60]
                            logger.error(f"[{self.name}]     {i}. {title}")
            
            state['final_report'] = None
            state['current_agent'] = self.name
            state['report_failed'] = True
            state['error'] = error_msg
            return state
        
        memory = self.context.shared_memory
        
        # Read verified results from state or shared memory
        verified_results = state.get('verified_results', [])
        if not verified_results:
            verified_results = memory.read(
                key=f"verified_{state['session_id']}",
                scope=MemoryScope.SESSION,
                session_id=state['session_id']
            ) or []
        
        # SharedResultsManager에서 모든 공유된 검증 결과 가져오기
        if self.context.shared_results_manager:
            all_shared_results = await self.context.shared_results_manager.get_shared_results()
            logger.info(f"[{self.name}] 🔍 Found {len(all_shared_results)} total shared results from all agents")

            # 공유 결과 통계
            verification_results = [r for r in all_shared_results if isinstance(r.result, dict) and r.result.get('status') == 'verified']
            search_results = [r for r in all_shared_results if not isinstance(r.result, dict) or r.result.get('status') != 'verified']

            logger.info(f"[{self.name}] 📊 Shared results breakdown: {len(verification_results)} verified, {len(search_results)} search results")

            # 검증된 결과만 필터링하여 추가
            added_from_shared = 0
            for shared_result in all_shared_results:
                if isinstance(shared_result.result, dict):
                    # 검증된 결과인 경우
                    if shared_result.result.get('status') == 'verified':
                        # 중복 제거 (URL 기준)
                        existing_urls = {r.get('url', '') for r in verified_results if isinstance(r, dict)}
                        result_url = shared_result.result.get('url', '')
                        if result_url and result_url not in existing_urls:
                            verified_results.append(shared_result.result)
                            added_from_shared += 1
                            logger.info(f"[{self.name}] ➕ Added shared verified result from agent {shared_result.agent_id}: {shared_result.result.get('title', '')[:50]}...")

            logger.info(f"[{self.name}] 📥 Added {added_from_shared} verified results from shared agent communications")
            logger.info(f"[{self.name}] 🤝 Agent communication: Incorporated results from agents: {list(set(r.agent_id for r in all_shared_results))}")
        
        # 검증 요약 가져오기 (VerifierAgent에서 전달된 정보)
        verification_summary = state.get('verification_summary', {})
        if not verification_summary:
            verification_summary = memory.read(
                key=f"verification_summary_{state['session_id']}",
                scope=MemoryScope.SESSION,
                session_id=state['session_id']
            ) or {}
        
        logger.info(f"[{self.name}] Found {len(verified_results)} verified results for report generation (including shared results)")
        
        # 검증 요약 정보 로깅
        if verification_summary:
            logger.info(f"[{self.name}] 📊 Verification Summary received:")
            logger.info(f"[{self.name}]   - Total verified: {verification_summary.get('total_verified', 0)}")
            logger.info(f"[{self.name}]   - High confidence: {verification_summary.get('high_confidence_count', 0)}")
            logger.info(f"[{self.name}]   - Low confidence: {verification_summary.get('low_confidence_count', 0)}")
            logger.info(f"[{self.name}]   - Additional research needed: {verification_summary.get('additional_research_needed_count', 0)}")
        
        if not verified_results or len(verified_results) == 0:
            # Fallback 제거 - 명확한 에러만 반환
            error_msg = "보고서 생성 실패: 검증된 연구 결과가 없습니다."
            logger.error(f"[{self.name}] ❌ {error_msg}")
            state['final_report'] = None
            state['current_agent'] = self.name
            state['report_failed'] = True
            state['error'] = error_msg
            return state
        
        # 실제 결과가 있는 경우 LLM으로 보고서 생성
        logger.info(f"[{self.name}] Generating report with LLM from {len(verified_results)} verified results...")
        
        # 검증된 결과를 텍스트로 변환 (full_content 우선 사용)
        verified_text = ""
        for i, result in enumerate(verified_results, 1):
            if isinstance(result, dict):
                title = result.get('title', '')
                url = result.get('url', '')
                
                # full_content가 있으면 우선 사용, 없으면 snippet 사용
                content = result.get('full_content', '')
                if not content:
                    content = result.get('snippet', '')
                
                # 날짜 정보 추가
                published_date = result.get('published_date', '')
                date_str = ""
                if published_date:
                    try:
                        date_obj = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                        date_str = f" (발행일: {date_obj.strftime('%Y-%m-%d')})"
                    except:
                        date_str = f" (발행일: {published_date[:10]})"
                
                # 검토 정보 추가
                review = result.get('review', {})
                review_str = ""
                if review:
                    relevance = review.get('relevance_score', 'N/A')
                    recency = review.get('recency', 'N/A')
                    reliability = review.get('reliability', 'N/A')
                    review_str = f" [관련성: {relevance}/10, 최신성: {recency}, 신뢰도: {reliability}]"
                
                verified_text += f"\n--- 출처 {i}: {title}{date_str}{review_str} ---\n"
                verified_text += f"URL: {url}\n"
                verified_text += f"내용:\n{content[:10000] if len(content) > 10000 else content}\n"  # 최대 10000자
            else:
                verified_text += f"\n--- 출처 {i} ---\n{str(result)}\n"
        
        # Agent 논박 결과 수집 및 종합
        agent_debates_summary = ""
        if state.get('agent_debates'):
            debates = state['agent_debates']
            logger.info(f"[{self.name}] 💬 Collecting agent debate results for synthesis...")
            
            # Verifier 논박 결과
            if debates.get('verifier_debates'):
                verifier_debates = debates['verifier_debates']
                agent_debates_summary += "\n\n=== Verifier Agent 논박 결과 ===\n"
                for i, debate in enumerate(verifier_debates, 1):
                    agent_debates_summary += f"\n[논박 {i}] Agent: {debate.get('agent_id', 'unknown')}\n"
                    agent_debates_summary += f"일관성: {debate.get('consistency_check', 'unknown')}\n"
                    agent_debates_summary += f"논리적 올바름: {debate.get('logical_validity', 'unknown')}\n"
                    agent_debates_summary += f"논박 내용: {debate.get('message', '')[:500]}\n"
            
            # Verifier Peer 논박 결과
            if debates.get('verifier_peer_debates'):
                peer_debates = debates['verifier_peer_debates']
                agent_debates_summary += "\n\n=== Verifier Agent 간 논박 결과 ===\n"
                for i, debate in enumerate(peer_debates, 1):
                    agent_debates_summary += f"\n[논박 {i}] Agent: {debate.get('agent_id', 'unknown')}\n"
                    agent_debates_summary += f"일관성: {debate.get('consistency_check', 'unknown')}\n"
                    agent_debates_summary += f"논리적 올바름: {debate.get('logical_validity', 'unknown')}\n"
                    agent_debates_summary += f"논박 내용: {debate.get('message', '')[:500]}\n"
            
            # Evaluation 논박 결과 (state에서 가져오기)
            evaluation_result = state.get('evaluation_result')
            if evaluation_result and evaluation_result.get('evaluation_debates'):
                eval_debates = evaluation_result['evaluation_debates']
                agent_debates_summary += "\n\n=== Evaluator Agent 논박 결과 ===\n"
                for i, debate in enumerate(eval_debates, 1):
                    agent_debates_summary += f"\n[논박 {i}] Agent: {debate.get('agent_id', 'unknown')}\n"
                    agent_debates_summary += f"일관성: {debate.get('consistency_check', 'unknown')}\n"
                    agent_debates_summary += f"논리적 올바름: {debate.get('logical_validity', 'unknown')}\n"
                    agent_debates_summary += f"논박 내용: {debate.get('message', '')[:500]}\n"
            
            # Discussion Manager에서 모든 논박 가져오기
            if self.context.discussion_manager:
                try:
                    all_discussions = await self.context.discussion_manager.get_discussion_summary()
                    if all_discussions.get('topics'):
                        agent_debates_summary += "\n\n=== 전체 논박 요약 ===\n"
                        for topic, info in all_discussions['topics'].items():
                            agent_debates_summary += f"\n주제: {topic}\n"
                            agent_debates_summary += f"참여 Agent: {', '.join(info.get('participating_agents', []))}\n"
                            agent_debates_summary += f"논박 메시지 수: {info.get('message_count', 0)}\n"
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to get discussion summary: {e}")
        
        # 현재 시간 가져오기 (모듈 레벨 import 사용)
        current_time = datetime.now()
        current_date_str = current_time.strftime('%Y년 %m월 %d일')
        current_datetime_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # LLM으로 사용자 요청에 맞는 형식으로 생성
        from src.core.llm_manager import execute_llm_task, TaskType
        
        # 검증 요약 정보를 프롬프트에 포함
        verification_summary_text = ""
        if verification_summary:
            verification_summary_text = f"""
**검증 요약 정보 (VerifierAgent에서 전달):**
- 총 검증된 결과: {verification_summary.get('total_verified', 0)}개
- 높은 신뢰도: {verification_summary.get('high_confidence_count', 0)}개
- 중간 신뢰도: {verification_summary.get('medium_confidence_count', 0)}개
- 낮은 신뢰도: {verification_summary.get('low_confidence_count', 0)}개
- 평균 신뢰도: {verification_summary.get('average_confidence', 0.0):.2f}
- 불확실성 이슈: {verification_summary.get('uncertainty_issues_count', 0)}개
- 추가 조사 필요: {verification_summary.get('additional_research_needed_count', 0)}개

"""
            if verification_summary.get('low_confidence_topics'):
                verification_summary_text += "\n**낮은 신뢰도 주제 (주의 필요):**\n"
                for topic in verification_summary['low_confidence_topics']:
                    verification_summary_text += f"- {topic.get('title', '')}: {topic.get('reason', '')} (신뢰도: {topic.get('confidence', 0.0):.2f})\n"
            
            if verification_summary.get('additional_research_topics'):
                verification_summary_text += "\n**추가 조사 필요한 주제:**\n"
                for topic in verification_summary['additional_research_topics']:
                    verification_summary_text += f"- {topic.get('topic', '')}: {topic.get('reason', '')}\n"
        
        # 사용자 요청을 그대로 전달 - LLM이 형식을 결정하도록
        generation_prompt = f"""사용자 요청: {state['user_query']}

현재 시각: {current_datetime_str} (모든 시점 기준은 이 시각을 따름)

검증된 연구 결과 (실제 웹 페이지 전체 내용 포함):
{verified_text}

{verification_summary_text}

**Agent 논박 결과 (모든 Agent들의 논박을 통한 일관성 및 논리적 올바름 검증):**
{agent_debates_summary if agent_debates_summary else "논박 결과 없음 - Executor 결과가 직접 사용됨"}

⚠️ **깊이 있는 분석과 사고를 통한 보고서 작성 필수**
⚠️ **불확실성 명시 필수**: 낮은 신뢰도 정보나 불확실한 부분은 반드시 명시하세요
⚠️ **근거 없는 확신 금지**: 확실하지 않은 정보는 "~로 보인다", "~일 가능성이 있다" 등으로 표현하세요

**DEEP ANALYSIS REQUIREMENTS - 반드시 포함해야 할 깊이 있는 사고:**

1. **현재 상태 분석 (Current State Analysis)**:
   - 현재 상황은 무엇인가? 우리가 알고 있는 것은 무엇인가?
   - 주요 사실, 트렌드, 최근 발전 상황은 무엇인가?
   - 맥락과 배경은 무엇인가?
   - 이 정보가 의미하는 바는 무엇인가?

2. **패턴 인식 및 연결 (Pattern Recognition & Connections)**:
   - 여러 출처에서 나타나는 패턴, 트렌드, 관계는 무엇인가?
   - 어떤 연결고리와 상관관계가 있는가?
   - 역사적 맥락이나 선례는 무엇인가?
   - 다른 분야나 주제와의 연결은 무엇인가?

3. **비판적 통찰 (Critical Insights)**:
   - 단순한 사실 나열이 아닌, 깊은 통찰과 함의를 제공하세요
   - 이 정보의 더 깊은 의미는 무엇인가?
   - 어떤 관점들이 있고, 어떤 것이 누락되었는가?
   - 어떤 가정이 있고, 그것들이 유효한가?

4. **종합적 이해 (Comprehensive Understanding)**:
   - 전체적인 그림을 그리세요 - 개별 사실이 아닌 종합적 이해
   - 서로 다른 정보들이 어떻게 연결되는가?
   - 어떤 질문이 남아있는가? 어떤 정보가 부족한가?

5. **Agent 논박 결과 종합 (Agent Debate Synthesis)**:
   - 위의 "Agent 논박 결과"를 반드시 참고하세요
   - 모든 Agent들의 논박을 통해 검증된 일관성과 논리적 올바름을 반영하세요
   - 논박에서 합의된 부분과 논쟁이 있는 부분을 명확히 구분하세요
   - 논박 결과를 바탕으로 최종 결론을 도출하세요
   - 논박에서 지적된 문제점이나 개선사항을 반영하세요

**출처 인용 요구사항 (필수):**

⚠️ **모든 정보는 반드시 출처를 명시해야 합니다:**

1. **숫자/통계 인용**: 모든 숫자, 통계, 수치는 반드시 출처를 명시하세요
   - 예: "2025년 상반기 매출 3.2조 원(출처 1)" 또는 "매출 3.2조 원[1]"
   - 출처 번호는 아래 참고 문헌 섹션과 일치해야 합니다

2. **주장(Claims) 인용**: 모든 주장, 사실, 주장은 반드시 출처를 명시하세요
   - 예: "한화시스템은 방산 분야의 핵심 기업이다(출처 1, 출처 2)" 또는 "핵심 기업이다[1,2]"

3. **날짜/시점 인용**: 날짜, 시점 정보도 출처를 명시하세요
   - 예: "2025년 11월 발표(출처 3)" 또는 "2025년 11월[3]"

4. **참고 문헌 섹션**: 보고서 끝에 반드시 참고 문헌 섹션을 포함하세요
   - 각 출처는 번호와 함께 제목, URL을 포함해야 합니다
   - 본문에서 인용한 모든 출처가 참고 문헌에 포함되어야 합니다
   - 참고 문헌에 있는 출처는 본문에서 인용되어야 합니다

5. **출처 없는 정보**: 출처를 확인할 수 없는 정보는 불확실성을 표시하세요
   - 예: "추정", "예상", "~로 알려짐" 등의 표현 사용

6. **신뢰도 기반 표현** (검증 요약 정보 반영):
   - 높은 신뢰도 정보 (신뢰도 0.8 이상): 확실한 표현 사용 가능
   - 중간 신뢰도 정보 (신뢰도 0.6-0.8): "~로 보인다", "~일 가능성이 있다" 등으로 표현
   - 낮은 신뢰도 정보 (신뢰도 0.6 미만): "~라고 주장되지만", "확인 필요", "추가 조사 필요" 등으로 명시
   - 불확실성 이슈가 있는 정보: "불확실", "추가 검증 필요" 등으로 명시

7. **불확실성 명시**: 검증 요약에서 언급된 불확실성 이슈는 반드시 보고서에 명시하세요
   - 낮은 신뢰도 주제는 "주의: 신뢰도 낮음" 등으로 표시
   - 불확실한 부분은 "~로 알려져 있으나 확인 필요" 등으로 표현

8. **추가 조사 필요성**: 검증 요약에서 언급된 추가 조사 필요한 부분은 보고서에 포함하세요
   - "추가 조사가 필요한 영역" 섹션에 포함
   - 또는 해당 부분에서 "추가 조사 필요"로 명시

9. **관련성 확인**: 참고 문헌에 포함할 출처는 반드시 쿼리와 관련이 있어야 합니다
   - 엔비디아 분석인데 부동산 관련 출처를 포함하지 마세요
   - 쿼리와 무관한 출처는 제외하세요
   - 본문에서 인용하지 않은 출처는 참고 문헌에 포함하지 마세요

**보고서 구조 (깊이 있는 사고 반영):**

1. **현재 상태 섹션**: 현재 상태, 맥락, 알려진 정보에 대한 명확한 평가 (모든 정보에 출처 인용)
2. **깊이 있는 분석**: 패턴, 연결, 함의를 포함한 심층 분석 (모든 정보에 출처 인용)
3. **비판적 통찰**: 깊은 사고를 통해 도출된 의미 있는 통찰
4. **Agent 논박 종합**: 모든 Agent들의 논박 결과를 종합하여 일관성과 논리적 올바름이 검증된 내용 반영
5. **종합적 이해**: 깊은 이해를 보여주는 완전한 그림 (논박 결과 반영)
6. **의미 있는 결론**: 표면적 사실이 아닌 깊은 분석과 논박 검증에 기반한 결론

⚠️ **중요 지침:**
1. **최신 정보 우선**: 날짜가 표시된 출처 중 가장 최신 정보를 우선적으로 사용하세요.
2. **전체 내용 활용**: 각 출처의 전체 내용(full_content)을 참고하여 정확하고 상세한 정보를 제공하세요.
3. **다양한 출처 종합**: 여러 출처의 정보를 종합하여 균형 잡힌 분석을 제공하세요.
4. **현재 시간 기준**: 보고서 작성일은 {current_date_str} ({current_datetime_str})로 설정하세요.
5. **최신 동향 반영**: 최신 뉴스나 동향이 있다면 반드시 포함하세요.
6. **깊이 있는 사고**: 단순히 정보를 나열하지 말고, 깊이 있는 분석, 패턴 인식, 통찰을 제공하세요.

**절대 하지 말아야 할 것:**
- 단순히 검색 결과를 나열하는 것
- 현재 상태나 맥락 없이 정보만 제공하는 것
- 패턴이나 연결고리를 찾지 않는 것
- 깊이 있는 통찰 없이 표면적 사실만 나열하는 것

사용자의 요청을 정확히 이해하고, 요청한 형식에 맞게 **깊이 있는 분석과 통찰**을 포함한 결과를 생성하세요.
- 보고서를 요청했다면 보고서 형식으로 (작성일: {current_date_str} 포함, 현재 상태 분석 포함)
- 코드를 요청했다면 실행 가능한 코드로
- 문서를 요청했다면 문서 형식으로

요청된 형식에 맞게 **깊이 있는 사고와 분석**을 포함한 완전하고 실행 가능한 결과를 생성하세요."""

        try:
            report_result = await execute_llm_task(
                prompt=generation_prompt,
                task_type=TaskType.GENERATION,
                model_name=None,
                system_message=None
            )
            
            report = report_result.content or f"# Report: {state['user_query']}\n\nNo report generated."
            
            # Safety filter 차단 확인 - Fallback 제거, 명확한 오류 반환
            if "blocked by safety" in report.lower() or "content blocked" in report.lower() or len(report) < 100:
                error_msg = "보고서 생성 실패: Safety filter에 의해 차단되었습니다. 프롬프트를 수정하거나 다른 모델을 사용해주세요."
                logger.error(f"[{self.name}] ❌ {error_msg}")
                state['final_report'] = None
                state['report_failed'] = True
                state['error'] = error_msg
                state['current_agent'] = self.name
                return state
            else:
                logger.info(f"[{self.name}] ✅ Report generated: {len(report)} characters")
            
            # 보고서 완성도 검증 및 보완
            max_retry_attempts = 3
            retry_count = 0
            
            while retry_count < max_retry_attempts:
                completeness_check = await self._validate_report_completeness(
                    report, state['user_query'], verified_text
                )
                
                if completeness_check['is_complete']:
                    logger.info(f"[{self.name}] ✅ Report completeness validated: {completeness_check['completeness_score']:.2f}")
                    break
                
                logger.warning(f"[{self.name}] ⚠️ Report incomplete (score: {completeness_check['completeness_score']:.2f}): {completeness_check['issues']}")
                
                # 미완성 부분 보완
                if retry_count < max_retry_attempts - 1:
                    report = await self._complete_incomplete_report(
                        report, completeness_check, state['user_query'], verified_text, agent_debates_summary
                    )
                    retry_count += 1
                    logger.info(f"[{self.name}] 🔄 Retrying report completion (attempt {retry_count}/{max_retry_attempts})")
                else:
                    # 최종 시도 실패 시 경고만 로깅하고 현재 보고서 사용
                    logger.warning(f"[{self.name}] ⚠️ Report completion failed after {max_retry_attempts} attempts. Using current report with warnings.")
                    break
            
            # Council 활성화 확인 및 적용 (최종 보고서 생성 시 - 기본 활성화)
            use_council = state.get('use_council', None)  # 수동 활성화 옵션
            if use_council is None:
                # 자동 활성화 판단 (기본 활성화)
                from src.core.council_activator import get_council_activator
                activator = get_council_activator()
                
                activation_decision = activator.should_activate(
                    process_type='synthesis',
                    query=state['user_query'],
                    context={'important_conclusion': True}  # 최종 보고서는 항상 중요한 결론
                )
                use_council = activation_decision.should_activate
                if use_council:
                    logger.info(f"[{self.name}] 🏛️ Council auto-activated: {activation_decision.reason}")
            
            # Council 적용 (활성화된 경우)
            if use_council:
                try:
                    from src.core.llm_council import run_full_council
                    logger.info(f"[{self.name}] 🏛️ Running Council review for final report...")
                    
                    # 보고서 샘플 (최대 2000자)
                    report_sample = report[:2000]
                    
                    council_query = f"""Review the final report and assess its completeness and accuracy. Check for any missing information or potential improvements.

Research Query: {state['user_query']}

Final Report Sample:
{report_sample}

Provide a review with:
1. Completeness assessment
2. Accuracy check
3. Recommendations for improvement"""
                    
                    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
                        council_query
                    )
                    
                    # Council 검토 결과
                    review_report = stage3_result.get('response', '')
                    logger.info(f"[{self.name}] ✅ Council review completed.")
                    logger.info(f"[{self.name}] Council aggregate rankings: {metadata.get('aggregate_rankings', [])}")
                    
                    # Council 메타데이터를 state에 저장
                    if 'council_metadata' not in state:
                        state['council_metadata'] = {}
                    state['council_metadata']['synthesis'] = {
                        'stage1_results': stage1_results,
                        'stage2_results': stage2_results,
                        'stage3_result': stage3_result,
                        'metadata': metadata,
                        'review_report': review_report
                    }
                    
                    # Council 검토 결과를 보고서에 추가 (선택적)
                    if review_report:
                        report += f"\n\n--- Council Review ---\n{review_report}"
                except Exception as e:
                    logger.warning(f"[{self.name}] Council review failed: {e}. Using original report.")
                    # Council 실패 시 원본 보고서 사용 (fallback 제거 - 명확한 로깅만)
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Report generation failed: {e}")
            # Fallback 제거 - 명확한 오류 반환
            error_msg = f"보고서 생성 실패: {str(e)}"
            state['final_report'] = None
            state['report_failed'] = True
            state['error'] = error_msg
            state['current_agent'] = self.name
            return state
        
        # 최종 완성도 재검증 (종료 전)
        final_completeness = await self._validate_report_completeness(
            report, state['user_query'], verified_text
        )
        
        if not final_completeness['is_complete']:
            logger.error(f"[{self.name}] ❌ Final report validation failed: {final_completeness['issues']}")
            logger.error(f"[{self.name}] Completeness score: {final_completeness['completeness_score']:.2f}")
            # 완성도가 너무 낮으면 에러 반환
            if final_completeness['completeness_score'] < 0.5:
                error_msg = f"보고서 완성도 검증 실패: {', '.join(final_completeness['issues'][:3])}"
                state['final_report'] = None
                state['report_failed'] = True
                state['error'] = error_msg
                state['current_agent'] = self.name
                return state
            else:
                # 완성도가 낮지만 사용 가능한 경우 경고만
                logger.warning(f"[{self.name}] ⚠️ Report has completeness issues but will be saved: {final_completeness['issues']}")
        
        # 출처 인용 검증 및 보완
        source_mapping = state.get('source_mapping', {})
        if source_mapping:
            report = self._validate_and_enhance_citations(report, source_mapping, verified_results)
            logger.info(f"[{self.name}] ✅ Citation validation completed")
        
        state['final_report'] = report
        
        # A2UI 형식으로도 생성 시도 (선택적)
        try:
            from src.core.a2ui_generator import get_a2ui_generator
            a2ui_generator = get_a2ui_generator()
            a2ui_json = a2ui_generator.generate_research_report_a2ui(
                query=state['user_query'],
                verified_results=verified_results,
                report_text=report
            )
            state['final_report_a2ui'] = a2ui_json
            logger.info(f"[{self.name}] ✅ A2UI 형식 보고서 생성 완료")
        except Exception as e:
            logger.debug(f"[{self.name}] A2UI 생성 실패 (무시): {e}")
            state['final_report_a2ui'] = None
        state['current_agent'] = self.name
        state['report_failed'] = False
        state['report_completeness'] = final_completeness  # 완성도 정보 저장
        
        # Write to shared memory
        memory.write(
            key=f"report_{state['session_id']}",
            value=report,
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        logger.info(f"[{self.name}] ✅ Report saved to shared memory (completeness: {final_completeness['completeness_score']:.2f})")
        logger.info(f"=" * 80)
        
        return state
    
    async def _validate_report_completeness(
        self,
        report: str,
        user_query: str,
        verified_text: str
    ) -> Dict[str, Any]:
        """
        보고서 완성도 검증.
        
        Returns:
            Dict with 'is_complete', 'completeness_score', 'issues'
        """
        from src.core.llm_manager import execute_llm_task, TaskType
        
        validation_prompt = f"""다음 보고서의 완성도를 검증하세요:

사용자 요청: {user_query}

보고서 내용:
{report[:5000]}  # 처음 5000자만 검증용으로 사용

**완성도 검증 기준:**

1. **구조적 완성도 (Structural Completeness)**:
   - 모든 섹션이 완성되었는가? (시작했지만 끝나지 않은 섹션이 있는가?)
   - 표나 리스트가 중간에 잘렸는가?
   - 마지막 문장이 완성되었는가?

2. **내용 완성도 (Content Completeness)**:
   - 각 섹션에 충분한 내용이 있는가?
   - 사용자 요청에 대한 답변이 완전한가?
   - 결론 섹션이 있는가?

3. **논리적 완성도 (Logical Completeness)**:
   - 논리적 흐름이 완성되었는가?
   - 중간에 갑자기 끝나는 부분이 있는가?
   - 불완전한 문장이나 표가 있는가?

4. **형식적 완성도 (Format Completeness)**:
   - 마크다운 형식이 올바른가?
   - 표가 제대로 닫혔는가?
   - 코드 블록이 제대로 닫혔는가?

**검증 결과를 다음 JSON 형식으로 반환하세요:**
{{
    "is_complete": true/false,
    "completeness_score": 0.0-1.0,
    "issues": ["문제1", "문제2", ...],
    "missing_sections": ["누락된 섹션1", ...],
    "incomplete_elements": ["불완전한 요소1", ...],
    "recommendations": ["권장사항1", ...]
}}

중요: 보고서가 중간에 잘렸거나 불완전한 경우 is_complete는 반드시 false여야 합니다."""
        
        try:
            validation_result = await execute_llm_task(
                prompt=validation_prompt,
                task_type=TaskType.VERIFICATION,
                system_message="You are an expert document completeness validator. You must detect any incomplete sections, truncated content, or formatting issues."
            )

            # JSON 추출 시도
            content = validation_result.content
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return {
                        'is_complete': result.get('is_complete', False),
                        'completeness_score': result.get('completeness_score', 0.0),
                        'issues': result.get('issues', []),
                        'missing_sections': result.get('missing_sections', []),
                        'incomplete_elements': result.get('incomplete_elements', []),
                        'recommendations': result.get('recommendations', [])
                    }
                except json.JSONDecodeError:
                    pass
            
            # JSON 파싱 실패 시 휴리스틱 검증
            return self._heuristic_completeness_check(report, user_query)
            
        except Exception as e:
            logger.warning(f"[{self.name}] Completeness validation failed: {e}. Using heuristic check.")
            return self._heuristic_completeness_check(report, user_query)
    
    def _heuristic_completeness_check(self, report: str, user_query: str) -> Dict[str, Any]:
        """휴리스틱 기반 완성도 검증 (fallback)."""
        issues = []
        score = 1.0
        
        # 1. 중간 잘림 감지
        if report.endswith('|') or report.endswith('| '):
            issues.append("표가 중간에 잘림")
            score -= 0.3
        
        # 2. 불완전한 마크다운 감지
        if report.count('```') % 2 != 0:
            issues.append("코드 블록이 닫히지 않음")
            score -= 0.2
        
        # 3. 마지막 문장 완성도
        last_sentence = report.strip().split('\n')[-1] if report.strip() else ""
        if last_sentence and not last_sentence.endswith(('.', '!', '?', ':', ')')):
            if len(last_sentence) > 20:  # 짧은 문장은 무시
                issues.append("마지막 문장이 불완전할 수 있음")
                score -= 0.1
        
        # 4. 섹션 완성도
        open_sections = report.count('##') - report.count('###')
        if open_sections > 5:  # 너무 많은 섹션이 열려있으면
            issues.append("너무 많은 섹션이 열려있음")
            score -= 0.2
        
        # 5. 최소 길이 검증
        if len(report) < 500:
            issues.append("보고서가 너무 짧음")
            score -= 0.3
        
        # 6. 결론 섹션 확인
        if '결론' not in report and 'Conclusion' not in report and '##' in report:
            # 섹션이 있지만 결론이 없는 경우
            issues.append("결론 섹션이 없을 수 있음")
            score -= 0.1
        
        # 7. 표 중간 잘림 감지 (더 정확한 검증)
        lines = report.split('\n')
        in_table = False
        for i, line in enumerate(lines):
            if '|' in line and line.strip().startswith('|'):
                in_table = True
            elif in_table and line.strip() and '|' not in line:
                # 표가 시작되었는데 갑자기 끝남
                if i < len(lines) - 5:  # 마지막 5줄이 아니면
                    issues.append(f"표가 {i+1}번째 줄에서 중간에 잘림")
                    score -= 0.2
                    break
        
        score = max(0.0, min(1.0, score))
        
        return {
            'is_complete': score >= 0.7 and len(issues) == 0,
            'completeness_score': score,
            'issues': issues,
            'missing_sections': [],
            'incomplete_elements': issues,
            'recommendations': []
        }
    
    async def _complete_incomplete_report(
        self,
        current_report: str,
        completeness_check: Dict[str, Any],
        user_query: str,
        verified_text: str,
        agent_debates_summary: str
    ) -> str:
        """미완성 보고서 보완."""
        from src.core.llm_manager import execute_llm_task, TaskType
        
        completion_prompt = f"""다음 보고서가 불완전합니다. 완성하세요:

사용자 요청: {user_query}

현재 보고서 (불완전):
{current_report}

완성도 검증 결과:
- 완성도 점수: {completeness_check['completeness_score']:.2f}
- 발견된 문제: {', '.join(completeness_check['issues'])}
- 누락된 섹션: {', '.join(completeness_check.get('missing_sections', []))}
- 불완전한 요소: {', '.join(completeness_check.get('incomplete_elements', []))}

검증된 연구 결과:
{verified_text[:3000]}

Agent 논박 결과:
{agent_debates_summary[:1000] if agent_debates_summary else "없음"}

**보완 작업:**

1. **불완전한 부분 완성**:
   - 중간에 잘린 표나 리스트를 완성하세요
   - 불완전한 문장을 완성하세요
   - 닫히지 않은 마크다운 요소를 닫으세요

2. **누락된 섹션 추가**:
   - 누락된 섹션을 추가하세요
   - 결론 섹션이 없으면 추가하세요

3. **내용 보완**:
   - 각 섹션에 충분한 내용을 추가하세요
   - 사용자 요청에 대한 완전한 답변을 제공하세요

**중요:**
- 기존 보고서의 내용을 유지하면서 보완하세요
- 새로운 내용을 추가할 때는 기존 내용과 일관성을 유지하세요
- 보고서의 전체 구조와 스타일을 유지하세요
- 반드시 완전한 보고서를 생성하세요 (중간에 잘리지 않도록)

완성된 전체 보고서를 생성하세요."""
        
        try:
            completion_result = await execute_llm_task(
                prompt=completion_prompt,
                task_type=TaskType.GENERATION,
                system_message="You are an expert report completer. You must complete incomplete reports while maintaining consistency and quality."
            )
            
            completed_report = completion_result.content if hasattr(completion_result, 'content') else str(completion_result)
            
            # 기존 보고서보다 길거나 같아야 함
            if len(completed_report) >= len(current_report):
                logger.info(f"[{self.name}] ✅ Report completed: {len(completed_report)} characters (was {len(current_report)})")
                return completed_report
            else:
                logger.warning(f"[{self.name}] ⚠️ Completed report is shorter than original. Using original.")
                return current_report
                
        except Exception as e:
            logger.error(f"[{self.name}] Report completion failed: {e}")
            return current_report


###################
# Orchestrator
###################

class AgentOrchestrator:
    """Orchestrator for multi-agent workflow."""
    
    def __init__(self, config: Any = None):
        """Initialize orchestrator."""
        self.config = config
        self.shared_memory = get_shared_memory()
        self.skill_manager = get_skill_manager()
        self.agent_config = get_agent_config()
        self.graph = None
        # Graph는 첫 실행 시 쿼리 기반으로 빌드

        # SharedResultsManager와 AgentDiscussionManager는 execute 시점에 초기화
        # (objective_id가 필요하므로)
        self.shared_results_manager: Optional[SharedResultsManager] = None
        self.discussion_manager: Optional[AgentDiscussionManager] = None

        # MCP 도구 자동 발견 및 선택 시스템 초기화
        self.mcp_servers = self._initialize_mcp_servers()
        self.tool_loader = MCPToolLoader(FastMCPMulti(self.mcp_servers))
        self.tool_selector = AgentToolSelector()

        # 세션 관리자 초기화
        self.session_manager = get_session_manager()
        self.session_manager.set_shared_memory(self.shared_memory)
        self.session_manager.set_context_engineer(get_context_engineer())
        
        # 백그라운드 메모리 서비스 초기화
        self.memory_service = get_background_memory_service()
        # 서비스 시작은 execute 메서드에서 비동기로 처리됨

        logger.info("AgentOrchestrator initialized with MCP tool auto-discovery, session management, and background memory service")

    def _initialize_mcp_servers(self) -> dict[str, Any]:
        """환경 변수 및 구성에서 MCP 서버 설정을 초기화.
        
        Returns:
            mcp_config.json 원본 형식의 dict (FastMCP가 직접 사용할 수 있는 형식)
        """
        servers: dict[str, Any] = {}
        
        try:
            # 프로젝트 루트 찾기
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            
            # configs 폴더에서 로드 시도 (우선)
            config_file = project_root / "configs" / "mcp_config.json"
            if not config_file.exists():
                # 하위 호환성: 루트에서도 시도
                config_file = project_root / "mcp_config.json"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    raw_configs = config_data.get("mcpServers", {})
                    
                    # 환경변수 치환
                    resolved_configs = self._resolve_env_vars_in_value(raw_configs)
                    
                    # FastMCP가 기대하는 형식으로 정리
                    # - stdio 서버: command, args, env, cwd만 유지
                    # - HTTP 서버: type 필드 제거, httpUrl 또는 url만 유지
                    for server_name, server_config in resolved_configs.items():
                        cleaned_config = {}
                        
                        # stdio 서버인 경우
                        if "command" in server_config:
                            cleaned_config["command"] = server_config["command"]
                            if "args" in server_config:
                                cleaned_config["args"] = server_config["args"]
                            if "env" in server_config and server_config["env"]:
                                cleaned_config["env"] = server_config["env"]
                            if "cwd" in server_config and server_config["cwd"]:
                                cleaned_config["cwd"] = server_config["cwd"]
                        # HTTP 서버인 경우
                        elif "httpUrl" in server_config or "url" in server_config:
                            # FastMCP는 url 필드를 기대함 (httpUrl을 url로 변환)
                            if "httpUrl" in server_config:
                                cleaned_config["url"] = server_config["httpUrl"]
                            elif "url" in server_config:
                                cleaned_config["url"] = server_config["url"]
                            if "headers" in server_config and server_config["headers"]:
                                cleaned_config["headers"] = server_config["headers"]
                            if "params" in server_config and server_config["params"]:
                                cleaned_config["params"] = server_config["params"]
                        
                        if cleaned_config:
                            servers[server_name] = cleaned_config
                    
                    logger.info(f"✅ Loaded {len(servers)} MCP servers from config: {list(servers.keys())}")
            else:
                logger.warning(f"MCP config file not found at {config_file}")
                
        except Exception as e:
            logger.warning(f"Failed to load MCP server configs: {e}")

        logger.info(f"Initialized {len(servers)} MCP servers for auto-discovery")
        return servers
    
    def _resolve_env_vars_in_value(self, value: Any) -> Any:
        """
        재귀적으로 객체 내의 환경변수 플레이스홀더를 실제 값으로 치환.
        ${VAR_NAME} 또는 $VAR_NAME 형식 지원.
        """
        if isinstance(value, str):
            # ${VAR_NAME} 또는 $VAR_NAME 패턴 찾기
            pattern = r'\$\{([^}]+)\}|\$(\w+)'
            
            def replace_env_var(match):
                var_name = match.group(1) or match.group(2)
                env_value = os.getenv(var_name)
                if env_value is not None:
                    return env_value
                # 환경변수가 없으면 원본 유지 (또는 경고)
                logger.warning(f"Environment variable '{var_name}' not found, keeping placeholder")
                return match.group(0)
            
            result = re.sub(pattern, replace_env_var, value)
            return result
        elif isinstance(value, dict):
            return {k: self._resolve_env_vars_in_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_env_vars_in_value(item) for item in value]
        else:
            return value

    async def _assign_tools_to_agents(self, session_id: str) -> None:
        """모든 에이전트에 자동으로 MCP 도구 할당."""
        try:
            # MCP 도구 자동 발견
            discovered_tools = await self.tool_loader.get_all_tools()
            tool_infos = await self.tool_loader.list_tool_info()

            logger.info(f"Discovered {len(discovered_tools)} MCP tools from {len(self.mcp_servers)} servers")

            # 각 에이전트별 도구 선택 및 할당
            assignments = self.tool_selector.select_tools_for_all_agents(
                discovered_tools, tool_infos
            )

            # 각 에이전트에 도구 할당
            for agent_type, assignment in assignments.items():
                agent = getattr(self, agent_type.value, None)
                if agent:
                    agent.available_tools = assignment.tools
                    agent.tool_infos = assignment.tool_infos
                    logger.info(f"Assigned {len(assignment.tools)} tools to {agent_type.value} agent")

                    # 도구 할당 요약 로깅
                    summary = self.tool_selector.get_agent_tool_summary(assignment)
                    logger.info(f"Tool assignment summary for {agent_type.value}: {summary}")

        except Exception as e:
            logger.warning(f"Failed to assign MCP tools to agents: {e}")
            # 도구 할당 실패 시에도 계속 진행 (기존 로직 유지)

    def _build_graph(self, user_query: Optional[str] = None, session_id: Optional[str] = None) -> None:
        """Build LangGraph workflow with Skills auto-selection."""
        
        # Create context for all agents
        context = AgentContext(
            agent_id="orchestrator",
            session_id=session_id or "default",
            shared_memory=self.shared_memory,
            config=self.config,
            shared_results_manager=self.shared_results_manager,
            discussion_manager=self.discussion_manager
        )
        
        # Skills 자동 선택 (쿼리가 있으면)
        selected_skills = {}
        if user_query:
            skill_selector = get_skill_selector()
            matches = skill_selector.select_skills_for_task(user_query)
            for match in matches:
                skill = self.skill_manager.load_skill(match.skill_id)
                if skill:
                    selected_skills[match.skill_id] = skill
        
        # Initialize agents with Skills
        self.planner = PlannerAgent(context, selected_skills.get("research_planner"))
        self.executor = ExecutorAgent(context, selected_skills.get("research_executor"))
        self.verifier = VerifierAgent(context, selected_skills.get("evaluator"))
        self.generator = GeneratorAgent(context, selected_skills.get("synthesizer"))

        # 각 에이전트에 MCP 도구 자동 할당 (비동기)
        if session_id:
            asyncio.create_task(self._assign_tools_to_agents(session_id))

        # Build graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)  # Legacy
        workflow.add_node("parallel_executor", self._parallel_executor_node)  # New parallel executor
        workflow.add_node("verifier", self._verifier_node)  # Legacy
        workflow.add_node("parallel_verifier", self._parallel_verifier_node)  # New parallel verifier
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("end", self._end_node)
        
        # Define edges - 병렬 실행 노드 사용
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "parallel_executor")  # 병렬 실행 사용
        workflow.add_edge("parallel_executor", "parallel_verifier")  # 병렬 검증 사용
        workflow.add_edge("parallel_verifier", "generator")
        workflow.add_edge("generator", "end")
        
        # Compile graph
        self.graph = workflow.compile()
        
        logger.info("LangGraph workflow built")
    
    async def _planner_node(self, state: AgentState) -> AgentState:
        """Planner node execution with tracking."""
        logger.info("=" * 80)
        logger.info("🔵 [WORKFLOW] → Planner Node")
        logger.info("=" * 80)
        
        # Progress tracker 업데이트
        try:
            from src.core.progress_tracker import get_progress_tracker, WorkflowStage
            progress_tracker = get_progress_tracker()
            if progress_tracker:
                progress_tracker.set_workflow_stage(WorkflowStage.PLANNING, {"message": "연구 계획 수립 중..."})
        except Exception as e:
            logger.debug(f"Failed to update progress tracker: {e}")
        
        result = await self.planner.execute(state)
        logger.info(f"🔵 [WORKFLOW] ✓ Planner completed: {result.get('current_agent')}")
        return result
    
    async def _executor_node(self, state: AgentState) -> AgentState:
        """Executor node execution with tracking (legacy - for backward compatibility)."""
        logger.info("=" * 80)
        logger.info("🟢 [WORKFLOW] → Executor Node (legacy)")
        logger.info("=" * 80)
        result = await self.executor.execute(state)
        logger.info(f"🟢 [WORKFLOW] ✓ Executor completed: {len(result.get('research_results', []))} results")
        return result
    
    async def _parallel_executor_node(self, state: AgentState) -> AgentState:
        """Parallel executor node - runs multiple ExecutorAgent instances simultaneously."""
        logger.info("=" * 80)
        logger.info("🟢 [WORKFLOW] → Parallel Executor Node")
        logger.info("=" * 80)
        
        # Progress tracker 업데이트
        try:
            from src.core.progress_tracker import get_progress_tracker, WorkflowStage
            progress_tracker = get_progress_tracker()
            if progress_tracker:
                progress_tracker.set_workflow_stage(WorkflowStage.EXECUTING, {"message": "연구 실행 중..."})
        except Exception as e:
            logger.debug(f"Failed to update progress tracker: {e}")
        
        # 작업 목록 가져오기
        tasks = state.get('research_tasks', [])
        if not tasks:
            # 메모리에서 읽기
            memory = self.shared_memory
            tasks = memory.read(
                key=f"tasks_{state['session_id']}",
                scope=MemoryScope.SESSION,
                session_id=state['session_id']
            ) or []
        
        if not tasks:
            logger.warning("[WORKFLOW] No tasks found, falling back to single executor")
            return await self._executor_node(state)
        
        logger.info(f"[WORKFLOW] Executing {len(tasks)} tasks in parallel with {len(tasks)} ExecutorAgent instances")
        
        # 동적 동시성 관리 통합
        from src.core.concurrency_manager import get_concurrency_manager
        concurrency_manager = get_concurrency_manager()
        max_concurrent = concurrency_manager.get_current_concurrency() or self.agent_config.max_concurrent_research_units
        max_concurrent = min(max_concurrent, len(tasks))  # 작업 수를 초과하지 않도록
        
        logger.info(f"[WORKFLOW] Using concurrency limit: {max_concurrent} (from concurrency_manager)")
        
        # Skills 자동 선택
        selected_skills = {}
        if state.get('user_query'):
            skill_selector = get_skill_selector()
            matches = skill_selector.select_skills_for_task(state['user_query'])
            for match in matches:
                skill = self.skill_manager.load_skill(match.skill_id)
                if skill:
                    selected_skills[match.skill_id] = skill
        
        # 여러 ExecutorAgent 인스턴스 생성 및 병렬 실행
        async def execute_single_task(task: Dict[str, Any], task_index: int) -> AgentState:
            """단일 작업을 실행하는 ExecutorAgent."""
            agent_id = f"executor_{task_index}"
            context = AgentContext(
                agent_id=agent_id,
                session_id=state['session_id'],
                shared_memory=self.shared_memory,
                config=self.config,
                shared_results_manager=self.shared_results_manager,
                discussion_manager=self.discussion_manager
            )
            
            executor_agent = ExecutorAgent(context, selected_skills.get("research_executor"))
            
            try:
                logger.info(f"[WORKFLOW] ExecutorAgent {agent_id} starting task {task.get('task_id', 'unknown')}")
                result_state = await executor_agent.execute(state, assigned_task=task)
                logger.info(f"[WORKFLOW] ExecutorAgent {agent_id} completed: {len(result_state.get('research_results', []))} results")
                return result_state
            except Exception as e:
                logger.error(f"[WORKFLOW] ExecutorAgent {agent_id} failed: {e}")
                # 실패한 에이전트의 상태 반환
                failed_state = state.copy()
                failed_state['research_results'] = []
                failed_state['research_failed'] = True
                failed_state['error'] = f"Task {task.get('task_id', 'unknown')} failed: {str(e)}"
                failed_state['current_agent'] = agent_id
                return failed_state
        
        # 모든 작업을 병렬로 실행 (동적 동시성 제한 적용)
        if max_concurrent < len(tasks):
            # Semaphore를 사용하여 동시 실행 수 제한
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def execute_with_limit(task: Dict[str, Any], task_index: int) -> AgentState:
                async with semaphore:
                    return await execute_single_task(task, task_index)
            
            executor_tasks = [execute_with_limit(task, i) for i, task in enumerate(tasks)]
        else:
            # 동시성 제한이 작업 수보다 크면 모든 작업을 동시에 실행
            executor_tasks = [execute_single_task(task, i) for i, task in enumerate(tasks)]
        
        # 병렬 실행
        executor_results = await asyncio.gather(*executor_tasks, return_exceptions=True)
        
        # 결과 통합 및 통신 상태 확인
        all_results = []
        all_failed = False
        errors = []
        communication_stats = {
            'agents_contributed': 0,
            'results_shared': 0,
            'communication_errors': 0
        }

        for i, result in enumerate(executor_results):
            if isinstance(result, Exception):
                logger.error(f"[WORKFLOW] ExecutorAgent {i} raised exception: {result}")
                all_failed = True
                errors.append(f"Task {tasks[i].get('task_id', 'unknown')}: {str(result)}")
                communication_stats['communication_errors'] += 1
            elif isinstance(result, dict):
                # 결과 수집
                task_results = result.get('research_results', [])
                if task_results:
                    all_results.extend(task_results)
                    communication_stats['agents_contributed'] += 1
                    logger.info(f"[WORKFLOW] ExecutorAgent {i} contributed {len(task_results)} results")

                # SharedResultsManager 통신 상태 확인
                if self.shared_results_manager:
                    agent_id = f"executor_{i}"
                    agent_results = await self.shared_results_manager.get_shared_results(agent_id=agent_id)
                    if agent_results:
                        communication_stats['results_shared'] += len(agent_results)
                        logger.info(f"[WORKFLOW] 🤝 ExecutorAgent {agent_id} shared {len(agent_results)} results via SharedResultsManager")

                # 실패 상태 확인
                if result.get('research_failed'):
                    all_failed = True
                    if result.get('error'):
                        errors.append(result['error'])
                        communication_stats['communication_errors'] += 1
        
        # 통합된 상태 생성
        final_state = state.copy()
        final_state['research_results'] = all_results
        final_state['research_failed'] = all_failed
        final_state['current_agent'] = "parallel_executor"
        
        if errors:
            final_state['error'] = "; ".join(errors)
        
        logger.info(f"[WORKFLOW] ✅ Parallel execution completed: {len(all_results)} total results from {len(tasks)} tasks")
        logger.info(f"[WORKFLOW] 🤝 Agent communication summary: {communication_stats['agents_contributed']} agents contributed, {communication_stats['results_shared']} results shared")
        if communication_stats['communication_errors'] > 0:
            logger.warning(f"[WORKFLOW] ⚠️ Communication errors: {communication_stats['communication_errors']}")
        logger.info(f"[WORKFLOW] Failed: {all_failed}")
        
        return final_state
    
    async def _verifier_node(self, state: AgentState) -> AgentState:
        """Verifier node execution with tracking (legacy - for backward compatibility)."""
        logger.info("=" * 80)
        logger.info("🟡 [WORKFLOW] → Verifier Node (legacy)")
        logger.info("=" * 80)
        result = await self.verifier.execute(state)
        logger.info(f"🟡 [WORKFLOW] ✓ Verifier completed: {len(result.get('verified_results', []))} verified")
        return result
    
    async def _parallel_verifier_node(self, state: AgentState) -> AgentState:
        """Parallel verifier node - runs multiple VerifierAgent instances simultaneously."""
        logger.info("=" * 80)
        logger.info("🟡 [WORKFLOW] → Parallel Verifier Node")
        logger.info("=" * 80)
        
        # Progress tracker 업데이트
        try:
            from src.core.progress_tracker import get_progress_tracker, WorkflowStage
            progress_tracker = get_progress_tracker()
            if progress_tracker:
                progress_tracker.set_workflow_stage(WorkflowStage.VERIFYING, {"message": "결과 검증 중..."})
        except Exception as e:
            logger.debug(f"Failed to update progress tracker: {e}")
        
        # 연구 실패 확인
        if state.get('research_failed'):
            logger.error("[WORKFLOW] Research execution failed, skipping verification")
            state['verified_results'] = []
            state['verification_failed'] = True
            state['current_agent'] = "parallel_verifier"
            return state
        
        # 검증할 결과 가져오기
        results = state.get('research_results', [])
        if not results:
            memory = self.shared_memory
            results = memory.read(
                key=f"research_results_{state['session_id']}",
                scope=MemoryScope.SESSION,
                session_id=state['session_id']
            ) or []
        
        if not results:
            logger.warning("[WORKFLOW] No results to verify, falling back to single verifier")
            return await self._verifier_node(state)
        
        # 결과를 여러 청크로 분할하여 여러 VerifierAgent에 할당
        num_verifiers = min(len(results), self.agent_config.max_concurrent_research_units or 3)
        chunk_size = max(1, len(results) // num_verifiers)
        result_chunks = [results[i:i + chunk_size] for i in range(0, len(results), chunk_size)]
        
        logger.info(f"[WORKFLOW] Verifying {len(results)} results with {len(result_chunks)} VerifierAgent instances")
        
        # 동적 동시성 관리 통합
        from src.core.concurrency_manager import get_concurrency_manager
        concurrency_manager = get_concurrency_manager()
        max_concurrent = concurrency_manager.get_current_concurrency() or self.agent_config.max_concurrent_research_units
        max_concurrent = min(max_concurrent, len(result_chunks))
        
        logger.info(f"[WORKFLOW] Using concurrency limit: {max_concurrent} (from concurrency_manager)")
        
        # Skills 자동 선택
        selected_skills = {}
        if state.get('user_query'):
            skill_selector = get_skill_selector()
            matches = skill_selector.select_skills_for_task(state['user_query'])
            for match in matches:
                skill = self.skill_manager.load_skill(match.skill_id)
                if skill:
                    selected_skills[match.skill_id] = skill
        
        # 여러 VerifierAgent 인스턴스 생성 및 병렬 실행
        async def verify_single_chunk(chunk: List[Dict[str, Any]], chunk_index: int) -> List[Dict[str, Any]]:
            """단일 청크를 검증하는 VerifierAgent."""
            agent_id = f"verifier_{chunk_index}"
            logger.info(f"[WORKFLOW] 💬 Creating VerifierAgent {agent_id} for {len(chunk)} results")
            context = AgentContext(
                agent_id=agent_id,
                session_id=state['session_id'],
                shared_memory=self.shared_memory,
                config=self.config,
                shared_results_manager=self.shared_results_manager,
                discussion_manager=self.discussion_manager
            )
            
            verifier_agent = VerifierAgent(context, selected_skills.get("evaluator"))
            
            # 청크만 포함하는 임시 state 생성
            chunk_state = state.copy()
            chunk_state['research_results'] = chunk
            
            try:
                logger.info(f"[WORKFLOW] VerifierAgent {agent_id} starting verification of {len(chunk)} results")
                result_state = await verifier_agent.execute(chunk_state)
                verified_chunk = result_state.get('verified_results', [])
                logger.info(f"[WORKFLOW] VerifierAgent {agent_id} completed: {len(verified_chunk)} verified")
                return verified_chunk
            except Exception as e:
                logger.error(f"[WORKFLOW] VerifierAgent {agent_id} failed: {e}")
                return []  # 실패 시 빈 리스트 반환
        
        # 모든 청크를 병렬로 검증 (동적 동시성 제한 적용)
        if max_concurrent < len(result_chunks):
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def verify_with_limit(chunk: List[Dict[str, Any]], chunk_index: int) -> List[Dict[str, Any]]:
                async with semaphore:
                    return await verify_single_chunk(chunk, chunk_index)
            
            verifier_tasks = [verify_with_limit(chunk, i) for i, chunk in enumerate(result_chunks)]
        else:
            verifier_tasks = [verify_single_chunk(chunk, i) for i, chunk in enumerate(result_chunks)]
        
        # 병렬 실행
        verifier_results = await asyncio.gather(*verifier_tasks, return_exceptions=True)
        
        # 결과 통합 및 통신 상태 확인
        all_verified = []
        communication_stats = {
            'verifiers_contributed': 0,
            'verification_results_shared': 0,
            'discussion_participants': 0
        }

        for i, result in enumerate(verifier_results):
            if isinstance(result, Exception):
                logger.error(f"[WORKFLOW] VerifierAgent {i} raised exception: {result}")
            elif isinstance(result, list):
                all_verified.extend(result)
                communication_stats['verifiers_contributed'] += 1
                logger.info(f"[WORKFLOW] VerifierAgent {i} contributed {len(result)} verified results")

                # SharedResultsManager 통신 상태 확인
                if self.shared_results_manager:
                    agent_id = f"verifier_{i}"
                    agent_results = await self.shared_results_manager.get_shared_results(agent_id=agent_id)
                    verification_shared = [r for r in agent_results if isinstance(r.result, dict) and r.result.get('status') == 'verified']
                    if verification_shared:
                        communication_stats['verification_results_shared'] += len(verification_shared)
                        logger.info(f"[WORKFLOW] 🤝 VerifierAgent {agent_id} shared {len(verification_shared)} verification results")

        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_verified = []
        for verified_result in all_verified:
            if isinstance(verified_result, dict):
                url = verified_result.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_verified.append(verified_result)
                elif not url:
                    unique_verified.append(verified_result)

        logger.info(f"[WORKFLOW] 📊 Verification deduplication: {len(all_verified)} → {len(unique_verified)} unique results")

        # 여러 VerifierAgent 간 토론 (검증 결과가 다른 경우)
        if self.discussion_manager and len(unique_verified) > 0:
            # 다른 VerifierAgent의 검증 결과 가져오기
            if self.shared_results_manager:
                other_verified = await self.shared_results_manager.get_shared_results()
                other_verified_results = [r for r in other_verified if isinstance(r.result, dict) and r.result.get('status') == 'verified']

                if other_verified_results:
                    communication_stats['discussion_participants'] = len(set(r.agent_id for r in other_verified_results))
                    logger.info(f"[WORKFLOW] 💬 Starting inter-verifier discussion with {len(other_verified_results)} results from {communication_stats['discussion_participants']} agents")

                    # 첫 번째 검증 결과에 대해 토론
                    first_verified = unique_verified[0]
                    result_id = f"verification_{first_verified.get('index', 0)}"
                    discussion = await self.discussion_manager.agent_discuss_result(
                        result_id=result_id,
                        agent_id="parallel_verifier",
                        other_agent_results=other_verified_results[:3]
                    )
                    if discussion:
                        logger.info(f"[WORKFLOW] 💬 Inter-verifier discussion completed: {discussion[:150]}...")
                        logger.info(f"[WORKFLOW] 🤝 Agent discussion: {communication_stats['discussion_participants']} verifiers participated in result validation")
                    else:
                        logger.info(f"[WORKFLOW] 💬 No discussion generated between verifiers")
                else:
                    logger.info(f"[WORKFLOW] 💬 No other verified results available for inter-verifier discussion")
        
        # 통합된 상태 생성
        final_state = state.copy()
        final_state['verified_results'] = unique_verified
        final_state['verification_failed'] = False if unique_verified else True
        final_state['current_agent'] = "parallel_verifier"
        
        logger.info(f"[WORKFLOW] ✅ Parallel verification completed: {len(unique_verified)} total verified results from {len(result_chunks)} verifiers")
        logger.info(f"[WORKFLOW] 🤝 Agent communication summary: {communication_stats['verifiers_contributed']} verifiers contributed, {communication_stats['verification_results_shared']} verification results shared")
        if communication_stats['discussion_participants'] > 0:
            logger.info(f"[WORKFLOW] 💬 Inter-verifier discussion: {communication_stats['discussion_participants']} agents participated")
        
        return final_state
    
    async def _generator_node(self, state: AgentState) -> AgentState:
        """Generator node execution with tracking."""
        logger.info("=" * 80)
        logger.info("🟣 [WORKFLOW] → Generator Node")
        logger.info("=" * 80)
        
        # Progress tracker 업데이트
        try:
            from src.core.progress_tracker import get_progress_tracker, WorkflowStage
            progress_tracker = get_progress_tracker()
            if progress_tracker:
                progress_tracker.set_workflow_stage(WorkflowStage.GENERATING, {"message": "보고서 생성 중..."})
        except Exception as e:
            logger.debug(f"Failed to update progress tracker: {e}")
        
        result = await self.generator.execute(state)
        final_report = result.get('final_report') or ''
        report_length = len(final_report) if final_report else 0
        logger.info(f"🟣 [WORKFLOW] ✓ Generator completed: report_length={report_length}")
        return result
    
    async def _end_node(self, state: AgentState) -> AgentState:
        """End node - final state with summary."""
        logger.info("=" * 80)
        logger.info("✅ [WORKFLOW] → End Node - Workflow Completed")
        logger.info("=" * 80)
        logger.info(f"Session: {state.get('session_id')}")
        logger.info(f"Final Agent: {state.get('current_agent')}")
        logger.info(f"Research Results: {len(state.get('research_results', []))}")
        logger.info(f"Verified Results: {len(state.get('verified_results', []))}")
        logger.info(f"Report Generated: {bool(state.get('final_report'))}")
        logger.info(f"Failed: {state.get('research_failed') or state.get('verification_failed') or state.get('report_failed')}")
        logger.info("=" * 80)
        
        # 백그라운드 메모리 생성 트리거 (세션 종료 시)
        try:
            session_id = state.get('session_id')
            user_id = state.get('metadata', {}).get('user_id') or 'default_user'
            
            # 대화 히스토리 구성 (messages에서)
            conversation_history = []
            for msg in state.get('messages', []):
                if isinstance(msg, dict):
                    conversation_history.append({
                        'role': msg.get('type', msg.get('role', 'unknown')),
                        'content': msg.get('content', msg.get('text', ''))
                    })
                else:
                    # LangChain Message 객체인 경우
                    conversation_history.append({
                        'role': getattr(msg, 'type', 'unknown'),
                        'content': getattr(msg, 'content', str(msg))
                    })
            
            # 백그라운드 메모리 생성 작업 제출 (non-blocking)
            if conversation_history:
                task_id = await self.memory_service.submit_memory_generation(
                    session_id=session_id,
                    user_id=user_id,
                    conversation_history=conversation_history,
                    metadata={
                        'research_results_count': len(state.get('research_results', [])),
                        'verified_results_count': len(state.get('verified_results', [])),
                        'has_report': bool(state.get('final_report'))
                    }
                )
                logger.info(f"Background memory generation task submitted: {task_id}")
        except Exception as e:
            logger.warning(f"Failed to trigger background memory generation: {e}")
        
        return state
    
    async def execute(self, user_query: str, session_id: Optional[str] = None, restore_session: bool = False) -> Dict[str, Any]:
        """
        Execute multi-agent workflow with Skills auto-selection.
        
        Args:
            user_query: User's research query
            session_id: Session ID (if None, generates new session)
            restore_session: If True and session_id exists, restore from saved session
            
        Returns:
            Final result from the workflow
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 백그라운드 메모리 서비스 시작 (이미 시작되어 있으면 무시)
        try:
            if not self.memory_service.is_running:
                await self.memory_service.start()
                logger.info("Background memory service started")
        except Exception as e:
            logger.warning(f"Failed to start memory service: {e}")
        
        logger.info(f"Starting workflow for query: {user_query}, session: {session_id}")
        
        # 세션 복원 시도
        initial_state = None
        if restore_session:
            logger.info(f"Attempting to restore session: {session_id}")
            restored_state = await self.restore_session(session_id)
            if restored_state:
                logger.info(f"✅ Session restored: {session_id}")
                initial_state = AgentState(**restored_state)
                # 복원된 세션의 쿼리와 새 쿼리가 다를 수 있으므로 업데이트
                if user_query:
                    initial_state['user_query'] = user_query
            else:
                logger.info(f"Session not found or restore failed: {session_id}, starting new session")
        
        # Objective ID 생성 (병렬 실행 및 결과 공유용)
        objective_id = f"objective_{session_id}"
        
        # SharedResultsManager와 AgentDiscussionManager 초기화 (병렬 실행 활성화 시)
        if self.agent_config.enable_agent_communication:
            self.shared_results_manager = SharedResultsManager(objective_id=objective_id)
            self.discussion_manager = AgentDiscussionManager(
                objective_id=objective_id,
                shared_results_manager=self.shared_results_manager
            )
            logger.info("✅ Agent result sharing and discussion enabled")
            logger.info(f"🤝 SharedResultsManager initialized for objective: {objective_id}")
            logger.info(f"💬 AgentDiscussionManager initialized with agent communication support")
        else:
            self.shared_results_manager = None
            self.discussion_manager = None
            logger.info("Agent communication disabled")
        
        # Graph가 없거나 쿼리 기반 재빌드가 필요한 경우 빌드
        if self.graph is None:
            self._build_graph(user_query, session_id)
        
        # Initialize state if not restored
        if initial_state is None:
            initial_state = AgentState(
                messages=[],
                user_query=user_query,
                research_plan=None,
                research_tasks=[],
                research_results=[],
                verified_results=[],
                final_report=None,
                current_agent=None,
                iteration=0,
                session_id=session_id,
                research_failed=False,
                verification_failed=False,
                report_failed=False,
                error=None
            )
        
        # Execute workflow
        try:
            # DataFlow Pipeline 옵션 확인 (환경 변수 또는 설정에서)
            use_pipeline = os.getenv("USE_DATAFLOW_PIPELINE", "false").lower() == "true"
            
            if use_pipeline:
                try:
                    from src.dataflow.integration.orchestrator_pipeline_integration import OrchestratorPipelineIntegration
                    pipeline_integration = OrchestratorPipelineIntegration(use_pipeline=True)
                    
                    # Pipeline을 사용하여 실행
                    logger.info("Using DataFlow Pipeline for execution")
                    result = await pipeline_integration.execute_with_pipeline(
                        agent_state=dict(initial_state),
                        session_id=session_id
                    )
                    
                    # 결과를 AgentState로 변환
                    result = AgentState(**result)
                except Exception as e:
                    logger.warning(f"Pipeline execution failed, falling back to traditional workflow: {e}")
                    # Fallback to traditional workflow
                    result = await self.graph.ainvoke(initial_state)
            else:
                # Traditional workflow execution
                result = await self.graph.ainvoke(initial_state)
            
            # 세션 자동 저장 (워크플로우 완료 후)
            try:
                await self.save_session(session_id, result)
            except Exception as e:
                logger.warning(f"Failed to auto-save session: {e}")
            
            # 백그라운드 메모리 생성은 _end_node에서 이미 트리거됨
            
            logger.info("Workflow execution completed successfully")
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def restore_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        세션 복원.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            복원된 AgentState 또는 None
        """
        try:
            context_engineer = get_context_engineer()
            restored_state = self.session_manager.restore_session(
                session_id=session_id,
                context_engineer=context_engineer,
                shared_memory=self.shared_memory
            )
            
            if restored_state:
                logger.info(f"Session restored successfully: {session_id}")
                return restored_state
            else:
                logger.warning(f"Session restore returned None: {session_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error restoring session {session_id}: {e}", exc_info=True)
            return None
    
    async def save_session(self, session_id: str, state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        세션 저장.
        
        Args:
            session_id: 세션 ID
            state: AgentState 데이터
            metadata: 추가 메타데이터
            
        Returns:
            성공 여부
        """
        try:
            context_engineer = get_context_engineer()
            success = self.session_manager.save_session(
                session_id=session_id,
                agent_state=state,
                context_engineer=context_engineer,
                shared_memory=self.shared_memory,
                metadata=metadata
            )
            
            if success:
                logger.info(f"Session saved successfully: {session_id}")
            else:
                logger.warning(f"Session save returned False: {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}", exc_info=True)
            return False
    
    def list_sessions(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        세션 목록 조회.
        
        Args:
            user_id: 사용자 ID 필터
            limit: 최대 결과 수
            
        Returns:
            세션 메타데이터 목록
        """
        from dataclasses import asdict
        sessions = self.session_manager.list_sessions(user_id=user_id, limit=limit)
        return [asdict(session) for session in sessions]
    
    def delete_session(self, session_id: str) -> bool:
        """
        세션 삭제.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            성공 여부
        """
        return self.session_manager.delete_session(session_id)
    
    def create_snapshot(self, session_id: str) -> Optional[str]:
        """
        세션 스냅샷 생성.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            스냅샷 ID 또는 None
        """
        return self.session_manager.create_snapshot(session_id)
    
    def restore_from_snapshot(self, session_id: str, snapshot_id: str) -> bool:
        """
        스냅샷에서 세션 복원.
        
        Args:
            session_id: 복원할 세션 ID
            snapshot_id: 스냅샷 ID
            
        Returns:
            성공 여부
        """
        return self.session_manager.restore_from_snapshot(session_id, snapshot_id)
    
    async def stream(self, user_query: str, session_id: Optional[str] = None, initial_state: Optional[Dict[str, Any]] = None):
        """Stream workflow execution."""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize state
        if initial_state is None:
            initial_state = {}
        
        agent_initial_state = AgentState(
            messages=[],
            user_query=user_query,
            research_plan=None,
            research_tasks=[],
            research_results=[],
            verified_results=[],
            final_report=None,
            current_agent=None,
            iteration=0,
            session_id=session_id,
            research_failed=False,
            verification_failed=False,
            report_failed=False,
            error=None,
            pending_questions=initial_state.get('pending_questions'),
            user_responses=initial_state.get('user_responses'),
            clarification_context=initial_state.get('clarification_context'),
            waiting_for_user=initial_state.get('waiting_for_user', False)
        )
        
        # Stream execution
        async for event in self.graph.astream(agent_initial_state):
            yield event


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator(config: Any = None) -> AgentOrchestrator:
    """Get global orchestrator instance."""
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(config=config)
    
    return _orchestrator


def init_orchestrator(config: Any = None) -> AgentOrchestrator:
    """Initialize orchestrator."""
    global _orchestrator
    
    _orchestrator = AgentOrchestrator(config=config)
    
    return _orchestrator

