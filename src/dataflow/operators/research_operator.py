"""
Research Operator

ResearchAgent의 연구 실행 로직을 Operator로 추출합니다.
"""

import logging
import asyncio
from typing import Any, Dict, Optional, Literal

from .base_operator import SparkleForgeOperatorABC
from ..storage.agent_storage_adapter import DataFlowStorage

logger = logging.getLogger(__name__)


class ResearchOperator(SparkleForgeOperatorABC):
    """
    연구 실행 Operator.
    
    ResearchAgent의 로직을 Operator로 변환하여 Pipeline에서 사용할 수 있도록 합니다.
    
    input_key: research_tasks (연구 태스크 목록)
    output_key: research_results (연구 결과 목록)
    """
    
    def __init__(
        self,
        research_agent=None,
        max_concurrent_tasks: int = 5,
        enable_async: bool = True,
    ):
        """
        초기화.
        
        Args:
            research_agent: ResearchAgent 인스턴스 (None이면 내부에서 생성)
            max_concurrent_tasks: 최대 동시 실행 태스크 수
            enable_async: 비동기 실행 여부
        """
        super().__init__()
        self.research_agent = research_agent
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_async = enable_async
        
        # ResearchAgent가 없으면 지연 로딩
        self._agent_initialized = False
    
    def _ensure_agent_initialized(self):
        """ResearchAgent가 초기화되었는지 확인하고 필요시 초기화합니다."""
        if self._agent_initialized:
            return
        
        if self.research_agent is None:
            try:
                from src.agents.research_agent import ResearchAgent
                self.research_agent = ResearchAgent()
                self.logger.info("ResearchAgent initialized for ResearchOperator")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ResearchAgent: {e}")
                self.research_agent = None
        
        self._agent_initialized = True
    
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "research_tasks",
        output_key: str = "research_results",
        **kwargs
    ) -> Optional[str]:
        """
        연구를 실행합니다.
        
        Args:
            storage: DataFlowStorage 인스턴스
            input_key: 입력 키 (기본값: "research_tasks")
            output_key: 출력 키 (기본값: "research_results")
            **kwargs: 추가 파라미터
            
        Returns:
            실행 결과 메시지
        """
        self.logger.info(f"Executing research from '{input_key}' to '{output_key}'")
        
        # ResearchAgent 초기화 확인
        self._ensure_agent_initialized()
        
        if self.research_agent is None:
            self.logger.error("ResearchAgent is not available")
            return "ResearchAgent is not available"
        
        # 입력 데이터 읽기
        df = storage.read("dataframe")
        
        if input_key not in df.columns:
            self.logger.warning(f"Input key '{input_key}' not found. Creating empty results.")
            df[output_key] = None
            storage.write(df)
            return "No research tasks found"
        
        # 각 행에서 연구 태스크 실행
        research_results = []
        
        for idx, row in df.iterrows():
            tasks = row.get(input_key, [])
            
            if not isinstance(tasks, list):
                tasks = [tasks] if tasks else []
            
            if not tasks:
                research_results.append(None)
                continue
            
            # 연구 태스크 실행
            if self.enable_async:
                # 비동기 실행
                result = self._execute_research_tasks_async(tasks, row)
            else:
                # 동기 실행
                result = self._execute_research_tasks_sync(tasks, row)
            
            research_results.append(result)
        
        # 결과를 DataFrame에 추가
        df[output_key] = research_results
        storage.write(df)
        
        non_null_count = sum(1 for r in research_results if r is not None)
        self.logger.info(f"Executed research for {non_null_count} tasks")
        
        return f"Executed research for {non_null_count} tasks"
    
    def _execute_research_tasks_sync(self, tasks: list, row: Any) -> list:
        """연구 태스크를 동기적으로 실행합니다."""
        results = []
        
        for task in tasks:
            try:
                if isinstance(task, dict):
                    task_id = task.get("task_id", task.get("id", "unknown"))
                    task_description = task.get("description", task.get("query", ""))
                    
                    self.logger.info(f"Executing research task: {task_id}")
                    
                    # ResearchAgent의 execute_research_task 호출
                    # 동기 함수이므로 asyncio.run 사용
                    try:
                        result = asyncio.run(
                            self.research_agent.execute_research_task(
                                task=task,
                                objective_id=task.get("objective_id", "default")
                            )
                        )
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Research task {task_id} failed: {e}")
                        results.append({
                            "task_id": task_id,
                            "status": "failed",
                            "error": str(e),
                        })
                elif isinstance(task, str):
                    # 문자열인 경우 간단한 검색으로 처리
                    self.logger.info(f"Executing research query: {task[:50]}...")
                    try:
                        result = asyncio.run(
                            self.research_agent.execute_research_task(
                                task={"description": task, "query": task},
                                objective_id="default"
                            )
                        )
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Research query failed: {e}")
                        results.append({
                            "query": task,
                            "status": "failed",
                            "error": str(e),
                        })
            except Exception as e:
                self.logger.error(f"Error processing research task: {e}")
                results.append({
                    "status": "error",
                    "error": str(e),
                })
        
        return results
    
    def _execute_research_tasks_async(self, tasks: list, row: Any) -> list:
        """연구 태스크를 비동기적으로 실행합니다."""
        # 동기 함수이므로 동기 실행으로 fallback
        return self._execute_research_tasks_sync(tasks, row)
    
    async def _execute_research_task_async(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """단일 연구 태스크를 비동기적으로 실행합니다."""
        if self.research_agent is None:
            return {
                "status": "error",
                "error": "ResearchAgent not available",
            }
        
        try:
            result = await self.research_agent.execute_research_task(
                task=task,
                objective_id=task.get("objective_id", "default")
            )
            return result
        except Exception as e:
            self.logger.error(f"Research task execution failed: {e}")
            return {
                "task_id": task.get("task_id", "unknown"),
                "status": "failed",
                "error": str(e),
            }








