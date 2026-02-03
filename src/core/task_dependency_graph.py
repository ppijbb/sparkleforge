"""
Task Dependency Graph for Smart Parallel Execution

ROMA의 DAG 기반 의존성 분석 아이디어를 SparkleForge 방식으로 구현.
NetworkX를 사용하여 태스크 간 의존성을 관리하고 병렬 실행을 최적화합니다.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import deque, defaultdict

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger = logging.getLogger(__name__)
    logger.warning("networkx not available. Install with: pip install networkx")

logger = logging.getLogger(__name__)


class TaskDependencyGraph:
    """
    태스크 의존성 그래프 관리 클래스.
    
    NetworkX 기반 DAG를 사용하여:
    - 태스크 간 의존성 분석
    - 위상 정렬로 실행 순서 결정
    - 병렬 실행 가능 그룹 식별
    - 동적 스케줄링 지원
    """
    
    def __init__(self, tasks: List[Dict[str, Any]]):
        """
        태스크 목록으로부터 의존성 그래프 구축.
        
        Args:
            tasks: 태스크 딕셔너리 리스트 (task_id, dependencies 필드 포함)
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required. Install with: pip install networkx")
        
        self.tasks = {task.get('task_id', f'task_{i}'): task for i, task in enumerate(tasks)}
        self.graph = nx.DiGraph()
        self._build_graph()
        
        # 실행 상태 추적
        self.completed_tasks: Set[str] = set()
        self.running_tasks: Set[str] = set()
        
        logger.info(f"TaskDependencyGraph initialized with {len(self.tasks)} tasks")
    
    def _build_graph(self):
        """태스크 목록으로부터 DAG 구축."""
        # 노드 추가
        for task_id, task in self.tasks.items():
            self.graph.add_node(task_id, **task)
        
        # 엣지 추가 (의존성)
        for task_id, task in self.tasks.items():
            dependencies = task.get('dependencies', [])
            if isinstance(dependencies, str):
                dependencies = [dependencies]
            
            for dep_id in dependencies:
                if dep_id in self.tasks:
                    self.graph.add_edge(dep_id, task_id)
                    logger.debug(f"  Dependency: {dep_id} -> {task_id}")
                else:
                    logger.warning(f"  ⚠️ Unknown dependency: {dep_id} (referenced by {task_id})")
        
        # 사이클 검증
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            logger.error(f"  ❌ Circular dependencies detected: {cycles}")
            # 사이클이 있어도 계속 진행 (경고만)
        
        logger.info(f"  Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def get_execution_order(self) -> List[str]:
        """
        위상 정렬로 실행 순서 반환.
        
        Returns:
            의존성을 고려한 태스크 실행 순서 리스트
        """
        try:
            # 위상 정렬
            execution_order = list(nx.topological_sort(self.graph))
            logger.info(f"  Execution order determined: {len(execution_order)} tasks")
            return execution_order
        except nx.NetworkXError as e:
            # 사이클이 있는 경우
            logger.warning(f"  ⚠️ Cannot perform topological sort (cycle detected): {e}")
            # 노드 순서대로 반환 (의존성 무시)
            return list(self.graph.nodes())
    
    def get_parallel_groups(self) -> List[List[str]]:
        """
        의존성이 없는 태스크 그룹 반환.
        
        각 그룹은 동시에 실행 가능한 태스크들의 집합입니다.
        
        Returns:
            병렬 실행 가능한 태스크 그룹 리스트
        """
        parallel_groups = []
        processed = set()
        
        # 위상 정렬 순서로 처리
        execution_order = self.get_execution_order()
        
        for task_id in execution_order:
            if task_id in processed:
                continue
            
            # 현재 태스크의 모든 의존성이 완료되었는지 확인
            dependencies = list(self.graph.predecessors(task_id))
            if all(dep in self.completed_tasks for dep in dependencies):
                # 의존성이 해결된 태스크들을 그룹화
                current_group = [task_id]
                processed.add(task_id)
                
                # 다른 의존성이 해결된 태스크들 찾기
                for other_id in execution_order:
                    if other_id in processed or other_id == task_id:
                        continue
                    
                    other_deps = list(self.graph.predecessors(other_id))
                    if all(dep in self.completed_tasks for dep in other_deps):
                        current_group.append(other_id)
                        processed.add(other_id)
                
                if current_group:
                    parallel_groups.append(current_group)
        
        logger.info(f"  Identified {len(parallel_groups)} parallel groups")
        return parallel_groups
    
    def get_ready_tasks(self) -> List[str]:
        """
        현재 실행 가능한 태스크 반환 (의존성이 모두 해결된 태스크).
        
        Returns:
            실행 가능한 태스크 ID 리스트
        """
        ready_tasks = []
        
        for task_id in self.graph.nodes():
            if task_id in self.completed_tasks or task_id in self.running_tasks:
                continue
            
            # 의존성 확인
            dependencies = list(self.graph.predecessors(task_id))
            if all(dep in self.completed_tasks for dep in dependencies):
                ready_tasks.append(task_id)
        
        return ready_tasks
    
    def mark_completed(self, task_id: str):
        """태스크 완료 표시."""
        if task_id in self.running_tasks:
            self.running_tasks.remove(task_id)
        self.completed_tasks.add(task_id)
        logger.debug(f"  Task {task_id} marked as completed")
    
    def mark_running(self, task_id: str):
        """태스크 실행 중 표시."""
        self.running_tasks.add(task_id)
        logger.debug(f"  Task {task_id} marked as running")
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """태스크 정보 반환."""
        return self.tasks.get(task_id)
    
    def get_dependencies(self, task_id: str) -> List[str]:
        """태스크의 의존성 리스트 반환."""
        return list(self.graph.predecessors(task_id))
    
    def get_dependents(self, task_id: str) -> List[str]:
        """태스크에 의존하는 다른 태스크 리스트 반환."""
        return list(self.graph.successors(task_id))
    
    def get_execution_levels(self) -> List[List[str]]:
        """
        실행 레벨별로 태스크 그룹화.
        
        같은 레벨의 태스크는 병렬 실행 가능합니다.
        
        Returns:
            레벨별 태스크 그룹 리스트
        """
        levels = []
        remaining = set(self.graph.nodes())
        completed = set()
        
        while remaining:
            # 현재 레벨: 의존성이 모두 해결된 태스크들
            current_level = []
            for task_id in remaining:
                dependencies = list(self.graph.predecessors(task_id))
                if all(dep in completed for dep in dependencies):
                    current_level.append(task_id)
            
            if not current_level:
                # 사이클이 있거나 의존성 문제
                logger.warning(f"  ⚠️ Cannot determine next level. Remaining tasks: {remaining}")
                # 남은 태스크를 모두 현재 레벨에 추가
                current_level = list(remaining)
            
            levels.append(current_level)
            completed.update(current_level)
            remaining -= set(current_level)
        
        logger.info(f"  Execution levels: {len(levels)} levels")
        return levels
    
    def get_statistics(self) -> Dict[str, Any]:
        """그래프 통계 정보 반환."""
        return {
            "total_tasks": self.graph.number_of_nodes(),
            "total_dependencies": self.graph.number_of_edges(),
            "completed_tasks": len(self.completed_tasks),
            "running_tasks": len(self.running_tasks),
            "pending_tasks": self.graph.number_of_nodes() - len(self.completed_tasks) - len(self.running_tasks),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "execution_levels": len(self.get_execution_levels())
        }
