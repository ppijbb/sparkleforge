"""
DAG Visualizer for Task Execution Tracking

ROMA의 DAG 시각화 아이디어를 SparkleForge 방식으로 구현.
태스크 의존성 그래프를 시각화하고 실행 상태를 실시간으로 추적합니다.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """태스크 실행 상태."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DAGVisualizer:
    """
    DAG 시각화 및 실행 추적 클래스.
    
    태스크 의존성 그래프를 시각화하고 실행 상태를 실시간으로 추적합니다.
    스트리밍 파이프라인과 통합하여 실시간 업데이트를 제공합니다.
    """
    
    def __init__(self):
        """초기화."""
        self.task_states: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def initialize(self, tasks: List[Dict[str, Any]]):
        """
        DAG 초기화.
        
        Args:
            tasks: 태스크 리스트
        """
        self.start_time = datetime.now()
        self.task_states = {}
        
        for task in tasks:
            task_id = task.get('task_id', 'unknown')
            self.task_states[task_id] = {
                'task': task,
                'status': TaskStatus.PENDING,
                'started_at': None,
                'completed_at': None,
                'execution_time': None,
                'dependencies': task.get('dependencies', []),
                'dependents': [],
                'error': None
            }
        
        # 의존성 관계 구축
        for task_id, state in self.task_states.items():
            for dep_id in state['dependencies']:
                if dep_id in self.task_states:
                    if 'dependents' not in self.task_states[dep_id]:
                        self.task_states[dep_id]['dependents'] = []
                    self.task_states[dep_id]['dependents'].append(task_id)
        
        logger.info(f"DAG Visualizer initialized with {len(self.task_states)} tasks")
    
    def mark_task_started(self, task_id: str):
        """태스크 시작 표시."""
        if task_id in self.task_states:
            self.task_states[task_id]['status'] = TaskStatus.RUNNING
            self.task_states[task_id]['started_at'] = datetime.now()
            
            # 실행 이벤트 기록
            self.execution_history.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'task_started',
                'task_id': task_id,
                'task_name': self.task_states[task_id]['task'].get('name', '')
            })
    
    def mark_task_completed(self, task_id: str, result: Optional[Dict[str, Any]] = None):
        """태스크 완료 표시."""
        if task_id in self.task_states:
            state = self.task_states[task_id]
            state['status'] = TaskStatus.COMPLETED
            state['completed_at'] = datetime.now()
            
            if state['started_at']:
                state['execution_time'] = (state['completed_at'] - state['started_at']).total_seconds()
            
            # 실행 이벤트 기록
            self.execution_history.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'task_completed',
                'task_id': task_id,
                'task_name': state['task'].get('name', ''),
                'execution_time': state['execution_time'],
                'result': result
            })
    
    def mark_task_failed(self, task_id: str, error: str):
        """태스크 실패 표시."""
        if task_id in self.task_states:
            state = self.task_states[task_id]
            state['status'] = TaskStatus.FAILED
            state['completed_at'] = datetime.now()
            state['error'] = error
            
            if state['started_at']:
                state['execution_time'] = (state['completed_at'] - state['started_at']).total_seconds()
            
            # 실행 이벤트 기록
            self.execution_history.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'task_failed',
                'task_id': task_id,
                'task_name': state['task'].get('name', ''),
                'error': error
            })
    
    def get_dag_summary(self) -> Dict[str, Any]:
        """DAG 실행 요약 반환."""
        if not self.task_states:
            return {}
        
        total_tasks = len(self.task_states)
        completed = sum(1 for s in self.task_states.values() if s['status'] == TaskStatus.COMPLETED)
        running = sum(1 for s in self.task_states.values() if s['status'] == TaskStatus.RUNNING)
        failed = sum(1 for s in self.task_states.values() if s['status'] == TaskStatus.FAILED)
        pending = sum(1 for s in self.task_states.values() if s['status'] == TaskStatus.PENDING)
        
        # 실행 시간 계산
        total_execution_time = sum(
            s.get('execution_time', 0) or 0 
            for s in self.task_states.values() 
            if s.get('execution_time')
        )
        
        # 의존성 대기 시간 계산
        dependency_wait_times = []
        for task_id, state in self.task_states.items():
            if state['started_at'] and state['dependencies']:
                # 가장 늦게 완료된 의존성 찾기
                max_dep_completion = None
                for dep_id in state['dependencies']:
                    if dep_id in self.task_states:
                        dep_completion = self.task_states[dep_id].get('completed_at')
                        if dep_completion:
                            if max_dep_completion is None or dep_completion > max_dep_completion:
                                max_dep_completion = dep_completion
                
                if max_dep_completion and state['started_at']:
                    wait_time = (state['started_at'] - max_dep_completion).total_seconds()
                    if wait_time > 0:
                        dependency_wait_times.append(wait_time)
        
        avg_dependency_wait = sum(dependency_wait_times) / len(dependency_wait_times) if dependency_wait_times else 0
        
        return {
            'total_tasks': total_tasks,
            'completed': completed,
            'running': running,
            'failed': failed,
            'pending': pending,
            'success_rate': completed / total_tasks if total_tasks > 0 else 0,
            'total_execution_time': total_execution_time,
            'average_dependency_wait_time': avg_dependency_wait,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """태스크 상태 반환."""
        if task_id in self.task_states:
            state = self.task_states[task_id].copy()
            state['status'] = state['status'].value if isinstance(state['status'], TaskStatus) else state['status']
            return state
        return None
    
    def get_all_task_statuses(self) -> Dict[str, Dict[str, Any]]:
        """모든 태스크 상태 반환."""
        result = {}
        for task_id, state in self.task_states.items():
            result[task_id] = self.get_task_status(task_id)
        return result
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """실행 이벤트 히스토리 반환."""
        return self.execution_history.copy()
    
    def finalize(self):
        """DAG 실행 종료."""
        self.end_time = datetime.now()
        logger.info(f"DAG Visualizer finalized. Duration: {(self.end_time - self.start_time).total_seconds() if self.start_time else 0:.2f}s")
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """
        시각화용 데이터 반환.
        
        Returns:
            DAG 시각화에 필요한 모든 데이터
        """
        return {
            'nodes': [
                {
                    'id': task_id,
                    'label': state['task'].get('name', task_id),
                    'status': state['status'].value if isinstance(state['status'], TaskStatus) else str(state['status']),
                    'type': state['task'].get('type', 'research'),
                    'started_at': state['started_at'].isoformat() if state['started_at'] else None,
                    'completed_at': state['completed_at'].isoformat() if state['completed_at'] else None,
                    'execution_time': state.get('execution_time'),
                    'error': state.get('error')
                }
                for task_id, state in self.task_states.items()
            ],
            'edges': [
                {
                    'from': dep_id,
                    'to': task_id,
                    'type': 'dependency'
                }
                for task_id, state in self.task_states.items()
                for dep_id in state['dependencies']
                if dep_id in self.task_states
            ],
            'summary': self.get_dag_summary(),
            'execution_history': self.execution_history[-50:]  # 최근 50개 이벤트만
        }


# 전역 인스턴스
_visualizer: Optional[DAGVisualizer] = None


def get_dag_visualizer() -> DAGVisualizer:
    """전역 DAGVisualizer 인스턴스 반환."""
    global _visualizer
    if _visualizer is None:
        _visualizer = DAGVisualizer()
    return _visualizer
