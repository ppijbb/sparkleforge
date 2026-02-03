#!/usr/bin/env python3
"""
Agent Visualizer Component for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade Streamlit 컴포넌트로 에이전트 활동을 실시간 시각화.
각 에이전트의 상태, 작업 타임라인, 병렬 작업 진행 상황을
직관적으로 표시합니다.

2025년 10월 최신 기술 스택:
- Streamlit 1.39+ with custom components
- Plotly 5.18+ for interactive charts
- AgGrid for data tables
- Real-time updates with st.empty() + st.rerun()
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import time
import threading
from collections import defaultdict, deque
import logging

# Import streaming manager
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.streaming_manager import get_streaming_manager, EventType, AgentStatus
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class AgentVisualizer:
    """
    Production-grade 에이전트 시각화 컴포넌트.
    
    Features:
    - 실시간 에이전트 상태 표시
    - 작업 타임라인 시각화
    - 병렬 작업 진행 상황 표시
    - 에이전트 간 데이터 흐름 다이어그램
    - 창의성 에이전트 특별 표시
    """
    
    def __init__(self):
        """에이전트 비주얼라이저 초기화."""
        self.streaming_manager = get_streaming_manager()
        self.update_interval = 1.0  # 1초마다 업데이트
        self.max_timeline_points = 100  # 타임라인 최대 포인트 수
        
        # 세션 상태 초기화
        if 'agent_timeline' not in st.session_state:
            st.session_state.agent_timeline = defaultdict(list)
        if 'agent_activities' not in st.session_state:
            st.session_state.agent_activities = deque(maxlen=50)
        if 'workflow_start_time' not in st.session_state:
            st.session_state.workflow_start_time = None
    
    def render_live_dashboard(self, workflow_id: str) -> None:
        """
        실시간 대시보드를 렌더링합니다.
        
        Args:
            workflow_id: 워크플로우 ID
        """
        st.subheader("🔴 Live Research Dashboard")
        
        # 워크플로우 상태 가져오기
        workflow_status = self.streaming_manager.get_workflow_status(workflow_id)
        
        if not workflow_status.get('agents'):
            st.info("No active agents in this workflow")
            return
        
        # 상단 메트릭 카드
        self._render_metrics_cards(workflow_status)
        
        # 에이전트 상태 그리드
        self._render_agent_status_grid(workflow_status['agents'])
        
        # 실시간 활동 피드
        self._render_activity_feed()
        
        # 타임라인 차트
        self._render_timeline_chart(workflow_id)
        
        # 창의성 인사이트 (있는 경우)
        self._render_creative_insights(workflow_id)
    
    def _render_metrics_cards(self, workflow_status: Dict[str, Any]) -> None:
        """메트릭 카드를 렌더링합니다."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Active Agents",
                value=workflow_status.get('total_agents', 0),
                delta=None
            )
        
        with col2:
            completed = workflow_status.get('completed_agents', 0)
            total = workflow_status.get('total_agents', 1)
            completion_rate = (completed / total) * 100 if total > 0 else 0
            st.metric(
                label="Completion Rate",
                value=f"{completion_rate:.1f}%",
                delta=f"{completed}/{total}"
            )
        
        with col3:
            overall_progress = workflow_status.get('overall_progress', 0)
            st.metric(
                label="Overall Progress",
                value=f"{overall_progress:.1f}%",
                delta=None
            )
        
        with col4:
            # 스트리밍 통계
            stats = self.streaming_manager.get_stats()
            events_per_sec = stats.get('events_per_second', 0)
            st.metric(
                label="Events/sec",
                value=f"{events_per_sec:.1f}",
                delta=None
            )
    
    def _render_agent_status_grid(self, agents: List[Dict[str, Any]]) -> None:
        """에이전트 상태 그리드를 렌더링합니다."""
        st.subheader("🤖 Agent Status")
        
        # 에이전트를 상태별로 그룹화
        status_groups = defaultdict(list)
        for agent in agents:
            status = agent.get('status', 'idle')
            status_groups[status].append(agent)
        
        # 상태별로 컬럼 생성
        if status_groups:
            cols = st.columns(len(status_groups))
            for i, (status, agent_list) in enumerate(status_groups.items()):
                with cols[i]:
                    self._render_status_column(status, agent_list)
    
    def _render_status_column(self, status: str, agents: List[Dict[str, Any]]) -> None:
        """상태별 컬럼을 렌더링합니다."""
        # 상태별 아이콘과 색상
        status_config = {
            'working': {'icon': '⚡', 'color': '#FFA500', 'bg_color': '#FFF3CD'},
            'waiting': {'icon': '⏳', 'color': '#6C757D', 'bg_color': '#F8F9FA'},
            'completed': {'icon': '✅', 'color': '#28A745', 'bg_color': '#D4EDDA'},
            'error': {'icon': '❌', 'color': '#DC3545', 'bg_color': '#F8D7DA'},
            'creating': {'icon': '💡', 'color': '#9C27B0', 'bg_color': '#F3E5F5'},
            'idle': {'icon': '😴', 'color': '#6C757D', 'bg_color': '#F8F9FA'}
        }
        
        config = status_config.get(status, status_config['idle'])
        
        st.markdown(f"**{config['icon']} {status.title()} ({len(agents)})**")
        
        for agent in agents:
            with st.container():
                # 에이전트 카드
                st.markdown(f"""
                <div style="
                    background-color: {config['bg_color']};
                    border-left: 4px solid {config['color']};
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                ">
                    <strong>{agent.get('agent_id', 'Unknown')}</strong><br>
                    <small>{agent.get('current_task', 'No task')}</small><br>
                    <small>Progress: {agent.get('progress_percentage', 0):.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_activity_feed(self) -> None:
        """실시간 활동 피드를 렌더링합니다."""
        st.subheader("📋 Recent Activities")
        
        # 최근 활동 가져오기
        activities = list(st.session_state.agent_activities)[-10:]  # 최근 10개
        
        if not activities:
            st.info("No recent activities")
            return
        
        # 활동 타임라인
        for activity in reversed(activities):
            self._render_activity_item(activity)
    
    def _render_activity_item(self, activity: Dict[str, Any]) -> None:
        """개별 활동 아이템을 렌더링합니다."""
        timestamp = activity.get('timestamp', '')
        agent_id = activity.get('agent_id', 'Unknown')
        action = activity.get('action', 'Unknown action')
        status = activity.get('status', 'unknown')
        
        # 시간 포맷팅
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            time_str = dt.strftime('%H:%M:%S')
        except:
            time_str = str(timestamp)
        
        # 상태별 아이콘
        status_icons = {
            'working': '⚡',
            'waiting': '⏳',
            'completed': '✅',
            'error': '❌',
            'creating': '💡'
        }
        icon = status_icons.get(status, '📝')
        
        st.markdown(f"""
        <div style="
            padding: 8px;
            margin: 2px 0;
            border-left: 3px solid #007bff;
            background-color: #f8f9fa;
        ">
            <small style="color: #6c757d;">{time_str}</small><br>
            <strong>{icon} {agent_id}</strong>: {action}
        </div>
        """, unsafe_allow_html=True)
    
    def _render_timeline_chart(self, workflow_id: str) -> None:
        """타임라인 차트를 렌더링합니다."""
        st.subheader("📈 Progress Timeline")
        
        # 타임라인 데이터 준비
        timeline_data = st.session_state.agent_timeline.get(workflow_id, [])
        
        if not timeline_data:
            st.info("No timeline data available")
            return
        
        # DataFrame 생성
        df = pd.DataFrame(timeline_data)
        
        if df.empty:
            st.info("No timeline data to display")
            return
        
        # Plotly 타임라인 차트 생성
        fig = go.Figure()
        
        # 에이전트별로 라인 추가
        agents = df['agent_id'].unique()
        colors = px.colors.qualitative.Set3[:len(agents)]
        
        for i, agent_id in enumerate(agents):
            agent_data = df[df['agent_id'] == agent_id].sort_values('timestamp')
            
            fig.add_trace(go.Scatter(
                x=agent_data['timestamp'],
                y=agent_data['progress'],
                mode='lines+markers',
                name=agent_id,
                line=dict(color=colors[i], width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>{agent_id}</b><br>' +
                             'Time: %{x}<br>' +
                             'Progress: %{y:.1f}%<br>' +
                             'Task: %{customdata}<extra></extra>',
                customdata=agent_data['task']
            ))
        
        # 차트 레이아웃 설정
        fig.update_layout(
            title="Agent Progress Over Time",
            xaxis_title="Time",
            yaxis_title="Progress (%)",
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400
        )
        
        # 차트 표시
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_creative_insights(self, workflow_id: str) -> None:
        """창의성 인사이트를 렌더링합니다."""
        # 창의성 인사이트가 있는지 확인
        creative_insights = []
        for activity in st.session_state.agent_activities:
            if activity.get('event_type') == 'creative_insight':
                creative_insights.append(activity)
        
        if not creative_insights:
            return
        
        st.subheader("💡 Creative Insights")
        
        for insight in creative_insights[-5:]:  # 최근 5개
            self._render_creative_insight_item(insight)
    
    def _render_creative_insight_item(self, insight: Dict[str, Any]) -> None:
        """개별 창의성 인사이트 아이템을 렌더링합니다."""
        timestamp = insight.get('timestamp', '')
        agent_id = insight.get('agent_id', 'Unknown')
        insight_text = insight.get('data', {}).get('insight', 'No insight')
        confidence = insight.get('data', {}).get('confidence', 0.0)
        concepts = insight.get('data', {}).get('related_concepts', [])
        
        # 시간 포맷팅
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            time_str = dt.strftime('%H:%M:%S')
        except:
            time_str = str(timestamp)
        
        # 신뢰도에 따른 색상
        confidence_color = '#28A745' if confidence > 0.7 else '#FFC107' if confidence > 0.4 else '#DC3545'
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong>💡 {agent_id}</strong>
                <small>Confidence: <span style="color: {confidence_color};">{confidence:.1%}</span></small>
            </div>
            <p style="margin: 10px 0; font-size: 14px;">{insight_text}</p>
            <div style="margin-top: 10px;">
                <small>Related: {', '.join(concepts[:3])}{'...' if len(concepts) > 3 else ''}</small>
                <br><small style="opacity: 0.8;">{time_str}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def update_agent_activity(self, event_data: Dict[str, Any]) -> None:
        """에이전트 활동을 업데이트합니다."""
        activity = {
            'timestamp': event_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            'agent_id': event_data.get('agent_id', 'Unknown'),
            'action': event_data.get('data', {}).get('action', 'Unknown action'),
            'status': event_data.get('data', {}).get('status', 'unknown'),
            'event_type': event_data.get('event_type', 'unknown')
        }
        
        st.session_state.agent_activities.append(activity)
    
    def update_timeline_data(self, workflow_id: str, agent_id: str, progress: float, task: str) -> None:
        """타임라인 데이터를 업데이트합니다."""
        timeline_point = {
            'timestamp': datetime.now(timezone.utc),
            'agent_id': agent_id,
            'progress': progress,
            'task': task
        }
        
        st.session_state.agent_timeline[workflow_id].append(timeline_point)
        
        # 최대 포인트 수 제한
        if len(st.session_state.agent_timeline[workflow_id]) > self.max_timeline_points:
            st.session_state.agent_timeline[workflow_id] = st.session_state.agent_timeline[workflow_id][-self.max_timeline_points:]
    
    def render_agent_flow_diagram(self, workflow_id: str) -> None:
        """에이전트 간 데이터 흐름 다이어그램을 렌더링합니다."""
        st.subheader("🔄 Agent Flow Diagram")
        
        # 워크플로우 상태 가져오기
        workflow_status = self.streaming_manager.get_workflow_status(workflow_id)
        agents = workflow_status.get('agents', [])
        
        if not agents:
            st.info("No agents to display")
            return
        
        # Sankey 다이어그램 생성
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[agent['agent_id'] for agent in agents],
                color=[self._get_agent_color(agent['status']) for agent in agents]
            ),
            link=dict(
                source=[0, 1, 2],  # 간단한 연결 (실제로는 더 복잡한 로직 필요)
                target=[1, 2, 3],
                value=[10, 20, 30]
            )
        )])
        
        fig.update_layout(
            title_text="Agent Data Flow",
            font_size=10,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_agent_color(self, status: str) -> str:
        """에이전트 상태에 따른 색상을 반환합니다."""
        color_map = {
            'working': '#FFA500',
            'waiting': '#6C757D',
            'completed': '#28A745',
            'error': '#DC3545',
            'creating': '#9C27B0',
            'idle': '#6C757D'
        }
        return color_map.get(status, '#6C757D')
    
    def start_auto_refresh(self, workflow_id: str) -> None:
        """자동 새로고침을 시작합니다."""
        if st.button("🔄 Refresh Dashboard", key=f"refresh_{workflow_id}"):
            st.rerun()
        
        # 자동 새로고침 (5초마다)
        if st.checkbox("Auto-refresh (5s)", key=f"auto_refresh_{workflow_id}"):
            time.sleep(5)
            st.rerun()


# Streamlit 컴포넌트 함수들
def render_agent_dashboard(workflow_id: str) -> None:
    """에이전트 대시보드를 렌더링합니다."""
    visualizer = AgentVisualizer()
    visualizer.render_live_dashboard(workflow_id)


def render_agent_timeline(workflow_id: str) -> None:
    """에이전트 타임라인을 렌더링합니다."""
    visualizer = AgentVisualizer()
    visualizer.render_timeline_chart(workflow_id)


def render_agent_flow(workflow_id: str) -> None:
    """에이전트 플로우 다이어그램을 렌더링합니다."""
    visualizer = AgentVisualizer()
    visualizer.render_agent_flow_diagram(workflow_id)
