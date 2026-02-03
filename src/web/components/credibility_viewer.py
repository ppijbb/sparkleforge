#!/usr/bin/env python3
"""
Credibility Viewer Component for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade 신뢰도 시각화 컴포넌트.
출처별 신뢰도 레이더 차트, 정보 신뢰도 히트맵,
검증 상태 배지를 통해 정보의 신뢰성을 직관적으로 표시합니다.

2025년 10월 최신 기술 스택:
- Streamlit 1.39+ with custom components
- Plotly 5.18+ for interactive charts
- Real-time credibility updates
- Production-grade visualization
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import json
import logging

# Import verification modules
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.verification.source_validator import SourceCredibility, DomainType, AuthorityLevel
from src.verification.fact_checker import FactCheckResult, FactStatus
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class CredibilityViewer:
    """
    Production-grade 신뢰도 시각화 컴포넌트.
    
    Features:
    - 출처별 신뢰도 레이더 차트
    - 정보 신뢰도 히트맵
    - 검증 상태 배지 (Verified, Partially Verified, Unverified)
    - 상충 정보 경고 표시
    """
    
    def __init__(self):
        """신뢰도 뷰어 초기화."""
        self.color_scheme = {
            'verified': '#28A745',
            'partially_verified': '#FFC107',
            'unverified': '#DC3545',
            'conflicting': '#FD7E14',
            'disputed': '#6F42C1'
        }
        
        self.domain_colors = {
            DomainType.ACADEMIC: '#1f77b4',
            DomainType.GOVERNMENT: '#ff7f0e',
            DomainType.NEWS: '#2ca02c',
            DomainType.BLOG: '#d62728',
            DomainType.SOCIAL: '#9467bd',
            DomainType.COMMERCIAL: '#8c564b',
            DomainType.UNKNOWN: '#7f7f7f'
        }
    
    def render_credibility_dashboard(self, sources: List[SourceCredibility], facts: List[FactCheckResult] = None) -> None:
        """
        신뢰도 대시보드를 렌더링합니다.
        
        Args:
            sources: 출처 신뢰도 정보
            facts: 팩트 체크 결과 (선택사항)
        """
        st.subheader("🔍 Source Credibility Analysis")
        
        if not sources:
            st.info("No sources available for credibility analysis")
            return
        
        # 상단 메트릭 카드
        self._render_credibility_metrics(sources)
        
        # 신뢰도 분포 차트
        self._render_credibility_distribution(sources)
        
        # 출처별 신뢰도 레이더 차트
        self._render_credibility_radar_chart(sources)
        
        # 도메인별 신뢰도 분석
        self._render_domain_analysis(sources)
        
        # 팩트 체크 결과 (있는 경우)
        if facts:
            self._render_fact_check_results(facts)
        
        # 상세 출처 테이블
        self._render_detailed_sources_table(sources)
    
    def _render_credibility_metrics(self, sources: List[SourceCredibility]) -> None:
        """신뢰도 메트릭 카드를 렌더링합니다."""
        col1, col2, col3, col4 = st.columns(4)
        
        # 평균 신뢰도
        avg_credibility = sum(s.overall_score for s in sources) / len(sources)
        with col1:
            st.metric(
                label="Average Credibility",
                value=f"{avg_credibility:.2f}",
                delta=None
            )
        
        # 검증된 출처 수
        verified_count = sum(1 for s in sources if s.verification_status == "verified")
        with col2:
            st.metric(
                label="Verified Sources",
                value=f"{verified_count}/{len(sources)}",
                delta=f"{verified_count/len(sources)*100:.1f}%"
            )
        
        # 고신뢰도 출처 수
        high_credibility_count = sum(1 for s in sources if s.overall_score >= 0.8)
        with col3:
            st.metric(
                label="High Credibility",
                value=f"{high_credibility_count}/{len(sources)}",
                delta=f"{high_credibility_count/len(sources)*100:.1f}%"
            )
        
        # 도메인 타입 분포
        domain_types = {}
        for source in sources:
            domain_type = source.domain_type.value
            domain_types[domain_type] = domain_types.get(domain_type, 0) + 1
        
        most_common_domain = max(domain_types.items(), key=lambda x: x[1])[0] if domain_types else "unknown"
        with col4:
            st.metric(
                label="Most Common Domain",
                value=most_common_domain.title(),
                delta=f"{domain_types.get(most_common_domain, 0)} sources"
            )
    
    def _render_credibility_distribution(self, sources: List[SourceCredibility]) -> None:
        """신뢰도 분포 차트를 렌더링합니다."""
        st.subheader("📊 Credibility Score Distribution")
        
        # 신뢰도 점수 분포
        scores = [s.overall_score for s in sources]
        
        fig = go.Figure()
        
        # 히스토그램
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=20,
            name='Credibility Scores',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # 평균선
        avg_score = np.mean(scores)
        fig.add_vline(
            x=avg_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {avg_score:.2f}"
        )
        
        fig.update_layout(
            title="Distribution of Source Credibility Scores",
            xaxis_title="Credibility Score",
            yaxis_title="Number of Sources",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_credibility_radar_chart(self, sources: List[SourceCredibility]) -> None:
        """신뢰도 레이더 차트를 렌더링합니다."""
        st.subheader("🎯 Credibility Radar Chart")
        
        if len(sources) == 0:
            st.info("No sources available for radar chart")
            return
        
        # 평균 신뢰도 지표 계산
        avg_domain_trust = np.mean([s.domain_trust for s in sources])
        avg_recency = np.mean([s.recency_score for s in sources])
        avg_authority = np.mean([s.authority_score for s in sources])
        avg_citations = np.mean([min(1.0, s.citation_count / 100) for s in sources])  # 0-1 스케일
        
        categories = ['Domain Trust', 'Recency', 'Authority', 'Citations']
        values = [avg_domain_trust, avg_recency, avg_authority, avg_citations]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Average Credibility',
            line_color='blue',
            fillcolor='rgba(0, 100, 255, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Average Credibility Metrics",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_domain_analysis(self, sources: List[SourceCredibility]) -> None:
        """도메인별 신뢰도 분석을 렌더링합니다."""
        st.subheader("🌐 Domain Type Analysis")
        
        # 도메인별 데이터 준비
        domain_data = {}
        for source in sources:
            domain_type = source.domain_type.value
            if domain_type not in domain_data:
                domain_data[domain_type] = []
            domain_data[domain_type].append(source.overall_score)
        
        # 도메인별 평균 신뢰도 계산
        domain_stats = []
        for domain_type, scores in domain_data.items():
            domain_stats.append({
                'domain_type': domain_type.title(),
                'count': len(scores),
                'avg_credibility': np.mean(scores),
                'min_credibility': np.min(scores),
                'max_credibility': np.max(scores)
            })
        
        if not domain_stats:
            st.info("No domain data available")
            return
        
        # 도메인별 신뢰도 바 차트
        df = pd.DataFrame(domain_stats)
        
        fig = px.bar(
            df,
            x='domain_type',
            y='avg_credibility',
            color='avg_credibility',
            color_continuous_scale='RdYlGn',
            title="Average Credibility by Domain Type",
            labels={'avg_credibility': 'Average Credibility Score', 'domain_type': 'Domain Type'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 도메인별 상세 통계 테이블
        st.subheader("📋 Domain Statistics")
        
        # 컬럼 추가
        df['credibility_range'] = df.apply(
            lambda row: f"{row['min_credibility']:.2f} - {row['max_credibility']:.2f}",
            axis=1
        )
        
        # 테이블 표시
        st.dataframe(
            df[['domain_type', 'count', 'avg_credibility', 'credibility_range']],
            use_container_width=True
        )
    
    def _render_fact_check_results(self, facts: List[FactCheckResult]) -> None:
        """팩트 체크 결과를 렌더링합니다."""
        st.subheader("✅ Fact Check Results")
        
        if not facts:
            st.info("No fact check results available")
            return
        
        # 팩트 상태별 분포
        fact_status_counts = {}
        for fact in facts:
            status = fact.fact_status.value
            fact_status_counts[status] = fact_status_counts.get(status, 0) + 1
        
        # 팩트 상태 파이 차트
        fig = px.pie(
            values=list(fact_status_counts.values()),
            names=list(fact_status_counts.keys()),
            title="Fact Verification Status Distribution",
            color_discrete_map={
                'verified': self.color_scheme['verified'],
                'partially_verified': self.color_scheme['partially_verified'],
                'unverified': self.color_scheme['unverified'],
                'conflicting': self.color_scheme['conflicting'],
                'disputed': self.color_scheme['disputed']
            }
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 팩트 체크 상세 결과
        st.subheader("📝 Detailed Fact Check Results")
        
        for i, fact in enumerate(facts, 1):
            with st.expander(f"Fact {i}: {fact.original_text[:100]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Status:** {self._get_status_badge(fact.fact_status)}")
                    st.write(f"**Confidence:** {fact.confidence_score:.2f}")
                    st.write(f"**Verification Stage:** {fact.verification_stage.value}")
                
                with col2:
                    st.write(f"**Supporting Sources:** {len(fact.supporting_sources)}")
                    st.write(f"**Conflicting Sources:** {len(fact.conflicting_sources)}")
                    st.write(f"**Timestamp:** {fact.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 상세 정보
                if fact.verification_details:
                    st.write("**Verification Details:**")
                    for stage, details in fact.verification_details.items():
                        if isinstance(details, dict):
                            st.write(f"- {stage}: {details.get('confidence', 'N/A'):.2f}")
    
    def _render_detailed_sources_table(self, sources: List[SourceCredibility]) -> None:
        """상세 출처 테이블을 렌더링합니다."""
        st.subheader("📋 Detailed Sources Table")
        
        # 데이터 준비
        table_data = []
        for source in sources:
            table_data.append({
                'URL': source.url,
                'Domain Type': source.domain_type.value.title(),
                'Overall Score': f"{source.overall_score:.3f}",
                'Domain Trust': f"{source.domain_trust:.3f}",
                'Recency': f"{source.recency_score:.3f}",
                'Authority': f"{source.authority_score:.3f}",
                'Citations': source.citation_count,
                'Status': self._get_status_badge(source.verification_status),
                'Confidence': f"{source.confidence:.3f}",
                'Author': source.author or 'Unknown',
                'Institution': source.institution or 'Unknown'
            })
        
        df = pd.DataFrame(table_data)
        
        # 정렬 (신뢰도 높은 순)
        df = df.sort_values('Overall Score', ascending=False)
        
        # 테이블 표시
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )
    
    def _get_status_badge(self, status: str) -> str:
        """상태 배지를 반환합니다."""
        color = self.color_scheme.get(status, '#6C757D')
        return f"<span style='background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{status.upper()}</span>"
    
    def render_credibility_heatmap(self, sources: List[SourceCredibility]) -> None:
        """신뢰도 히트맵을 렌더링합니다."""
        st.subheader("🔥 Credibility Heatmap")
        
        if len(sources) < 2:
            st.info("Need at least 2 sources for heatmap")
            return
        
        # 신뢰도 매트릭스 생성
        credibility_matrix = []
        source_names = []
        
        for source in sources:
            source_names.append(source.url.split('/')[-1][:20])  # URL의 마지막 부분
            credibility_matrix.append([
                source.domain_trust,
                source.recency_score,
                source.authority_score,
                min(1.0, source.citation_count / 100)  # 0-1 스케일
            ])
        
        # 히트맵 생성
        fig = go.Figure(data=go.Heatmap(
            z=credibility_matrix,
            x=['Domain Trust', 'Recency', 'Authority', 'Citations'],
            y=source_names,
            colorscale='RdYlGn',
            showscale=True
        ))
        
        fig.update_layout(
            title="Source Credibility Heatmap",
            xaxis_title="Credibility Metrics",
            yaxis_title="Sources",
            height=max(400, len(sources) * 30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_conflict_warnings(self, facts: List[FactCheckResult]) -> None:
        """상충 정보 경고를 렌더링합니다."""
        st.subheader("⚠️ Conflict Warnings")
        
        conflicts = []
        for fact in facts:
            if fact.fact_status in [FactStatus.CONFLICTING, FactStatus.DISPUTED]:
                conflicts.append(fact)
        
        if not conflicts:
            st.success("No conflicts detected in the information")
            return
        
        for i, conflict in enumerate(conflicts, 1):
            with st.expander(f"⚠️ Conflict {i}: {conflict.original_text[:100]}...", expanded=True):
                st.warning(f"**Status:** {conflict.fact_status.value.upper()}")
                st.write(f"**Confidence:** {conflict.confidence_score:.2f}")
                
                if conflict.conflicting_sources:
                    st.write("**Conflicting Sources:**")
                    for source in conflict.conflicting_sources:
                        st.write(f"- {source}")
                
                if conflict.supporting_sources:
                    st.write("**Supporting Sources:**")
                    for source in conflict.supporting_sources:
                        st.write(f"- {source}")


# Streamlit 컴포넌트 함수들
def render_credibility_dashboard(sources: List[SourceCredibility], facts: List[FactCheckResult] = None) -> None:
    """신뢰도 대시보드를 렌더링합니다."""
    viewer = CredibilityViewer()
    viewer.render_credibility_dashboard(sources, facts)


def render_credibility_heatmap(sources: List[SourceCredibility]) -> None:
    """신뢰도 히트맵을 렌더링합니다."""
    viewer = CredibilityViewer()
    viewer.render_credibility_heatmap(sources)


def render_conflict_warnings(facts: List[FactCheckResult]) -> None:
    """상충 정보 경고를 렌더링합니다."""
    viewer = CredibilityViewer()
    viewer.render_conflict_warnings(facts)
