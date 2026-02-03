#!/usr/bin/env python3
"""
Real-time System Monitor for Local Researcher - Production-Grade Reliability (Innovation 8)

This module provides comprehensive system monitoring capabilities including
performance metrics, health checks, and real-time alerts with 8 core innovations.
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque
import threading
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.researcher_config import config
from src.core.reliability import execute_with_reliability, CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics with 8 innovations support."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_processes: int
    research_tasks: int
    agent_status: Dict[str, str]
    error_count: int
    warning_count: int
    # 8 innovations metrics
    adaptive_supervisor_metrics: Dict[str, Any] = None
    hierarchical_compression_metrics: Dict[str, Any] = None
    multi_model_orchestration_metrics: Dict[str, Any] = None
    continuous_verification_metrics: Dict[str, Any] = None
    streaming_pipeline_metrics: Dict[str, Any] = None
    universal_mcp_hub_metrics: Dict[str, Any] = None
    adaptive_context_window_metrics: Dict[str, Any] = None
    production_reliability_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.adaptive_supervisor_metrics is None:
            self.adaptive_supervisor_metrics = {}
        if self.hierarchical_compression_metrics is None:
            self.hierarchical_compression_metrics = {}
        if self.multi_model_orchestration_metrics is None:
            self.multi_model_orchestration_metrics = {}
        if self.continuous_verification_metrics is None:
            self.continuous_verification_metrics = {}
        if self.streaming_pipeline_metrics is None:
            self.streaming_pipeline_metrics = {}
        if self.universal_mcp_hub_metrics is None:
            self.universal_mcp_hub_metrics = {}
        if self.adaptive_context_window_metrics is None:
            self.adaptive_context_window_metrics = {}
        if self.production_reliability_metrics is None:
            self.production_reliability_metrics = {}


@dataclass
class Alert:
    """System alert with innovation-specific categories."""
    timestamp: datetime
    level: str  # info, warning, error, critical
    category: str  # performance, memory, disk, network, research, innovation
    innovation_type: Optional[str] = None  # Which of the 8 innovations
    message: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class HealthMonitor:
    """Production-Grade Health Monitor implementing Innovation 8."""
    
    def __init__(self):
        """Initialize the health monitor with 8 innovations support."""
        self.config = config
        
        # Monitoring settings from config
        self.monitoring_interval = getattr(config.reliability, 'monitoring_interval', 5)
        self.metrics_history_size = getattr(config.reliability, 'metrics_history_size', 1000)
        self.alert_thresholds = {
            'cpu_usage': getattr(config.reliability, 'cpu_threshold', 80.0),
            'memory_usage': getattr(config.reliability, 'memory_threshold', 85.0),
            'disk_usage': getattr(config.reliability, 'disk_threshold', 90.0),
            'error_rate': getattr(config.reliability, 'error_rate_threshold', 10.0),
            'response_time': getattr(config.reliability, 'response_time_threshold', 30.0),
            'mcp_tool_failure_rate': getattr(config.reliability, 'mcp_failure_threshold', 20.0)
        }
        
        # Data storage
        self.metrics_history = deque(maxlen=self.metrics_history_size)
        self.alerts = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Circuit breaker for reliability
        from src.core.reliability import CircuitBreakerConfig
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60
        )
        self.circuit_breaker = CircuitBreaker("health_monitor", circuit_config)
        
        # Callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        logger.info("Health Monitor initialized with 8 innovations support")
    
    async def start_monitoring(self):
        """Start real-time monitoring with production-grade reliability."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Health monitoring started with 8 innovations support")
    
    async def stop_monitoring(self):
        """Stop real-time monitoring gracefully."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Sleep until next check
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics with 8 innovations support."""
        try:
            # Basic system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process count
            active_processes = len(psutil.pids())
            
            # Research-specific metrics
            research_tasks = self._get_research_task_count()
            agent_status = self._get_agent_status()
            error_count, warning_count = self._get_log_counts()
            
            # 8 innovations metrics
            adaptive_supervisor_metrics = self._get_adaptive_supervisor_metrics()
            hierarchical_compression_metrics = self._get_hierarchical_compression_metrics()
            multi_model_orchestration_metrics = self._get_multi_model_orchestration_metrics()
            continuous_verification_metrics = self._get_continuous_verification_metrics()
            streaming_pipeline_metrics = self._get_streaming_pipeline_metrics()
            universal_mcp_hub_metrics = self._get_universal_mcp_hub_metrics()
            adaptive_context_window_metrics = self._get_adaptive_context_window_metrics()
            production_reliability_metrics = self._get_production_reliability_metrics()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_processes=active_processes,
                research_tasks=research_tasks,
                agent_status=agent_status,
                error_count=error_count,
                warning_count=warning_count,
                adaptive_supervisor_metrics=adaptive_supervisor_metrics,
                hierarchical_compression_metrics=hierarchical_compression_metrics,
                multi_model_orchestration_metrics=multi_model_orchestration_metrics,
                continuous_verification_metrics=continuous_verification_metrics,
                streaming_pipeline_metrics=streaming_pipeline_metrics,
                universal_mcp_hub_metrics=universal_mcp_hub_metrics,
                adaptive_context_window_metrics=adaptive_context_window_metrics,
                production_reliability_metrics=production_reliability_metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics with 8 innovations
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                active_processes=0,
                research_tasks=0,
                agent_status={},
                error_count=0,
                warning_count=0,
                adaptive_supervisor_metrics={},
                hierarchical_compression_metrics={},
                multi_model_orchestration_metrics={},
                continuous_verification_metrics={},
                streaming_pipeline_metrics={},
                universal_mcp_hub_metrics={},
                adaptive_context_window_metrics={},
                production_reliability_metrics={}
            )
    
    def _get_research_task_count(self) -> int:
        """Get current research task count."""
        # This would be implemented to get actual task count from orchestrator
        return 0
    
    def _get_agent_status(self) -> Dict[str, str]:
        """Get current agent status."""
        # This would be implemented to get actual agent status
        return {
            "analyzer": "running",
            "decomposer": "running",
            "researcher": "running",
            "evaluator": "running",
            "validator": "running",
            "synthesizer": "running"
        }
    
    def _get_log_counts(self) -> Tuple[int, int]:
        """Get error and warning counts from logs."""
        # This would be implemented to parse actual log files
        return 0, 0
    
    # 8 Innovations Metrics Collection Methods
    
    def _get_adaptive_supervisor_metrics(self) -> Dict[str, Any]:
        """Get Adaptive Supervisor (Innovation 1) metrics."""
        return {
            "active_researchers": getattr(self.config.agent, 'max_concurrent_research_units', 0),
            "min_researchers": getattr(self.config.agent, 'min_researchers', 1),
            "max_researchers": getattr(self.config.agent, 'max_researchers', 10),
            "fast_track_enabled": getattr(self.config.agent, 'enable_fast_track', False),
            "auto_retry_enabled": getattr(self.config.agent, 'enable_auto_retry', True),
            "priority_queue_enabled": getattr(self.config.agent, 'priority_queue_enabled', True),
            "quality_monitoring_enabled": getattr(self.config.agent, 'enable_quality_monitoring', True),
            "quality_threshold": getattr(self.config.agent, 'quality_threshold', 0.8)
        }
    
    def _get_hierarchical_compression_metrics(self) -> Dict[str, Any]:
        """Get Hierarchical Compression (Innovation 2) metrics."""
        return {
            "compression_enabled": getattr(self.config.compression, 'enabled', False),
            "target_compression_ratio": getattr(self.config.compression, 'target_compression_ratio', 0.7),
            "validation_enabled": getattr(self.config.compression, 'validation_enabled', True),
            "history_enabled": getattr(self.config.compression, 'history_enabled', True),
            "parallel_compression": getattr(self.config.research, 'enable_parallel_compression', False)
        }
    
    def _get_multi_model_orchestration_metrics(self) -> Dict[str, Any]:
        """Get Multi-Model Orchestration (Innovation 3) metrics."""
        return {
            "primary_model": getattr(self.config.llm, 'primary_model', 'gemini-2.5-flash-lite'),
            "planning_model": getattr(self.config.llm, 'planning_model', 'gemini-2.5-flash-lite'),
            "reasoning_model": getattr(self.config.llm, 'reasoning_model', 'gemini-2.5-pro'),
            "verification_model": getattr(self.config.llm, 'verification_model', 'claude-sonnet-4'),
            "generation_model": getattr(self.config.llm, 'generation_model', 'gemini-2.5-flash-lite'),
            "compression_model": getattr(self.config.llm, 'compression_model', 'gemini-2.5-flash-lite'),
            "cost_optimization_enabled": getattr(self.config.llm, 'enable_cost_optimization', True),
            "budget_limit": getattr(self.config.llm, 'budget_limit', 100.0)
        }
    
    def _get_continuous_verification_metrics(self) -> Dict[str, Any]:
        """Get Continuous Verification (Innovation 4) metrics."""
        return {
            "verification_enabled": getattr(self.config.verification, 'enabled', False),
            "self_verification_enabled": getattr(self.config.verification, 'self_verification_enabled', True),
            "cross_verification_enabled": getattr(self.config.verification, 'cross_verification_enabled', True),
            "external_verification_enabled": getattr(self.config.verification, 'external_verification_enabled', False),
            "confidence_scoring_enabled": getattr(self.config.verification, 'confidence_scoring_enabled', True),
            "early_warning_enabled": getattr(self.config.verification, 'early_warning_enabled', True),
            "fact_checking_enabled": getattr(self.config.verification, 'fact_checking_enabled', True),
            "parallel_verification": getattr(self.config.research, 'enable_parallel_verification', False)
        }
    
    def _get_streaming_pipeline_metrics(self) -> Dict[str, Any]:
        """Get Streaming Pipeline (Innovation 5) metrics."""
        return {
            "streaming_enabled": getattr(self.config.research, 'enable_streaming', False),
            "stream_chunk_size": getattr(self.config.research, 'stream_chunk_size', 1000),
            "progressive_reporting_enabled": getattr(self.config.research, 'enable_progressive_reporting', True),
            "incremental_save_enabled": getattr(self.config.research, 'enable_incremental_save', True),
            "pipeline_parallelization": getattr(self.config.research, 'enable_pipeline_parallelization', True)
        }
    
    def _get_universal_mcp_hub_metrics(self) -> Dict[str, Any]:
        """Get Universal MCP Hub (Innovation 6) metrics."""
        return {
            "mcp_enabled": getattr(self.config.mcp, 'enabled', False),
            "server_count": len(getattr(self.config.mcp, 'server_names', [])),
            "plugin_architecture_enabled": getattr(self.config.mcp, 'enable_plugin_architecture', True),
            "auto_fallback_enabled": getattr(self.config.mcp, 'enable_auto_fallback', True),
            "performance_tracking_enabled": getattr(self.config.mcp, 'enable_performance_tracking', True),
            "smart_tool_selection_enabled": getattr(self.config.mcp, 'enable_smart_tool_selection', True),
            "search_tools_count": len(getattr(self.config.mcp, 'search_tools', [])),
            "data_tools_count": len(getattr(self.config.mcp, 'data_tools', [])),
            "code_tools_count": len(getattr(self.config.mcp, 'code_tools', [])),
            "academic_tools_count": len(getattr(self.config.mcp, 'academic_tools', [])),
            "business_tools_count": len(getattr(self.config.mcp, 'business_tools', []))
        }
    
    def _get_adaptive_context_window_metrics(self) -> Dict[str, Any]:
        """Get Adaptive Context Window (Innovation 7) metrics."""
        return {
            "context_window_enabled": getattr(self.config.context_window, 'enabled', False),
            "min_tokens": getattr(self.config.context_window, 'min_tokens', 2000),
            "max_tokens": getattr(self.config.context_window, 'max_tokens', 1000000),
            "importance_preservation_enabled": getattr(self.config.context_window, 'importance_preservation_enabled', True),
            "auto_compression_enabled": getattr(self.config.context_window, 'auto_compression_enabled', True),
            "long_term_memory_enabled": getattr(self.config.context_window, 'long_term_memory_enabled', True),
            "memory_refresh_enabled": getattr(self.config.context_window, 'memory_refresh_enabled', True)
        }
    
    def _get_production_reliability_metrics(self) -> Dict[str, Any]:
        """Get Production-Grade Reliability (Innovation 8) metrics."""
        return {
            "circuit_breaker_enabled": getattr(self.config.reliability, 'circuit_breaker_enabled', True),
            "exponential_backoff_enabled": getattr(self.config.reliability, 'exponential_backoff_enabled', True),
            "state_persistence_enabled": getattr(self.config.reliability, 'state_persistence_enabled', True),
            "health_check_enabled": getattr(self.config.reliability, 'health_check_enabled', True),
            "graceful_degradation_enabled": getattr(self.config.reliability, 'graceful_degradation_enabled', True),
            "detailed_logging_enabled": getattr(self.config.reliability, 'detailed_logging_enabled', True),
            "failure_threshold": getattr(self.config.reliability, 'failure_threshold', 5),
            "recovery_timeout": getattr(self.config.reliability, 'recovery_timeout', 60),
            "circuit_breaker_state": self.circuit_breaker.state if hasattr(self, 'circuit_breaker') else 'closed'
        }
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check metrics against thresholds and generate alerts."""
        try:
            # CPU usage alert
            if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
                self._create_alert(
                    level="warning",
                    category="performance",
                    message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                    details={"cpu_usage": metrics.cpu_usage, "threshold": self.alert_thresholds['cpu_usage']}
                )
            
            # Memory usage alert
            if metrics.memory_usage > self.alert_thresholds['memory_usage']:
                self._create_alert(
                    level="warning",
                    category="memory",
                    message=f"High memory usage: {metrics.memory_usage:.1f}%",
                    details={"memory_usage": metrics.memory_usage, "threshold": self.alert_thresholds['memory_usage']}
                )
            
            # Disk usage alert
            if metrics.disk_usage > self.alert_thresholds['disk_usage']:
                self._create_alert(
                    level="error",
                    category="disk",
                    message=f"High disk usage: {metrics.disk_usage:.1f}%",
                    details={"disk_usage": metrics.disk_usage, "threshold": self.alert_thresholds['disk_usage']}
                )
            
            # Error rate alert
            if metrics.error_count > self.alert_thresholds['error_rate']:
                self._create_alert(
                    level="error",
                    category="research",
                    message=f"High error rate: {metrics.error_count} errors",
                    details={"error_count": metrics.error_count, "threshold": self.alert_thresholds['error_rate']}
                )
            
        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")
    
    def _create_alert(self, level: str, category: str, message: str, details: Dict[str, Any]):
        """Create and process a new alert."""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            details=details
        )
        
        self.alerts.append(alert)
        
        # Log alert
        logger.warning(f"ALERT [{level.upper()}] {category}: {message}")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """Get metrics history for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get recent alerts for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alerts if a.timestamp >= cutoff_time]
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        try:
            if not self.metrics_history:
                return 100.0
            
            recent_metrics = self.get_metrics_history(hours=1)
            if not recent_metrics:
                return 100.0
            
            # Calculate average metrics
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_disk = sum(m.disk_usage for m in recent_metrics) / len(recent_metrics)
            
            # Calculate health score (lower usage = higher score)
            cpu_score = max(0, 100 - avg_cpu)
            memory_score = max(0, 100 - avg_memory)
            disk_score = max(0, 100 - avg_disk)
            
            # Weighted average
            health_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
            
            return min(100.0, max(0.0, health_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return 50.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for dashboard."""
        try:
            current_metrics = self.get_current_metrics()
            if not current_metrics:
                return {}
            
            recent_alerts = self.get_recent_alerts(hours=1)
            health_score = self.get_system_health_score()
            
            return {
                "timestamp": current_metrics.timestamp.isoformat(),
                "health_score": health_score,
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "disk_usage": current_metrics.disk_usage,
                "active_processes": current_metrics.active_processes,
                "research_tasks": current_metrics.research_tasks,
                "agent_status": current_metrics.agent_status,
                "recent_alerts": len(recent_alerts),
                "error_count": current_metrics.error_count,
                "warning_count": current_metrics.warning_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def export_metrics(self, file_path: str, format: str = "json"):
        """Export metrics to file."""
        try:
            metrics_data = [asdict(m) for m in self.metrics_history]
            
            if format == "json":
                with open(file_path, 'w') as f:
                    json.dump(metrics_data, f, indent=2, default=str)
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame(metrics_data)
                df.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Metrics exported to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise
    
    def export_alerts(self, file_path: str, format: str = "json"):
        """Export alerts to file."""
        try:
            alerts_data = [asdict(a) for a in self.alerts]
            
            if format == "json":
                with open(file_path, 'w') as f:
                    json.dump(alerts_data, f, indent=2, default=str)
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame(alerts_data)
                df.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Alerts exported to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export alerts: {e}")
            raise
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            current_metrics = self.get_current_metrics()
            recent_alerts = self.get_recent_alerts(hours=1)
            
            # Determine overall status
            error_alerts = [a for a in recent_alerts if a.level == "error"]
            warning_alerts = [a for a in recent_alerts if a.level == "warning"]
            
            if error_alerts:
                overall_status = "critical"
            elif warning_alerts:
                overall_status = "warning"
            else:
                overall_status = "healthy"
            
            # Calculate health score (0-100)
            health_score = 100
            if current_metrics:
                if current_metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
                    health_score -= 20
                if current_metrics.memory_usage > self.alert_thresholds['memory_usage']:
                    health_score -= 20
                if current_metrics.disk_usage > self.alert_thresholds['disk_usage']:
                    health_score -= 20
                if current_metrics.error_count > 0:
                    health_score -= min(30, current_metrics.error_count * 5)
            
            return {
                "overall_status": overall_status,
                "health_score": max(0, health_score),
                "timestamp": datetime.now().isoformat(),
                "metrics": current_metrics.__dict__ if current_metrics else {},
                "recent_alerts_count": len(recent_alerts),
                "error_alerts_count": len(error_alerts),
                "warning_alerts_count": len(warning_alerts),
                "monitoring_active": self.monitoring_active,
                "circuit_breaker_status": self.circuit_breaker.state.value if hasattr(self.circuit_breaker, 'state') else "unknown"
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "overall_status": "error",
                "health_score": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
