#!/usr/bin/env python3
"""
Concurrency Manager - Dynamic Concurrency Adjustment

Monitors system load and dynamically adjusts concurrency based on:
- CPU usage
- Memory usage
- Active task count
- System performance metrics

Auto-discovers optimal concurrency settings.
"""

import asyncio
import logging
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    active_tasks: int
    timestamp: datetime


@dataclass
class ConcurrencyConfig:
    """Concurrency configuration."""
    min_concurrency: int = 1
    max_concurrency: int = 50
    base_concurrency: int = 5
    cpu_threshold_high: float = 80.0  # Reduce concurrency if CPU > 80%
    cpu_threshold_low: float = 50.0   # Increase concurrency if CPU < 50%
    memory_threshold_high: float = 85.0  # Reduce concurrency if memory > 85%
    memory_threshold_low: float = 60.0   # Increase concurrency if memory < 60%
    adjustment_step: int = 2  # How much to adjust concurrency per step
    check_interval: float = 5.0  # Seconds between checks


class ConcurrencyManager:
    """Dynamic concurrency manager with system load monitoring."""
    
    def __init__(self, config: Optional[ConcurrencyConfig] = None):
        """
        Initialize concurrency manager.
        
        Args:
            config: Concurrency configuration (uses defaults if None)
        """
        self.config = config or ConcurrencyConfig()
        self.current_concurrency = self.config.base_concurrency
        
        # Metrics history (for trend analysis)
        self.metrics_history: deque = deque(maxlen=100)
        self.active_tasks_count = 0
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=50)
        self.optimal_concurrency = self.config.base_concurrency
        
        # Lock for thread-safe updates
        self.lock = asyncio.Lock()
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_active = False
        
        logger.info(
            f"ConcurrencyManager initialized: "
            f"base={self.config.base_concurrency}, "
            f"range=[{self.config.min_concurrency}, {self.config.max_concurrency}]"
        )
    
    async def start_monitoring(self):
        """Start background monitoring task."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Concurrency monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring task."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Concurrency monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                await self.adjust_concurrency()
                await asyncio.sleep(self.config.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.check_interval)
    
    async def monitor_system_load(self) -> SystemMetrics:
        """
        Monitor current system load.
        
        Returns:
            SystemMetrics object with current metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            async with self.lock:
                active_tasks = self.active_tasks_count
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                active_tasks=active_tasks,
                timestamp=datetime.now()
            )
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error monitoring system load: {e}")
            # Return safe defaults
            return SystemMetrics(
                cpu_percent=50.0,
                memory_percent=50.0,
                active_tasks=self.active_tasks_count,
                timestamp=datetime.now()
            )
    
    async def calculate_optimal_concurrency(self) -> int:
        """
        Calculate optimal concurrency based on system metrics.
        
        Returns:
            Optimal concurrency value
        """
        metrics = await self.monitor_system_load()
        
        # Start with current concurrency
        optimal = self.current_concurrency
        
        # CPU-based adjustment
        if metrics.cpu_percent > self.config.cpu_threshold_high:
            # High CPU - reduce concurrency
            optimal = max(
                self.config.min_concurrency,
                optimal - self.config.adjustment_step
            )
            logger.debug(f"High CPU ({metrics.cpu_percent:.1f}%), reducing concurrency to {optimal}")
        elif metrics.cpu_percent < self.config.cpu_threshold_low:
            # Low CPU - can increase concurrency
            optimal = min(
                self.config.max_concurrency,
                optimal + self.config.adjustment_step
            )
            logger.debug(f"Low CPU ({metrics.cpu_percent:.1f}%), increasing concurrency to {optimal}")
        
        # Memory-based adjustment
        if metrics.memory_percent > self.config.memory_threshold_high:
            # High memory - reduce concurrency
            optimal = max(
                self.config.min_concurrency,
                optimal - self.config.adjustment_step
            )
            logger.debug(f"High memory ({metrics.memory_percent:.1f}%), reducing concurrency to {optimal}")
        elif metrics.memory_percent < self.config.memory_threshold_low:
            # Low memory - can increase concurrency
            optimal = min(
                self.config.max_concurrency,
                optimal + self.config.adjustment_step
            )
            logger.debug(f"Low memory ({metrics.memory_percent:.1f}%), increasing concurrency to {optimal}")
        
        # Active tasks adjustment
        if metrics.active_tasks > self.current_concurrency * 1.5:
            # Too many active tasks - increase concurrency
            optimal = min(
                self.config.max_concurrency,
                optimal + self.config.adjustment_step
            )
            logger.debug(f"High active tasks ({metrics.active_tasks}), increasing concurrency to {optimal}")
        elif metrics.active_tasks < self.current_concurrency * 0.5:
            # Few active tasks - can reduce concurrency
            optimal = max(
                self.config.min_concurrency,
                optimal - self.config.adjustment_step
            )
            logger.debug(f"Low active tasks ({metrics.active_tasks}), reducing concurrency to {optimal}")
        
        # Ensure within bounds
        optimal = max(self.config.min_concurrency, min(self.config.max_concurrency, optimal))
        
        return optimal
    
    async def adjust_concurrency(self) -> int:
        """
        Adjust concurrency based on current system load.
        
        Returns:
            New concurrency value
        """
        optimal = await self.calculate_optimal_concurrency()
        
        async with self.lock:
            old_concurrency = self.current_concurrency
            self.current_concurrency = optimal
            self.optimal_concurrency = optimal
        
        if old_concurrency != optimal:
            logger.info(
                f"Concurrency adjusted: {old_concurrency} → {optimal} "
                f"(CPU: {self.metrics_history[-1].cpu_percent:.1f}%, "
                f"Memory: {self.metrics_history[-1].memory_percent:.1f}%, "
                f"Active tasks: {self.metrics_history[-1].active_tasks})"
            )
        
        return optimal
    
    def get_current_concurrency(self) -> int:
        """Get current concurrency value."""
        return self.current_concurrency
    
    def increment_active_tasks(self):
        """Increment active tasks count."""
        self.active_tasks_count += 1
    
    def decrement_active_tasks(self):
        """Decrement active tasks count."""
        self.active_tasks_count = max(0, self.active_tasks_count - 1)
    
    def record_performance(self, tasks_completed: int, duration: float):
        """
        Record performance metrics for optimal concurrency discovery.
        
        Args:
            tasks_completed: Number of tasks completed
            duration: Duration in seconds
        """
        throughput = tasks_completed / duration if duration > 0 else 0.0
        
        self.performance_history.append({
            'concurrency': self.current_concurrency,
            'throughput': throughput,
            'tasks_completed': tasks_completed,
            'duration': duration,
            'timestamp': datetime.now()
        })
        
        # Update optimal concurrency based on throughput
        if len(self.performance_history) >= 5:
            # Find concurrency with best throughput
            concurrency_throughput = {}
            for perf in self.performance_history:
                concurrency = perf['concurrency']
                if concurrency not in concurrency_throughput:
                    concurrency_throughput[concurrency] = []
                concurrency_throughput[concurrency].append(perf['throughput'])
            
            # Calculate average throughput per concurrency
            avg_throughput = {
                c: sum(t) / len(t)
                for c, t in concurrency_throughput.items()
            }
            
            if avg_throughput:
                best_concurrency = max(avg_throughput.items(), key=lambda x: x[1])[0]
                self.optimal_concurrency = best_concurrency
                logger.debug(f"Optimal concurrency discovered: {best_concurrency} (throughput: {avg_throughput[best_concurrency]:.2f} tasks/s)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get concurrency manager statistics."""
        if not self.metrics_history:
            return {
                'current_concurrency': self.current_concurrency,
                'optimal_concurrency': self.optimal_concurrency,
                'active_tasks': self.active_tasks_count,
                'monitoring_active': self.monitoring_active
            }
        
        latest = self.metrics_history[-1]
        avg_cpu = sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m.memory_percent for m in self.metrics_history) / len(self.metrics_history)
        
        return {
            'current_concurrency': self.current_concurrency,
            'optimal_concurrency': self.optimal_concurrency,
            'active_tasks': self.active_tasks_count,
            'monitoring_active': self.monitoring_active,
            'current_cpu_percent': latest.cpu_percent,
            'current_memory_percent': latest.memory_percent,
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'metrics_history_size': len(self.metrics_history)
        }


# Global concurrency manager instance
_concurrency_manager_instance: Optional[ConcurrencyManager] = None


def get_concurrency_manager() -> ConcurrencyManager:
    """Get or create global concurrency manager instance."""
    global _concurrency_manager_instance
    if _concurrency_manager_instance is None:
        _concurrency_manager_instance = ConcurrencyManager()
    return _concurrency_manager_instance

