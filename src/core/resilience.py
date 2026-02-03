"""
Resilience and Error Handling Patterns for MCP Agent System

This module provides standardized error handling, retry mechanisms, and resilience patterns
to improve the reliability and stability of the MCP Agent system.
"""

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_factor: float = 0.1


@dataclass
class ErrorContext:
    """Context for error handling."""
    operation: str
    component: str
    severity: ErrorSeverity
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascade failures.
    
    Prevents repeated calls to failing services by temporarily stopping
    calls after a threshold of failures is reached.
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type to consider as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to function."""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._call_async(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self._call_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    async def _call_async(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                logger.info("Circuit breaker transitioning to half-open")
            else:
                raise Exception(f"Circuit breaker is open for {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _call_sync(self, func: Callable, *args, **kwargs):
        """Execute sync function with circuit breaker."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                logger.info("Circuit breaker transitioning to half-open")
            else:
                raise Exception(f"Circuit breaker is open for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return self.last_failure_time is not None and self.recovery_timeout is not None and time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == "half_open":
            self.state = "closed"
            logger.info("Circuit breaker closed after successful call")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RetryHandler:
    """
    Retry handler with exponential backoff and jitter.
    
    Provides resilient retry logic for handling transient failures
    in external service calls and network operations.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
    
    def __call__(self, 
                 retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception):
        """Decorator to apply retry logic to function."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._retry_async(func, retry_exceptions, *args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._retry_sync(func, retry_exceptions, *args, **kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def _retry_async(self, 
                          func: Callable, 
                          retry_exceptions: Union[Type[Exception], List[Type[Exception]]],
                          *args, **kwargs):
        """Execute async function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Check if exception matches retry_exceptions
                if isinstance(retry_exceptions, list):
                    if not any(isinstance(e, exc_type) for exc_type in retry_exceptions):
                        raise
                elif not isinstance(e, retry_exceptions):
                    raise
                last_exception = e
                if attempt == self.config.max_attempts - 1:
                    logger.error(f"Function {func.__name__} failed after {self.config.max_attempts} attempts")
                    raise
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        raise last_exception if last_exception else Exception("Retry failed")
    
    def _retry_sync(self, 
                   func: Callable, 
                   retry_exceptions: Union[Type[Exception], List[Type[Exception]]],
                   *args, **kwargs):
        """Execute sync function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if exception matches retry_exceptions
                if isinstance(retry_exceptions, list):
                    if not any(isinstance(e, exc_type) for exc_type in retry_exceptions):
                        raise
                elif not isinstance(e, retry_exceptions):
                    raise
                last_exception = e
                if attempt == self.config.max_attempts - 1:
                    logger.error(f"Function {func.__name__} failed after {self.config.max_attempts} attempts")
                    raise
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise last_exception if last_exception else Exception("Retry failed")
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add jitter to prevent thundering herd
            jitter_amount = delay * self.config.backoff_factor
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay += jitter
        
        return max(0, delay)


class ErrorHandler:
    """
    Centralized error handling and logging.
    
    Provides consistent error handling across the MCP Agent system
    with proper logging, metrics, and error categorization.
    """
    
    def __init__(self, component: str):
        """
        Initialize error handler.
        
        Args:
            component: Component name for error tracking
        """
        self.component = component
        self.error_counts: Dict[str, int] = {}
        self.error_logger = logging.getLogger(f"{__name__}.{component}")
    
    def handle_error(self, 
                    error: Exception, 
                    operation: str = "",
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None,
                    reraise: bool = True) -> Optional[Exception]:
        """
        Handle error with logging and metrics.
        
        Args:
            error: Exception that occurred
            operation: Operation being performed
            severity: Error severity level
            context: Additional context information
            reraise: Whether to reraise the exception
            
        Returns:
            The exception if reraise is False, None otherwise
        """
        error_type = type(error).__name__
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Create error context
        error_context = ErrorContext(
            operation=operation,
            component=self.component,
            severity=severity,
            metadata=context
        )
        
        # Log error with appropriate level
        self._log_error(error, error_context)
        
        # Send to monitoring/metrics (placeholder)
        self._send_metrics(error, error_context)
        
        if reraise:
            raise error
        else:
            return error
    
    def _log_error(self, error: Exception, context: ErrorContext):
        """Log error with appropriate level based on severity."""
        error_msg = f"[{context.component}] {context.operation}: {str(error)}"
        
        if context.severity == ErrorSeverity.CRITICAL:
            self.error_logger.critical(error_msg, exc_info=True)
        elif context.severity == ErrorSeverity.HIGH:
            self.error_logger.error(error_msg, exc_info=True)
        elif context.severity == ErrorSeverity.MEDIUM:
            self.error_logger.warning(error_msg)
        else:
            self.error_logger.info(error_msg)
    
    def _send_metrics(self, error: Exception, context: ErrorContext):
        """Send error metrics to monitoring system."""
        # Placeholder for metrics integration
        # Could integrate with Prometheus, Datadog, etc.
        pass
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors for this component."""
        total_errors = sum(self.error_counts.values())
        return {
            "component": self.component,
            "total_errors": total_errors,
            "error_types": dict(self.error_counts),
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }


# Convenience decorators and instances
def with_retry(config: Optional[RetryConfig] = None, retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception):
    """Convenience decorator for retry logic."""
    retry_handler = RetryHandler(config)
    return retry_handler(retry_exceptions)


def with_circuit_breaker(failure_threshold: int = 5, 
                        recovery_timeout: float = 60.0,
                        expected_exception: Type[Exception] = Exception):
    """Convenience decorator for circuit breaker."""
    return CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)


def get_error_handler(component: str) -> ErrorHandler:
    """Get error handler for component."""
    return ErrorHandler(component)


# Example usage patterns:
"""
@with_retry(RetryConfig(max_attempts=3, base_delay=1.0))
async def api_call(url: str):
    response = await httpx.get(url)
    response.raise_for_status()
    return response.json()

@with_circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
def database_query(query: str):
    return db.execute(query)

error_handler = get_error_handler("LLMManager")
try:
    result = await llm_call(prompt)
except Exception as e:
    error_handler.handle_error(e, operation="llm_call", severity=ErrorSeverity.HIGH)
"""