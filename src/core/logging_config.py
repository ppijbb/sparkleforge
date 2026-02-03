"""
Structured Logging Configuration for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade logging with JSON format, sensitive data masking,
and comprehensive monitoring for the advanced multi-agent research system.
"""

import logging
import json
import sys
import os
import re
from typing import Any, Dict, Optional, Union
from datetime import datetime
from pathlib import Path
import structlog
from structlog.stdlib import LoggerFactory


class SensitiveDataMasker:
    """Mask sensitive data in log messages."""
    
    # Common sensitive data patterns
    SENSITIVE_PATTERNS = [
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'api_key="***MASKED***"'),
        (r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'token="***MASKED***"'),
        (r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'password="***MASKED***"'),
        (r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'secret="***MASKED***"'),
        (r'key["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'key="***MASKED***"'),
        (r'OPENROUTER_API_KEY["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'OPENROUTER_API_KEY="***MASKED***"'),
        (r'GOOGLE_API_KEY["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'GOOGLE_API_KEY="***MASKED***"'),
        (r'OPENAI_API_KEY["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'OPENAI_API_KEY="***MASKED***"'),
        (r'ANTHROPIC_API_KEY["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'ANTHROPIC_API_KEY="***MASKED***"'),
        (r'TAVILY_API_KEY["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'TAVILY_API_KEY="***MASKED***"'),
        (r'EXA_API_KEY["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'EXA_API_KEY="***MASKED***"'),
        (r'PUBMED_API_KEY["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'PUBMED_API_KEY="***MASKED***"'),
        (r'IEEE_API_KEY["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'IEEE_API_KEY="***MASKED***"'),
        (r'NEWSAPI_KEY["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'NEWSAPI_KEY="***MASKED***"'),
        (r'PERPLEXITY_API_KEY["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'PERPLEXITY_API_KEY="***MASKED***"'),
    ]
    
    @classmethod
    def mask_sensitive_data(cls, message: str) -> str:
        """Mask sensitive data in a message."""
        masked_message = message
        for pattern, replacement in cls.SENSITIVE_PATTERNS:
            masked_message = re.sub(pattern, replacement, masked_message, flags=re.IGNORECASE)
        return masked_message
    
    @classmethod
    def mask_dict_values(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in dictionary values."""
        masked_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Check if key suggests sensitive data
                if any(sensitive_key in key.lower() for sensitive_key in ['key', 'token', 'password', 'secret']):
                    masked_data[key] = "***MASKED***"
                else:
                    masked_data[key] = cls.mask_sensitive_data(value)
            elif isinstance(value, dict):
                masked_data[key] = cls.mask_dict_values(value)
            elif isinstance(value, list):
                masked_data[key] = [cls.mask_dict_values(item) if isinstance(item, dict) else item for item in value]
            else:
                masked_data[key] = value
        return masked_data


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_timestamp: bool = True, mask_sensitive: bool = True):
        self.include_timestamp = include_timestamp
        self.mask_sensitive = mask_sensitive
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z" if self.include_timestamp else None,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                          'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread', 'threadName', 'processName',
                          'process', 'getMessage', 'message']:
                log_data[key] = value
        
        # Mask sensitive data if enabled
        if self.mask_sensitive:
            log_data = SensitiveDataMasker.mask_dict_values(log_data)
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        if fmt is None:
            fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    mask_sensitive: bool = True,
    enable_structlog: bool = True
) -> logging.Logger:
    """
    Setup structured logging for the Local Researcher system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json, colored, simple)
        log_file: Path to log file
        enable_console: Enable console logging
        enable_file: Enable file logging
        mask_sensitive: Mask sensitive data in logs
        enable_structlog: Enable structlog for advanced logging
    
    Returns:
        Configured logger instance
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Setup structlog if enabled
    if enable_structlog:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if log_format == "json" else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        if log_format == "json":
            console_handler.setFormatter(JSONFormatter(mask_sensitive=mask_sensitive))
        elif log_format == "colored":
            console_handler.setFormatter(ColoredFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
            ))
        
        root_logger.addHandler(console_handler)
    
    # File handler
    if enable_file and log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(JSONFormatter(mask_sensitive=mask_sensitive))
        root_logger.addHandler(file_handler)
    
    # Create application logger
    app_logger = logging.getLogger("local_researcher")
    app_logger.setLevel(numeric_level)
    
    return app_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(f"local_researcher.{name}")


class LoggingContext:
    """Context manager for adding structured logging context."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_context = {}
    
    def __enter__(self):
        # Store old context
        for key in self.context:
            if hasattr(self.logger, key):
                self.old_context[key] = getattr(self.logger, key)
        
        # Set new context
        for key, value in self.context.items():
            setattr(self.logger, key, value)
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old context
        for key, value in self.old_context.items():
            setattr(self.logger, key, value)
        
        # Remove new context
        for key in self.context:
            if key not in self.old_context:
                if hasattr(self.logger, key):
                    delattr(self.logger, key)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """Log function call with parameters."""
    logger.info(
        f"Calling function: {func_name}",
        extra={
            "function": func_name,
            "parameters": SensitiveDataMasker.mask_dict_values(kwargs),
            "event_type": "function_call"
        }
    )


def log_performance(logger: logging.Logger, operation: str, duration: float, **metrics):
    """Log performance metrics."""
    logger.info(
        f"Performance: {operation}",
        extra={
            "operation": operation,
            "duration_seconds": duration,
            "event_type": "performance",
            **metrics
        }
    )


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any],
    severity: str = "ERROR"
):
    """Log error with additional context."""
    logger.error(
        f"Error occurred: {str(error)}",
        extra={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": SensitiveDataMasker.mask_dict_values(context),
            "event_type": "error",
            "severity": severity
        },
        exc_info=True
    )


# Initialize default logging
def initialize_logging():
    """Initialize default logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "json")
    log_file = os.getenv("LOG_FILE", "logs/local_researcher.log")
    mask_sensitive = os.getenv("MASK_SENSITIVE_DATA", "true").lower() == "true"
    
    return setup_logging(
        log_level=log_level,
        log_format=log_format,
        log_file=log_file,
        mask_sensitive=mask_sensitive
    )


# Export commonly used functions
__all__ = [
    'setup_logging',
    'get_logger',
    'LoggingContext',
    'log_function_call',
    'log_performance',
    'log_error_with_context',
    'initialize_logging',
    'SensitiveDataMasker',
    'JSONFormatter',
    'ColoredFormatter'
]
