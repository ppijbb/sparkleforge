"""
Logging system for Local Researcher

This module provides a centralized logging system with configurable
log levels, formats, and output destinations with color support.
Enhanced with structured logging, agent-specific contexts, and tool execution tracking.
"""

import logging
import logging.handlers
import sys
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextvars import ContextVar
from dataclasses import dataclass, field


# Context variables for agent-specific logging
current_agent_id: ContextVar[Optional[str]] = ContextVar('current_agent_id', default=None)
current_session_id: ContextVar[Optional[str]] = ContextVar('current_session_id', default=None)
current_workflow_stage: ContextVar[Optional[str]] = ContextVar('current_workflow_stage', default=None)


@dataclass
class LogContext:
    """Logging context for structured logging."""
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    workflow_stage: Optional[str] = None
    tool_name: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class ToolExecutionLog:
    """Tool execution logging data."""
    tool_name: str
    execution_time: float
    success: bool
    confidence: float = 0.0
    error_message: Optional[str] = None
    input_params: Optional[Dict[str, Any]] = None
    output_summary: Optional[str] = None


@dataclass
class AgentCommunicationLog:
    """Agent communication logging data."""
    from_agent: str
    action: str
    to_agent: Optional[str] = None
    result_count: int = 0
    discussion_topic: Optional[str] = None
    shared_data_size: int = 0


class StructuredFormatter(logging.Formatter):
    """JSON structured formatter for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # 기본 로그 데이터
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 컨텍스트 정보 추가
        if hasattr(record, 'agent_id') and record.agent_id:
            log_data["agent_id"] = record.agent_id
        if hasattr(record, 'session_id') and record.session_id:
            log_data["session_id"] = record.session_id
        if hasattr(record, 'workflow_stage') and record.workflow_stage:
            log_data["workflow_stage"] = record.workflow_stage

        # 추가 메타데이터
        if hasattr(record, 'metadata') and record.metadata:
            log_data["metadata"] = record.metadata
        if hasattr(record, 'tags') and record.tags:
            log_data["tags"] = record.tags

        # 도구 실행 정보
        if hasattr(record, 'tool_execution') and record.tool_execution:
            log_data["tool_execution"] = record.tool_execution

        # 에이전트 통신 정보
        if hasattr(record, 'agent_communication') and record.agent_communication:
            log_data["agent_communication"] = record.agent_communication

        # 예외 정보
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    # Agent-specific colors
    AGENT_COLORS = {
        'autonomous_orchestrator': '\033[94m',    # Blue
        'task_analyzer': '\033[95m',              # Magenta
        'task_decomposer': '\033[96m',            # Cyan
        'research_agent': '\033[92m',             # Light Green
        'evaluation_agent': '\033[93m',           # Light Yellow
        'validation_agent': '\033[91m',           # Light Red
        'synthesis_agent': '\033[97m',            # White
        'mcp_integration': '\033[90m',            # Gray
        'llm_methods': '\033[94m',                # Blue
        'RESET': '\033[0m'                        # Reset
    }
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        if not self.use_colors or not hasattr(record, 'levelname'):
            return super().format(record)
        
        # Get base format
        formatted = super().format(record)
        
        # Add colors
        level_color = self.COLORS.get(record.levelname, '')
        agent_color = self.AGENT_COLORS.get(record.name, '')
        reset_color = self.COLORS['RESET']
        
        # Apply colors to different parts
        if level_color:
            formatted = formatted.replace(record.levelname, f"{level_color}{record.levelname}{reset_color}")
        
        if agent_color and record.name in self.AGENT_COLORS:
            formatted = formatted.replace(record.name, f"{agent_color}{record.name}{reset_color}")
        
        return formatted


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    max_file_size: str = "10MB",
    backup_count: int = 5,
    detailed_format: bool = True,
    use_colors: bool = True
) -> logging.Logger:
    """Setup and configure a logger instance.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        detailed_format: Whether to use detailed formatting
        use_colors: Whether to use colors in console output
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatters
    if detailed_format:
        console_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
        file_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
    else:
        console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console formatter with colors
    console_formatter = ColoredFormatter(
        console_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        use_colors=use_colors
    )
    
    # File formatter without colors
    file_formatter = logging.Formatter(
        file_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        try:
            # Create logs directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Parse max file size
            max_bytes = _parse_size(max_file_size)
            
            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            # Fallback to basic file handler if rotation fails
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(level)
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception as e2:
                logger.warning(f"Failed to setup file logging: {e2}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def _parse_size(size_str: str) -> int:
    """Parse size string to bytes.
    
    Args:
        size_str: Size string (e.g., "10MB", "1GB", "100KB")
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    elif size_str.endswith('B'):
        return int(float(size_str[:-1]))
    else:
        # Assume bytes if no unit specified
        return int(float(size_str))


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(logger_name: str, level: str):
    """Set log level for a specific logger.
    
    Args:
        logger_name: Name of the logger
        level: Log level to set
    """
    logger = logging.getLogger(logger_name)
    level_enum = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level_enum)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(level_enum)


def add_file_handler(
    logger_name: str,
    log_file: str,
    level: str = "INFO",
    max_file_size: str = "10MB",
    backup_count: int = 5,
    detailed_format: bool = True
):
    """Add a file handler to an existing logger.
    
    Args:
        logger_name: Name of the logger
        log_file: Path to log file
        level: Log level for the file handler
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files
        detailed_format: Whether to use detailed formatting
    """
    logger = logging.getLogger(logger_name)
    
    # Create formatter
    if detailed_format:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    try:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max file size
        max_bytes = _parse_size(max_file_size)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        logger.warning(f"Failed to add file handler: {e}")


def remove_file_handler(logger_name: str, log_file: str):
    """Remove a specific file handler from a logger.
    
    Args:
        logger_name: Name of the logger
        log_file: Path to the log file to remove
    """
    logger = logging.getLogger(logger_name)
    
    # Find and remove handlers for the specific file
    handlers_to_remove = []
    for handler in logger.handlers:
        if (isinstance(handler, logging.FileHandler) and 
            handler.baseFilename == str(Path(log_file).absolute())):
            handlers_to_remove.append(handler)
    
    for handler in handlers_to_remove:
        logger.removeHandler(handler)
        handler.close()


def setup_default_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    use_colors: bool = True
):
    """Setup default logging configuration for the application.
    
    Args:
        log_level: Default log level
        log_file: Default log file path
        console_output: Whether to output to console
        use_colors: Whether to use colors in console output
    """
    if log_file is None:
        # Create default log file in logs directory
        project_root = Path(__file__).parent.parent.parent
        log_file = str(project_root / "logs" / "local_researcher.log")
    
    # Setup root logger
    root_logger = setup_logger(
        "local_researcher",
        log_level=log_level,
        log_file=log_file,
        console_output=console_output,
        use_colors=use_colors
    )
    
    # Setup common loggers
    setup_logger("research_orchestrator", log_level=log_level, console_output=console_output, use_colors=use_colors)
    setup_logger("gemini_cli_integration", log_level=log_level, console_output=console_output, use_colors=use_colors)
    setup_logger("open_deep_research_adapter", log_level=log_level, console_output=console_output, use_colors=use_colors)
    
    return root_logger


# Convenience function for quick logging setup
def quick_logger(name: str, use_colors: bool = True) -> logging.Logger:
    """Quick setup for a logger with default settings.
    
    Args:
        name: Logger name
        use_colors: Whether to use colors in console output
        
    Returns:
        Logger instance
    """
    return setup_logger(name, console_output=True, use_colors=use_colors)


class EnhancedLogger(logging.Logger):
    """Enhanced logger with structured logging and context support."""

    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
        self._context_stack: List[LogContext] = []
        self._lock = threading.RLock()

    def push_context(self, context: LogContext):
        """Push logging context onto stack."""
        with self._lock:
            self._context_stack.append(context)

    def pop_context(self):
        """Pop logging context from stack."""
        with self._lock:
            if self._context_stack:
                return self._context_stack.pop()
        return None

    def get_current_context(self) -> Optional[LogContext]:
        """Get current logging context."""
        with self._lock:
            return self._context_stack[-1] if self._context_stack else None

    def _log_with_context(
        self,
        level: int,
        msg: str,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
        **kwargs
    ):
        """Log with context information."""
        # 기본 extra 생성
        if extra is None:
            extra = {}

        # 현재 컨텍스트 정보 추가
        context = self.get_current_context()
        if context:
            if context.agent_id:
                extra['agent_id'] = context.agent_id
            if context.session_id:
                extra['session_id'] = context.session_id
            if context.workflow_stage:
                extra['workflow_stage'] = context.workflow_stage
            if context.tool_name:
                extra['tool_name'] = context.tool_name
            if context.operation:
                extra['operation'] = context.operation
            if context.metadata:
                extra['metadata'] = context.metadata
            if context.tags:
                extra['tags'] = context.tags

        # ContextVar 정보 추가
        agent_id = current_agent_id.get()
        session_id = current_session_id.get()
        workflow_stage = current_workflow_stage.get()

        if agent_id and 'agent_id' not in extra:
            extra['agent_id'] = agent_id
        if session_id and 'session_id' not in extra:
            extra['session_id'] = session_id
        if workflow_stage and 'workflow_stage' not in extra:
            extra['workflow_stage'] = workflow_stage

        # 도구 실행 정보 추가
        if 'tool_execution' in kwargs:
            extra['tool_execution'] = kwargs['tool_execution']

        # 에이전트 통신 정보 추가
        if 'agent_communication' in kwargs:
            extra['agent_communication'] = kwargs['agent_communication']

        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)

    def log_tool_execution(self, tool_log: ToolExecutionLog, level: int = logging.INFO):
        """Log tool execution with structured data."""
        message = f"Tool '{tool_log.tool_name}' executed in {tool_log.execution_time:.2f}s"
        if tool_log.success:
            message += f" (success, confidence: {tool_log.confidence:.2f})"
        else:
            message += f" (failed: {tool_log.error_message or 'Unknown error'})"

        self._log_with_context(
            level,
            message,
            (),
            tool_execution={
                'tool_name': tool_log.tool_name,
                'execution_time': tool_log.execution_time,
                'success': tool_log.success,
                'confidence': tool_log.confidence,
                'error_message': tool_log.error_message,
                'input_params': tool_log.input_params,
                'output_summary': tool_log.output_summary,
            }
        )

    def log_agent_communication(self, comm_log: AgentCommunicationLog, level: int = logging.INFO):
        """Log agent communication with structured data."""
        message = f"Agent {comm_log.from_agent}"
        if comm_log.to_agent:
            message += f" → {comm_log.to_agent}"
        message += f": {comm_log.action}"

        if comm_log.result_count > 0:
            message += f" ({comm_log.result_count} results)"
        if comm_log.discussion_topic:
            message += f" - Topic: {comm_log.discussion_topic}"

        self._log_with_context(
            level,
            message,
            (),
            agent_communication={
                'from_agent': comm_log.from_agent,
                'to_agent': comm_log.to_agent,
                'action': comm_log.action,
                'result_count': comm_log.result_count,
                'discussion_topic': comm_log.discussion_topic,
                'shared_data_size': comm_log.shared_data_size,
            }
        )

    def agent_context(self, agent_id: str, session_id: Optional[str] = None):
        """Context manager for agent-specific logging."""
        return AgentContextManager(self, agent_id, session_id)


class AgentContextManager:
    """Context manager for agent-specific logging."""

    def __init__(self, logger: EnhancedLogger, agent_id: str, session_id: Optional[str] = None):
        self.logger = logger
        self.agent_id = agent_id
        self.session_id = session_id
        self.context_token = None

    async def __aenter__(self):
        # ContextVar 설정
        self.context_token = (
            current_agent_id.set(self.agent_id),
            current_session_id.set(self.session_id) if self.session_id else None
        )

        # 로거 컨텍스트 설정
        context = LogContext(
            agent_id=self.agent_id,
            session_id=self.session_id
        )
        self.logger.push_context(context)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # ContextVar 복원
        if self.context_token:
            if self.context_token[1]:  # session_id token
                current_session_id.reset(self.context_token[1])
            current_agent_id.reset(self.context_token[0])

        # 로거 컨텍스트 제거
        self.logger.pop_context()


def setup_enhanced_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    structured_output: bool = False,
    max_file_size: str = "10MB",
    backup_count: int = 5,
    detailed_format: bool = True,
    use_colors: bool = True
) -> EnhancedLogger:
    """Setup enhanced logger with structured logging support.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        structured_output: Whether to use JSON structured output
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        detailed_format: Whether to use detailed formatting
        use_colors: Whether to use colors in console output

    Returns:
        Enhanced logger instance
    """
    # EnhancedLogger 인스턴스 생성
    logger = EnhancedLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 기존 핸들러 제거 방지
    if logger.handlers:
        logger.handlers.clear()

    # 포맷터 설정
    if structured_output:
        console_formatter = StructuredFormatter()
        file_formatter = StructuredFormatter()
    else:
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            use_colors=use_colors
        ) if use_colors else logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # 콘솔 핸들러
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logger.level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 파일 핸들러
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            max_bytes = _parse_size(max_file_size)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logger.level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")

    # 전파 방지
    logger.propagate = False

    return logger


def get_enhanced_logger(name: str) -> EnhancedLogger:
    """Get an existing enhanced logger instance.

    Args:
        name: Logger name

    Returns:
        Enhanced logger instance
    """
    logger = logging.getLogger(name)
    if isinstance(logger, EnhancedLogger):
        return logger

    # EnhancedLogger로 변환
    enhanced_logger = EnhancedLogger(name)
    enhanced_logger.setLevel(logger.level)
    enhanced_logger.handlers = logger.handlers[:]
    enhanced_logger.propagate = logger.propagate

    return enhanced_logger


# 전역 헬퍼 함수들
def set_agent_context(agent_id: str, session_id: Optional[str] = None):
    """Set current agent context for logging."""
    current_agent_id.set(agent_id)
    if session_id:
        current_session_id.set(session_id)


def set_workflow_stage(stage: str):
    """Set current workflow stage for logging."""
    current_workflow_stage.set(stage)


def get_current_agent_context() -> Dict[str, Any]:
    """Get current logging context."""
    return {
        'agent_id': current_agent_id.get(),
        'session_id': current_session_id.get(),
        'workflow_stage': current_workflow_stage.get(),
    }


def log_tool_execution(
    logger: logging.Logger,
    tool_name: str,
    execution_time: float,
    success: bool,
    confidence: float = 0.0,
    error_message: Optional[str] = None,
    input_params: Optional[Dict[str, Any]] = None,
    output_summary: Optional[str] = None
):
    """Helper function to log tool execution."""
    tool_log = ToolExecutionLog(
        tool_name=tool_name,
        execution_time=execution_time,
        success=success,
        confidence=confidence,
        error_message=error_message,
        input_params=input_params,
        output_summary=output_summary
    )

    if isinstance(logger, EnhancedLogger):
        logger.log_tool_execution(tool_log)
    else:
        # 기본 로깅
        status = "success" if success else "failed"
        message = f"Tool '{tool_name}' executed in {execution_time:.2f}s ({status})"
        logger.info(message)


def log_agent_communication(
    logger: logging.Logger,
    from_agent: str,
    action: str,
    to_agent: Optional[str] = None,
    result_count: int = 0,
    discussion_topic: Optional[str] = None,
    shared_data_size: int = 0
):
    """Helper function to log agent communication."""
    comm_log = AgentCommunicationLog(
        from_agent=from_agent,
        to_agent=to_agent,
        action=action,
        result_count=result_count,
        discussion_topic=discussion_topic,
        shared_data_size=shared_data_size
    )

    if isinstance(logger, EnhancedLogger):
        logger.log_agent_communication(comm_log)
    else:
        # 기본 로깅
        message = f"Agent {from_agent}"
        if to_agent:
            message += f" → {to_agent}"
        message += f": {action}"
        logger.info(message)
