"""
Custom Exception Classes for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade error handling with standardized exception types for
all components of the advanced multi-agent research system.
"""

from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification."""
    CONFIGURATION = "configuration"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    EXECUTION = "execution"
    RESOURCE = "resource"
    INTEGRATION = "integration"
    DATA = "data"
    SYSTEM = "system"


class BaseResearcherException(Exception):
    """Base exception class for all Local Researcher exceptions."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.suggestions = suggestions or []
        self.context = context or {}
    
    def __str__(self) -> str:
        """Return formatted string representation of the exception."""
        base_msg = f"[{self.category.value.upper()}] {self.message}"
        if self.details:
            base_msg += f" | Details: {self.details}"
        return base_msg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "suggestions": self.suggestions,
            "context": self.context
        }


# Configuration Exceptions
class ConfigurationError(BaseResearcherException):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            details={"config_key": config_key} if config_key else {},
            **kwargs
        )


class MissingConfigurationError(ConfigurationError):
    """Missing required configuration."""
    
    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            f"Missing required configuration: {config_key}",
            config_key=config_key,
            severity=ErrorSeverity.CRITICAL,
            suggestions=[
                f"Set the {config_key} environment variable",
                f"Add {config_key} to your configuration file",
                "Check the documentation for required configuration"
            ],
            **kwargs
        )


class InvalidConfigurationError(ConfigurationError):
    """Invalid configuration value."""
    
    def __init__(self, config_key: str, value: Any, expected: str, **kwargs):
        super().__init__(
            f"Invalid configuration for {config_key}: {value}. Expected: {expected}",
            config_key=config_key,
            severity=ErrorSeverity.HIGH,
            details={"actual_value": value, "expected": expected},
            suggestions=[
                f"Check the {config_key} value in your configuration",
                f"Ensure {config_key} matches the expected format: {expected}",
                "Refer to the configuration documentation"
            ],
            **kwargs
        )


# Network and API Exceptions
class NetworkError(BaseResearcherException):
    """Network-related errors."""
    
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            details={"url": url, "status_code": status_code} if url or status_code else {},
            suggestions=[
                "Check your internet connection",
                "Verify the API endpoint is accessible",
                "Check if the service is temporarily unavailable"
            ],
            **kwargs
        )


class APIConnectionError(NetworkError):
    """API connection errors."""
    
    def __init__(self, service: str, url: str, **kwargs):
        super().__init__(
            f"Failed to connect to {service} API: {url}",
            url=url,
            severity=ErrorSeverity.HIGH,
            details={"service": service},
            suggestions=[
                f"Check if {service} service is running",
                "Verify the API endpoint URL",
                "Check your network connectivity",
                "Review API rate limits and quotas"
            ],
            **kwargs
        )


class APIResponseError(NetworkError):
    """API response errors."""
    
    def __init__(self, service: str, status_code: int, response_text: str, **kwargs):
        super().__init__(
            f"{service} API returned error {status_code}: {response_text}",
            status_code=status_code,
            severity=ErrorSeverity.MEDIUM,
            details={"service": service, "response_text": response_text},
            suggestions=[
                "Check the API documentation for error codes",
                "Verify your request parameters",
                "Check your API key and permissions",
                "Review rate limits and quotas"
            ],
            **kwargs
        )


# Authentication Exceptions
class AuthenticationError(BaseResearcherException):
    """Authentication-related errors."""
    
    def __init__(self, service: str, **kwargs):
        super().__init__(
            f"Authentication failed for {service}",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.CRITICAL,
            details={"service": service},
            suggestions=[
                f"Check your {service} API key",
                "Verify the API key has correct permissions",
                "Ensure the API key is not expired",
                "Check if the service requires additional authentication"
            ],
            **kwargs
        )


class InvalidAPIKeyError(AuthenticationError):
    """Invalid API key error."""
    
    def __init__(self, service: str, **kwargs):
        super().__init__(
            f"Invalid API key for {service}",
            service=service,
            suggestions=[
                f"Verify your {service} API key is correct",
                "Check if the API key has been regenerated",
                "Ensure the API key is properly set in environment variables",
                f"Get a new API key from {service} if needed"
            ],
            **kwargs
        )


# Validation Exceptions
class ValidationError(BaseResearcherException):
    """Data validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            details={"field": field, "value": value} if field or value else {},
            suggestions=[
                "Check the input data format",
                "Verify all required fields are provided",
                "Ensure data types match expected format"
            ],
            **kwargs
        )


class DataValidationError(ValidationError):
    """Data validation errors."""
    
    def __init__(self, field: str, value: Any, expected_type: str, **kwargs):
        super().__init__(
            f"Invalid data for field '{field}': {value}. Expected type: {expected_type}",
            field=field,
            value=value,
            details={"expected_type": expected_type},
            **kwargs
        )


# Execution Exceptions
class ExecutionError(BaseResearcherException):
    """Execution-related errors."""
    
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.HIGH,
            details={"component": component} if component else {},
            suggestions=[
                "Check the component logs for more details",
                "Verify all dependencies are installed",
                "Check system resources and permissions"
            ],
            **kwargs
        )


class TaskExecutionError(ExecutionError):
    """Task execution errors."""
    
    def __init__(self, task_name: str, error_details: str, **kwargs):
        super().__init__(
            f"Task '{task_name}' execution failed: {error_details}",
            component=task_name,
            details={"error_details": error_details},
            suggestions=[
                f"Check the {task_name} task configuration",
                "Verify all required inputs are provided",
                "Check if the task dependencies are available",
                "Review the task execution logs"
            ],
            **kwargs
        )


# Resource Exceptions
class ResourceError(BaseResearcherException):
    """Resource-related errors."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.MEDIUM,
            details={"resource_type": resource_type} if resource_type else {},
            suggestions=[
                "Check available system resources",
                "Free up memory or disk space if needed",
                "Check resource limits and quotas"
            ],
            **kwargs
        )


class InsufficientResourcesError(ResourceError):
    """Insufficient resources error."""
    
    def __init__(self, resource_type: str, required: str, available: str, **kwargs):
        super().__init__(
            f"Insufficient {resource_type}: required {required}, available {available}",
            resource_type=resource_type,
            details={"required": required, "available": available},
            suggestions=[
                f"Free up {resource_type} resources",
                f"Request additional {resource_type} if possible",
                "Optimize resource usage",
                "Check resource allocation settings"
            ],
            **kwargs
        )


# Integration Exceptions
class IntegrationError(BaseResearcherException):
    """Integration-related errors."""
    
    def __init__(self, message: str, service: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.INTEGRATION,
            severity=ErrorSeverity.HIGH,
            details={"service": service} if service else {},
            suggestions=[
                "Check the service integration configuration",
                "Verify the service is available and accessible",
                "Check API compatibility and version requirements"
            ],
            **kwargs
        )


class MCPIntegrationError(IntegrationError):
    """MCP integration errors."""
    
    def __init__(self, tool_name: str, error_details: str, **kwargs):
        super().__init__(
            f"MCP tool '{tool_name}' integration failed: {error_details}",
            service="MCP",
            details={"tool_name": tool_name, "error_details": error_details},
            suggestions=[
                f"Check MCP tool '{tool_name}' configuration",
                "Verify the MCP tool is properly installed",
                "Check MCP tool dependencies and requirements",
                "Review MCP tool logs for more details"
            ],
            **kwargs
        )


# Data Exceptions
class DataError(BaseResearcherException):
    """Data-related errors."""
    
    def __init__(self, message: str, data_type: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA,
            severity=ErrorSeverity.MEDIUM,
            details={"data_type": data_type} if data_type else {},
            suggestions=[
                "Check data format and structure",
                "Verify data integrity and completeness",
                "Check data source accessibility"
            ],
            **kwargs
        )


class DataProcessingError(DataError):
    """Data processing errors."""
    
    def __init__(self, operation: str, error_details: str, **kwargs):
        super().__init__(
            f"Data processing failed during '{operation}': {error_details}",
            details={"operation": operation, "error_details": error_details},
            suggestions=[
                f"Check the '{operation}' operation parameters",
                "Verify input data format and quality",
                "Check data processing dependencies",
                "Review data processing logs"
            ],
            **kwargs
        )


# System Exceptions
class SystemError(BaseResearcherException):
    """System-level errors."""
    
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            details={"component": component} if component else {},
            suggestions=[
                "Check system logs for more details",
                "Verify system requirements and dependencies",
                "Check system permissions and access rights",
                "Restart the system if necessary"
            ],
            **kwargs
        )


class InitializationError(SystemError):
    """System initialization errors."""
    
    def __init__(self, component: str, error_details: str, **kwargs):
        super().__init__(
            f"Failed to initialize {component}: {error_details}",
            component=component,
            details={"error_details": error_details},
            suggestions=[
                f"Check {component} configuration",
                f"Verify {component} dependencies are installed",
                f"Check {component} logs for initialization errors",
                "Ensure all required services are running"
            ],
            **kwargs
        )


# Utility functions
def create_exception_from_dict(exception_data: Dict[str, Any]) -> BaseResearcherException:
    """Create exception from dictionary data."""
    exception_type = exception_data.get("type", "BaseResearcherException")
    message = exception_data.get("message", "Unknown error")
    category = ErrorCategory(exception_data.get("category", "system"))
    severity = ErrorSeverity(exception_data.get("severity", "medium"))
    details = exception_data.get("details", {})
    suggestions = exception_data.get("suggestions", [])
    context = exception_data.get("context", {})
    
    # Map exception types to classes
    exception_classes = {
        "ConfigurationError": ConfigurationError,
        "MissingConfigurationError": MissingConfigurationError,
        "InvalidConfigurationError": InvalidConfigurationError,
        "NetworkError": NetworkError,
        "APIConnectionError": APIConnectionError,
        "APIResponseError": APIResponseError,
        "AuthenticationError": AuthenticationError,
        "InvalidAPIKeyError": InvalidAPIKeyError,
        "ValidationError": ValidationError,
        "DataValidationError": DataValidationError,
        "ExecutionError": ExecutionError,
        "TaskExecutionError": TaskExecutionError,
        "ResourceError": ResourceError,
        "InsufficientResourcesError": InsufficientResourcesError,
        "IntegrationError": IntegrationError,
        "MCPIntegrationError": MCPIntegrationError,
        "DataError": DataError,
        "DataProcessingError": DataProcessingError,
        "SystemError": SystemError,
        "InitializationError": InitializationError,
    }
    
    exception_class = exception_classes.get(exception_type, BaseResearcherException)
    
    return exception_class(
        message=message,
        category=category,
        severity=severity,
        details=details,
        suggestions=suggestions,
        context=context
    )
