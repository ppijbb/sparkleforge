"""
Base Tool for Research System
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories."""
    SEARCH = "search"
    DATA = "data"
    CODE = "code"
    ACADEMIC = "academic"
    BUSINESS = "business"
    GENERAL = "general"


@dataclass
class ToolResponse:
    """Standardized tool response."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseResearchTool(ABC):
    """Base class for all research tools."""
    
    def __init__(self, name: str, category: ToolCategory):
        self.name = name
        self.category = category
        self.logger = logging.getLogger(f"tool.{name}")
    
    @abstractmethod
    async def run(self, query: str, **kwargs) -> ToolResponse:
        """Run the tool asynchronously."""
        pass
    
    @abstractmethod
    def run_sync(self, query: str, **kwargs) -> ToolResponse:
        """Run the tool synchronously."""
        pass
    
    def get_tool_category(self) -> ToolCategory:
        """Get tool category."""
        return self.category
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get tool health status."""
        return {
            "name": self.name,
            "category": self.category.value,
            "status": "healthy",
            "last_check": None
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get tool performance metrics."""
        return {
            "name": self.name,
            "total_calls": 0,
            "success_rate": 1.0,
            "average_response_time": 0.0
        }
