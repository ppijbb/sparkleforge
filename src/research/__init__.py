"""
Research Package

This package contains modules for managing research workflows
and coordinating research tasks.
"""

from .workflow_manager import ResearchWorkflowManager, WorkflowStatus, WorkflowStage

__all__ = [
    'ResearchWorkflowManager',
    'WorkflowStatus',
    'WorkflowStage'
]
