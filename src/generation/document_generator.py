"""
Document Generator

This module provides the base functionality for generating various types
of documents from research results.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type enumeration."""
    MARKDOWN = "markdown"
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"
    TXT = "txt"


@dataclass
class DocumentTemplate:
    """Document template data structure."""
    name: str
    template_type: DocumentType
    content: str
    metadata: Dict[str, Any] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class GeneratedDocument:
    """Generated document data structure."""
    document_id: str
    title: str
    content: str
    document_type: DocumentType
    file_path: str
    file_size: int
    created_at: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentGenerator(ABC):
    """Base class for document generators."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize document generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.output_directory = config.get('output_directory', './outputs')
        self.template_directory = config.get('template_directory', './templates')
        self.enabled = config.get('enabled', True)
        
        # Create output directory if it doesn't exist
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
        Path(self.template_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized document generator: {self.name}")
    
    @abstractmethod
    async def generate(self, data: Dict[str, Any], **kwargs) -> GeneratedDocument:
        """Generate a document.
        
        Args:
            data: Data to generate document from
            **kwargs: Additional generation parameters
            
        Returns:
            Generated document object
        """
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[DocumentType]:
        """Get supported document types.
        
        Returns:
            List of supported document types
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if generator is enabled."""
        return self.enabled
    
    def get_config(self) -> Dict[str, Any]:
        """Get generator configuration."""
        return self.config
    
    def _generate_document_id(self) -> str:
        """Generate a unique document ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"doc_{timestamp}"
    
    def _get_output_path(self, filename: str) -> str:
        """Get full output path for a filename.
        
        Args:
            filename: Filename
            
        Returns:
            Full output path
        """
        return str(Path(self.output_directory) / filename)
    
    def _get_template_path(self, template_name: str) -> str:
        """Get full template path for a template name.
        
        Args:
            template_name: Template name
            
        Returns:
            Full template path
        """
        return str(Path(self.template_directory) / template_name)
    
    def _load_template(self, template_name: str) -> Optional[str]:
        """Load template content.
        
        Args:
            template_name: Template name
            
        Returns:
            Template content or None if not found
        """
        try:
            template_path = self._get_template_path(template_name)
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
            return None
    
    def _save_document(self, content: str, filename: str) -> str:
        """Save document content to file.
        
        Args:
            content: Document content
            filename: Filename
            
        Returns:
            Full file path
        """
        file_path = self._get_output_path(filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Document saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save document {filename}: {e}")
            raise
    
    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes.
        
        Args:
            file_path: File path
            
        Returns:
            File size in bytes
        """
        try:
            return Path(file_path).stat().st_size
        except Exception as e:
            logger.error(f"Failed to get file size for {file_path}: {e}")
            return 0
    
    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp for display.
        
        Args:
            timestamp: Timestamp to format
            
        Returns:
            Formatted timestamp string
        """
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"
    
    def __repr__(self) -> str:
        return self.__str__()
