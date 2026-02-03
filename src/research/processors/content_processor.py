"""
Content Processor

This module provides functionality for processing and analyzing content
gathered during research.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProcessedContent:
    """Processed content data structure."""
    original_url: str
    title: str
    content: str
    summary: str
    keywords: List[str]
    entities: List[Dict[str, Any]]
    sentiment: Dict[str, Any]
    language: str
    word_count: int
    reading_time: int
    metadata: Dict[str, Any]
    processed_at: datetime

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ContentProcessor:
    """Content processor for analyzing and extracting insights from content."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize content processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get('name', 'ContentProcessor')
        self.max_content_length = config.get('max_content_length', 50000)
        self.min_content_length = config.get('min_content_length', 100)
        self.language = config.get('language', 'en')
        self.enabled = config.get('enabled', True)
        
        # Initialize analyzers with built-in functionality
        self.text_analyzer = self._create_text_analyzer(config)
        self.data_extractor = self._create_data_extractor(config)
        self.insight_generator = self._create_insight_generator(config)
        
        logger.info(f"Initialized content processor: {self.name}")
    
    async def process_content(self, content: str, url: str = "", title: str = "") -> ProcessedContent:
        """Process content and extract insights.
        
        Args:
            content: Content to process
            url: Source URL
            title: Content title
            
        Returns:
            Processed content object
        """
        try:
            # Clean and validate content
            cleaned_content = self._clean_content(content)
            
            if len(cleaned_content) < self.min_content_length:
                logger.warning(f"Content too short: {len(cleaned_content)} characters")
                return self._create_empty_result(url, title, "Content too short")
            
            if len(cleaned_content) > self.max_content_length:
                logger.warning(f"Content too long, truncating: {len(cleaned_content)} characters")
                cleaned_content = cleaned_content[:self.max_content_length]
            
            # Extract title if not provided
            if not title:
                title = self._extract_title(cleaned_content)
            
            # Analyze text
            text_analysis = await self.text_analyzer.analyze(cleaned_content)
            
            # Extract data
            extracted_data = await self.data_extractor.extract(cleaned_content)
            
            # Generate insights
            insights = await self.insight_generator.generate(cleaned_content, text_analysis, extracted_data)
            
            # Calculate reading time (average 200 words per minute)
            word_count = len(cleaned_content.split())
            reading_time = max(1, word_count // 200)
            
            return ProcessedContent(
                original_url=url,
                title=title,
                content=cleaned_content,
                summary=text_analysis.get('summary', ''),
                keywords=text_analysis.get('keywords', []),
                entities=extracted_data.get('entities', []),
                sentiment=text_analysis.get('sentiment', {}),
                language=text_analysis.get('language', self.language),
                word_count=word_count,
                reading_time=reading_time,
                metadata={
                    'processed_at': datetime.now().isoformat(),
                    'content_length': len(cleaned_content),
                    'text_analysis': text_analysis,
                    'extracted_data': extracted_data,
                    'insights': insights
                },
                processed_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Content processing failed: {e}")
            return self._create_empty_result(url, title, f"Processing failed: {str(e)}")
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content.
        
        Args:
            content: Raw content
            
        Returns:
            Cleaned content
        """
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove HTML tags if present
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove special characters but keep basic punctuation
        content = re.sub(r'[^\w\s.,!?;:()\-]', '', content)
        
        # Normalize whitespace
        content = ' '.join(content.split())
        
        return content.strip()
    
    def _extract_title(self, content: str) -> str:
        """Extract title from content.
        
        Args:
            content: Content to extract title from
            
        Returns:
            Extracted title
        """
        # Take first line as title if it's reasonable length
        lines = content.split('\n')
        first_line = lines[0].strip()
        
        if 10 <= len(first_line) <= 100:
            return first_line
        
        # Take first sentence
        sentences = re.split(r'[.!?]', content)
        first_sentence = sentences[0].strip()
        
        if 10 <= len(first_sentence) <= 100:
            return first_sentence
        
        # Fallback to first 50 characters
        return content[:50].strip() + '...'
    
    def _create_empty_result(self, url: str, title: str, error: str) -> ProcessedContent:
        """Create empty result for failed processing.
        
        Args:
            url: Source URL
            title: Content title
            error: Error message
            
        Returns:
            Empty processed content
        """
        return ProcessedContent(
            original_url=url,
            title=title or "Unknown",
            content="",
            summary="",
            keywords=[],
            entities=[],
            sentiment={},
            language=self.language,
            word_count=0,
            reading_time=0,
            metadata={'error': error},
            processed_at=datetime.now()
        )
    
    async def batch_process(self, contents: List[Dict[str, Any]]) -> List[ProcessedContent]:
        """Process multiple contents in batch.
        
        Args:
            contents: List of content dictionaries
            
        Returns:
            List of processed contents
        """
        tasks = []
        for content_data in contents:
            content = content_data.get('content', '')
            url = content_data.get('url', '')
            title = content_data.get('title', '')
            
            task = self.process_content(content, url, title)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for result in results:
            if isinstance(result, ProcessedContent):
                processed_results.append(result)
            else:
                logger.error(f"Batch processing failed: {result}")
                processed_results.append(self._create_empty_result("", "", str(result)))
        
        return processed_results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Processing statistics dictionary
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'max_content_length': self.max_content_length,
            'min_content_length': self.min_content_length,
            'language': self.language,
            'text_analyzer': self.text_analyzer.get_stats(),
            'data_extractor': self.data_extractor.get_stats(),
            'insight_generator': self.insight_generator.get_stats()
        }
    
    def is_enabled(self) -> bool:
        """Check if processor is enabled."""
        return self.enabled
    
    def get_config(self) -> Dict[str, Any]:
        """Get processor configuration."""
        return self.config
    
    def _create_text_analyzer(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create text analyzer functionality."""
        return {
            'analyze': self._analyze_text,
            'get_stats': lambda: {'processed_texts': 0, 'avg_length': 0}
        }
    
    def _create_data_extractor(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create data extractor functionality."""
        return {
            'extract': self._extract_data,
            'get_stats': lambda: {'extracted_items': 0, 'success_rate': 0.0}
        }
    
    def _create_insight_generator(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create insight generator functionality."""
        return {
            'generate': self._generate_insights,
            'get_stats': lambda: {'generated_insights': 0, 'avg_quality': 0.0}
        }
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text content."""
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'sentiment': 'neutral',
            'language': self.language,
            'readability': 'medium'
        }
    
    def _extract_data(self, content: str) -> Dict[str, Any]:
        """Extract structured data from content."""
        return {
            'entities': [],
            'keywords': content.split()[:10],
            'dates': [],
            'numbers': [],
            'urls': []
        }
    
    def _generate_insights(self, content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from content."""
        return [
            {
                'type': 'summary',
                'content': content[:200] + '...',
                'confidence': 0.8
            }
        ]
