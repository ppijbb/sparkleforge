"""
Simple Markdown Document Generator
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Generated document."""
    content: str
    file_path: str
    metadata: Dict[str, Any] = None


class MarkdownGenerator:
    """Simple markdown document generator."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize markdown generator."""
        self.config = config
        self.output_directory = config.get('output_directory', './outputs')
        self.include_toc = config.get('include_toc', True)
        self.include_metadata = config.get('include_metadata', True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directory if it doesn't exist
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
    
    def generate(self, data: Dict[str, Any]) -> Document:
        """Generate markdown document synchronously."""
        try:
            return asyncio.run(self.agenerate(data))
        except Exception as e:
            self.logger.error(f"Markdown generation failed: {e}")
            raise RuntimeError(f"Markdown generation failed: {e}. No fallback available.")
    
    async def agenerate(self, data: Dict[str, Any]) -> Document:
        """Generate markdown document asynchronously."""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_slug = data.get('topic', 'research').lower().replace(' ', '_')[:20]
            filename = f"research_report_{topic_slug}_{timestamp}.md"
            file_path = Path(self.output_directory) / filename
            
            # Generate content
            content = self._generate_content(data)
            
            # Write file
            file_path.write_text(content, encoding='utf-8')
            
            return Document(
                content=content,
                file_path=str(file_path),
                metadata={
                    'topic': data.get('topic', ''),
                    'generated_at': datetime.now().isoformat(),
                    'word_count': len(content.split())
                }
            )
            
        except Exception as e:
            self.logger.error(f"Markdown generation failed: {e}")
            raise RuntimeError(f"Markdown generation failed: {e}. No fallback available.")
    
    def _generate_content(self, data: Dict[str, Any]) -> str:
        """Generate markdown content."""
        lines = []
        
        # Title
        title = data.get('title', 'Research Report')
        lines.append(f"# {title}")
        lines.append("")
        
        # Metadata
        if self.include_metadata:
            lines.append("## Report Information")
            lines.append(f"- **Topic**: {data.get('topic', 'N/A')}")
            lines.append(f"- **Domain**: {data.get('domain', 'N/A')}")
            lines.append(f"- **Depth**: {data.get('depth', 'N/A')}")
            lines.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
        
        # Table of Contents
        if self.include_toc:
            lines.append("## Table of Contents")
            lines.append("- [Summary](#summary)")
            lines.append("- [Key Findings](#key-findings)")
            lines.append("- [Analysis](#analysis)")
            lines.append("- [Insights](#insights)")
            lines.append("- [Conclusion](#conclusion)")
            lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append(data.get('summary', 'No summary available.'))
        lines.append("")
        
        # Key Findings
        key_findings = data.get('key_findings', [])
        if key_findings:
            lines.append("## Key Findings")
            for i, finding in enumerate(key_findings, 1):
                lines.append(f"{i}. {finding}")
            lines.append("")
        
        # Analysis
        lines.append("## Analysis")
        lines.append(data.get('analysis', 'No analysis available.'))
        lines.append("")
        
        # Insights
        insights = data.get('insights', [])
        if insights:
            lines.append("## Insights")
            for i, insight in enumerate(insights, 1):
                lines.append(f"{i}. {insight}")
            lines.append("")
        
        # Sources
        sources = data.get('sources', [])
        if sources:
            lines.append("## Sources")
            for i, source in enumerate(sources[:10], 1):  # Limit to 10 sources
                title = source.get('title', 'Untitled')
                url = source.get('url', '#')
                lines.append(f"{i}. [{title}]({url})")
            lines.append("")
        
        # Content Stats
        content_stats = data.get('content_stats', {})
        if content_stats:
            lines.append("## Content Statistics")
            lines.append(f"- **Word Count**: {content_stats.get('word_count', 0):,}")
            lines.append(f"- **Reading Time**: {content_stats.get('reading_time_minutes', 0)} minutes")
            lines.append(f"- **Sources Processed**: {content_stats.get('sources_processed', 0)}")
            lines.append("")
        
        # Conclusion
        lines.append("## Conclusion")
        lines.append(data.get('conclusion', 'Research completed successfully.'))
        lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("*Generated by Local Researcher*")
        
        return "\n".join(lines)
    
    # Fallback document creation removed - no fallback responses allowed