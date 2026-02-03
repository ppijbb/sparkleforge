"""
Simple Research Workflow Manager
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStage(Enum):
    """Workflow execution stages."""
    TOPIC_ANALYSIS = "topic_analysis"
    SOURCE_DISCOVERY = "source_discovery"
    CONTENT_GATHERING = "content_gathering"
    REPORT_GENERATION = "report_generation"


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    output: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class ResearchWorkflowManager:
    """Simple research workflow manager."""
    
    def __init__(self, config_manager):
        """Initialize the workflow manager."""
        self.config_manager = config_manager
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.active_workflows: Dict[str, WorkflowResult] = {}
        self.logger = logging.getLogger(__name__)
        
        logger.info("Research Workflow Manager initialized")
    
    async def create_workflow(
        self,
        workflow_type: str,
        topic: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new research workflow."""
        workflow_id = self._generate_workflow_id(topic)
        
        workflow_data = {
            "type": workflow_type,
            "topic": topic,
            "parameters": parameters or {},
            "created_at": datetime.now(),
            "status": WorkflowStatus.PENDING
        }
        
        self.workflows[workflow_id] = workflow_data
        
        logger.info(f"Created workflow {workflow_id} for topic: {topic}")
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> WorkflowResult:
        """Execute a research workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow_data = self.workflows[workflow_id]
        
        # Create workflow result
        result = WorkflowResult(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.active_workflows[workflow_id] = result
        
        try:
            logger.info(f"Starting workflow execution: {workflow_id}")
            
            # Step 1: Topic Analysis
            logger.info("Performing topic analysis...")
            topic_analysis = await self._analyze_topic(workflow_data)
            result.output['topic_analysis'] = topic_analysis
            
            # Step 2: Source Discovery
            logger.info("Discovering sources...")
            source_discovery = await self._discover_sources(workflow_data)
            result.output['source_discovery'] = source_discovery
            
            # Step 3: Content Gathering
            logger.info("Gathering content...")
            content_gathering = await self._gather_content(workflow_data)
            result.output['content_gathering'] = content_gathering
            
            # Step 4: Report Generation
            logger.info("Generating report...")
            report_generation = await self._generate_report(workflow_data, result.output)
            result.output['report_generation'] = report_generation
            
            result.status = WorkflowStatus.COMPLETED
            result.end_time = datetime.now()
            
            logger.info(f"Workflow completed successfully: {workflow_id}")
            
        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.end_time = datetime.now()
            result.error_message = str(e)
            
            logger.error(f"Workflow failed: {workflow_id}, Error: {e}")
        
        return result
    
    async def _analyze_topic(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the research topic."""
        topic = workflow_data["topic"]
        
        # Simple keyword extraction
        keywords = topic.lower().split()
        keywords = [word for word in keywords if len(word) > 2]
        
        return {
            "topic_keywords": keywords,
            "complexity": "medium",
            "estimated_sources": 5,
            "analysis_complete": True
        }
    
    async def _discover_sources(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Discover sources for the topic."""
        from ..research.tools.web_search import WebSearchTool
        
        topic = workflow_data["topic"]
        
        try:
            # Use web search to find sources
            search_tool = WebSearchTool({
                'max_results': 5,
                'timeout': 30
            })
            
            search_results = await search_tool.arun(topic)
            sources = search_results.get('results', [])
            
            return {
                "sources_found": len(sources),
                "source_types": ["web"],
                "sources": sources,
                "discovery_complete": True
            }
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            raise RuntimeError(f"Web search failed: {e}. No fallback data available.")
    
    async def _gather_content(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gather content from sources."""
        import asyncio
        
        topic = workflow_data["topic"]
        
        # Get sources from previous step
        sources = workflow_data.get('output', {}).get('source_discovery', {}).get('sources', [])
        
        # If no sources, raise error instead of creating mock content
        if not sources:
            raise RuntimeError(f"No sources found for topic: {topic}. Please check your search configuration and try again.")
        
        # Gather content from sources
        content_data = []
        total_length = 0
        processed_sources = 0
        
        for source in sources[:3]:  # Limit to first 3 sources
            try:
                if 'url' in source:
                    # Use MCP tools to fetch content from URL
                    from src.core.mcp_integration import execute_tool
                    
                    fetch_result = await execute_tool("fetch", {"url": source['url']})
                    if fetch_result.get('success', False):
                        content = fetch_result.get('data', {}).get('content', '')
                        if len(content) > 100:
                            content_data.append({
                                'title': source.get('title', ''),
                                'content': content[:1000],  # Limit content length
                                'url': source['url']
                            })
                            total_length += len(content)
                            processed_sources += 1
                
                await asyncio.sleep(0.5)  # Be respectful
                
            except Exception as e:
                logger.warning(f"Failed to gather content from {source.get('url', 'unknown')}: {e}")
                continue
        
        # If no content gathered, raise error
        if not content_data:
            logger.error("No content could be gathered from sources")
            raise RuntimeError("No content available for analysis. Cannot proceed without real data.")
        
        return {
            "content_collected": True,
            "total_content_length": total_length,
            "sources_processed": processed_sources,
            "content_data": content_data,
            "gathering_complete": True
        }
    
    async def _generate_report(self, workflow_data: Dict[str, Any], all_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate research report."""
        from ..generation.markdown_generator import MarkdownGenerator
        
        topic = workflow_data["topic"]
        parameters = workflow_data.get("parameters", {})
        
        # Collect data from all steps
        topic_analysis = all_output.get('topic_analysis', {})
        source_discovery = all_output.get('source_discovery', {})
        content_gathering = all_output.get('content_gathering', {})
        
        # Prepare report data
        report_data = {
            'title': f"Research Report: {topic}",
            'topic': topic,
            'domain': parameters.get('domain', 'general'),
            'depth': workflow_data.get('type', 'basic'),
            'summary': f"Research completed on '{topic}' with {source_discovery.get('sources_found', 0)} sources analyzed.",
            'key_findings': [
                f"Found {source_discovery.get('sources_found', 0)} relevant sources",
                f"Processed {content_gathering.get('sources_processed', 0)} sources for content",
                f"Generated comprehensive report on {topic}"
            ],
            'analysis': f"Analysis of {topic} based on gathered content and sources.",
            'insights': [
                f"Topic complexity: {topic_analysis.get('complexity', 'medium')}",
                f"Keywords identified: {', '.join(topic_analysis.get('topic_keywords', [])[:5])}",
                f"Content length: {content_gathering.get('total_content_length', 0)} characters"
            ],
            'conclusion': f"Research on '{topic}' completed successfully with comprehensive analysis.",
            'sources': source_discovery.get('sources', []),
            'content_stats': {
                'word_count': content_gathering.get('total_content_length', 0),
                'reading_time_minutes': max(1, content_gathering.get('total_content_length', 0) // 200),
                'sources_processed': content_gathering.get('sources_processed', 0)
            }
        }
        
        # Generate report
        generator = MarkdownGenerator({
            'output_directory': './outputs',
            'include_toc': True,
            'include_metadata': True
        })
        
        document = await generator.agenerate(report_data)
        
        return {
            "report_generated": True,
            "report_path": document.file_path,
            "report_length": len(document.content),
            "generation_complete": True
        }
    
    def _generate_workflow_id(self, topic: str) -> str:
        """Generate a unique workflow ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = topic.lower().replace(" ", "_")[:20]
        return f"workflow_{topic_slug}_{timestamp}"
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowResult]:
        """Get the status of a workflow."""
        return self.active_workflows.get(workflow_id)
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self.active_workflows:
            result = self.active_workflows[workflow_id]
            if result.status == WorkflowStatus.RUNNING:
                result.status = WorkflowStatus.CANCELLED
                result.end_time = datetime.now()
                logger.info(f"Workflow cancelled: {workflow_id}")
                return True
        return False
    
    def list_workflows(self, status_filter: Optional[WorkflowStatus] = None) -> List[Dict[str, Any]]:
        """List all workflows with optional status filtering."""
        workflows = []
        
        for workflow_id, workflow_data in self.workflows.items():
            if status_filter is None or workflow_data["status"] == status_filter:
                workflow_info = {
                    "id": workflow_id,
                    "type": workflow_data["type"],
                    "topic": workflow_data["topic"],
                    "status": workflow_data["status"],
                    "created_at": workflow_data["created_at"]
                }
                
                if workflow_id in self.active_workflows:
                    result = self.active_workflows[workflow_id]
                    workflow_info.update({
                        "start_time": result.start_time,
                        "end_time": result.end_time
                    })
                
                workflows.append(workflow_info)
        
        return workflows