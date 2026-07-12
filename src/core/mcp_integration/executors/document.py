"""Document tool dispatch (ToolCategory.DOCUMENT): report generation."""
import logging
import time
from typing import Any, Dict

from src.core.tools.registry import ToolResult

logger = logging.getLogger(__name__)


async def _execute_document_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """문서 생성 도구 실행."""
    start_time = time.time()

    try:
        from src.generation.report_generator import ReportGenerator

        generator = ReportGenerator()
        research_data = parameters.get("research_data", {})
        report_type = parameters.get("report_type", "comprehensive")

        if not research_data:
            raise ValueError("research_data parameter is required for document generation")

        # 도구 이름에서 형식 추출
        if tool_name == "generate_pdf":
            output_format = "pdf"
        elif tool_name == "generate_docx":
            output_format = "docx"
        elif tool_name == "generate_pptx":
            output_format = "pptx"
        elif tool_name == "generate_html":
            output_format = "html"
        elif tool_name == "generate_markdown":
            output_format = "markdown"
        else:
            raise ValueError(f"Unknown document tool: {tool_name}")

        # 문서 생성
        file_path = await generator.generate_research_report(
            research_data=research_data,
            report_type=report_type,
            output_format=output_format,
        )

        return ToolResult(
            success=True,
            data={
                "file_path": file_path,
                "format": output_format,
                "report_type": report_type,
            },
            execution_time=time.time() - start_time,
            confidence=0.9,
        )

    except Exception as e:
        logger.error(f"Document tool execution failed: {tool_name} - {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=f"Document tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0,
        )
