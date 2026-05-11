import json
import logging

try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None

from src.core.document_processing.docling_processor import DoclingProcessor
from src.storage.hybrid_storage import HybridStorage

logger = logging.getLogger(__name__)

# Initialize MCP server
if FastMCP:
    mcp = FastMCP("docling")
else:
    mcp = None

# Initialize processor and storage
processor = DoclingProcessor()
storage = HybridStorage()


@mcp.tool()
async def process_document(
    source: str, user_id: str = "default_user", instruction: str | None = None
) -> str:
    """Process a document (PDF, DOCX, PPTX, XLSX, HTML) using Docling and save to history.

    Args:
        source: File path or URL of the document.
        user_id: ID of the user requesting the processing.
        instruction: Optional user instruction to guide selective extraction.

    Returns:
        JSON string containing processing results, summary, and storage path.
    """
    try:
        # 1. Process document
        result = await processor.process(source, user_id, instruction=instruction)

        if not result.get("success"):
            return json.dumps({"error": result.get("error"), "source": source})

        # 2. Store to history
        await processor.store_to_history(storage, result)

        # 3. Return summary for the agent
        summary = {
            "status": "success",
            "doc_id": result["doc_id"],
            "title": result["metadata"]["title"],
            "tables_found": result["tables_count"],
            "storage_path": result["storage_path"],
            "markdown_preview": (
                result["markdown"][:500] + "..."
                if len(result["markdown"]) > 500
                else result["markdown"]
            ),
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Error in process_document tool: {e}")
        return json.dumps({"error": str(e), "source": source})


if __name__ == "__main__":
    if mcp:
        mcp.run()
    else:
        print("FastMCP not installed.")
