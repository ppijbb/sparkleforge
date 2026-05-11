import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class DoclingProcessor:
    """Document processing using IBM Docling for high-quality extraction."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or "./storage/processed_documents")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._converter = None

    def _get_converter(self):
        """Lazy initialization of Docling converter to save resources."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter

                self._converter = DocumentConverter()
                logger.info("Docling DocumentConverter initialized.")
            except ImportError:
                logger.error(
                    "Docling library not installed. Please install it with 'pip install docling'."
                )
                raise ImportError("Docling not installed.")
        return self._converter

    def _infer_extraction_plan(
        self, instruction: str | None
    ) -> Tuple[str, int | None, Dict[str, bool]]:
        """Decide what to keep based on user instruction.

        Returns:
          - extraction_mode: human readable string
          - max_chars: optional truncation budget for summary-like requests
          - keep_flags: which item types to keep
        """
        if not instruction:
            return "full", None, {"text": True, "tables": True, "key_value": True}

        q = instruction.strip().lower()

        wants_summary = any(k in q for k in ["요약", "핵심", "정리", "summary", "summarize"])
        # "표만" 같은 명시가 있으면 tables_only로 축소
        tables_only = any(k in q for k in ["표만", "tables only", "tables_only", "table only"])
        wants_tables = any(k in q for k in ["표", "table", "tables", "엑셀", "xlsx", "xls"])
        wants_key_value = any(
            k in q
            for k in [
                "키값",
                "key-value",
                "key value",
                "kv",
                "form",
                "필드",
                "항목",
                "field",
            ]
        )

        # default: don't try to be too aggressive; keep text even if tables are requested
        if tables_only and wants_tables:
            keep = {"text": False, "tables": True, "key_value": False}
            return "tables_only", 2500 if wants_summary else None, keep

        if wants_key_value and not wants_tables:
            keep = {"text": False, "tables": False, "key_value": True}
            return "key_value_only", 2500 if wants_summary else None, keep

        if wants_tables or wants_key_value:
            keep = {
                "text": True,
                "tables": wants_tables,
                "key_value": wants_key_value,
            }
            return "tables_and_text", 2500 if wants_summary else None, keep

        # no explicit extraction intent -> keep everything
        return (
            "full",
            2500 if wants_summary else None,
            {"text": True, "tables": True, "key_value": True},
        )

    def _table_to_markdown(self, doc: Any, table_item: Any) -> str:
        """Convert Docling TableItem into a markdown table (best-effort)."""
        try:
            grid = table_item.data.grid
            if not grid:
                return ""

            rows: List[List[str]] = []
            for row in grid:
                row_text: List[str] = []
                for cell in row:
                    if hasattr(cell, "_get_text"):
                        cell_text = cell._get_text(doc=doc).strip()
                    else:
                        cell_text = str(getattr(cell, "text", "")).strip()
                    row_text.append(cell_text)
                rows.append(row_text)

            max_cols = max((len(r) for r in rows), default=0)
            if max_cols == 0:
                return ""
            # pad rows
            for r in rows:
                if len(r) < max_cols:
                    r.extend([""] * (max_cols - len(r)))

            header = rows[0]
            sep = ["---"] * max_cols

            md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
            for body_row in rows[1:]:
                md_lines.append("| " + " | ".join(body_row) + " |")
            return "\n".join(md_lines).strip()
        except Exception as e:
            logger.warning("Failed to convert table to markdown: %s", e)
            # fallback: at least provide a placeholder
            try:
                return f"`table(rows={table_item.data.num_rows}, cols={table_item.data.num_cols})`"
            except Exception:
                return "`table`"

    async def process(
        self,
        source: str,
        user_id: str = "default_user",
        instruction: str | None = None,
    ) -> Dict[str, Any]:
        """Process a document (file path or URL) and extract resources."""
        logger.info(f"Processing document: {source}")

        try:
            converter = self._get_converter()
            result = converter.convert(source)

            doc = result.document
            document_dict = doc.export_to_dict()
            extraction_mode, max_chars, keep_flags = self._infer_extraction_plan(instruction)

            # Default(full) mode: preserve existing behavior for compatibility.
            # If the caller provided an instruction but it doesn't request a selective scope,
            # keep export_to_markdown() output to avoid formatting/regression.
            should_use_full_export = (
                extraction_mode == "full"
                and max_chars is None
                and keep_flags.get("text") is True
                and keep_flags.get("tables") is True
                and keep_flags.get("key_value") is True
            )

            if should_use_full_export:
                markdown_content = doc.export_to_markdown()
            else:
                # Selective extraction: reconstruct Markdown from chosen item types.
                from docling_core.transforms.serializer.markdown import (
                    MarkdownDocSerializer,
                )
                from docling_core.types.doc import (
                    ContentLayer,
                    KeyValueItem,
                    TableItem,
                    TextItem,
                )

                parts: List[str] = []
                total_len = 0
                key_value_serializer: MarkdownDocSerializer | None = None
                if keep_flags.get("key_value"):
                    key_value_serializer = MarkdownDocSerializer(doc=doc, traverse_pictures=False)

                def _append(text: str) -> None:
                    nonlocal total_len
                    if not text:
                        return
                    if max_chars and total_len >= max_chars:
                        return
                    parts.append(text)
                    total_len += len(text)

                for item, _level in doc.iterate_items(
                    with_groups=False,
                    traverse_pictures=False,
                    included_content_layers={ContentLayer.BODY},
                ):
                    # Text
                    if keep_flags.get("text") and isinstance(item, TextItem):
                        text = (item.text or "").strip()
                        if text:
                            # Keep paragraphs short; don't spam whitespace
                            _append(text + "\n")
                    # Tables
                    if keep_flags.get("tables") and isinstance(item, TableItem):
                        md_table = self._table_to_markdown(doc, item)
                        if md_table:
                            _append("\n## Table\n" + md_table + "\n")
                    # Key-value
                    if keep_flags.get("key_value") and isinstance(item, KeyValueItem):
                        if key_value_serializer is not None:
                            ser_res = key_value_serializer.serialize(item=item)
                            kv_text = (ser_res.text or "").strip()
                            if kv_text:
                                _append(kv_text + "\n")

                    if max_chars and total_len >= max_chars:
                        break

                markdown_content = "\n".join([p.strip() for p in parts if p.strip()])

            # Generate a unique ID for this processing session
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Extract metadata
            metadata = {
                "source": source,
                "processed_at": datetime.now().isoformat(),
                "file_type": Path(source).suffix if os.path.exists(source) else "url",
                "doc_id": doc_id,
                "user_id": user_id,
                "title": document_dict.get("metadata", {}).get("title", Path(source).stem),
                "extraction_mode": extraction_mode,
            }

            # Extract tables if any
            try:
                # Prefer doc.tables since it is the canonical representation.
                tables = list(getattr(doc, "tables", []) or [])
            except Exception:
                tables = []

            table_entries: List[Dict[str, Any]] = []
            if tables and keep_flags.get("tables"):
                for i, table_item in enumerate(tables):
                    md_table = self._table_to_markdown(doc, table_item)
                    if not md_table:
                        continue

                    # Best-effort provenance info (may be empty depending on parser)
                    prov_info: Dict[str, Any] = {}
                    try:
                        prov_list = getattr(table_item, "prov", None) or []
                        if prov_list:
                            first = prov_list[0]
                            if hasattr(first, "page_no"):
                                prov_info["page_no"] = first.page_no
                            if hasattr(first, "bbox"):
                                bbox_obj = getattr(first, "bbox", None)
                                # Ensure provenance metadata is JSON-serializable.
                                if bbox_obj is None:
                                    prov_info["bbox"] = None
                                elif all(hasattr(bbox_obj, k) for k in ("l", "t", "r", "b")):
                                    prov_info["bbox"] = {
                                        "l": getattr(bbox_obj, "l", None),
                                        "t": getattr(bbox_obj, "t", None),
                                        "r": getattr(bbox_obj, "r", None),
                                        "b": getattr(bbox_obj, "b", None),
                                    }
                                elif isinstance(bbox_obj, (list, tuple)):
                                    prov_info["bbox"] = list(bbox_obj)
                                else:
                                    prov_info["bbox"] = str(bbox_obj)
                    except Exception:
                        prov_info = {}

                    table_entries.append(
                        {
                            "table_index": i,
                            "markdown": md_table,
                            "num_rows": getattr(
                                getattr(table_item, "data", None), "num_rows", None
                            ),
                            "num_cols": getattr(
                                getattr(table_item, "data", None), "num_cols", None
                            ),
                            "provenance": prov_info,
                        }
                    )

            # Save results to disk
            doc_dir = self.output_dir / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            with open(doc_dir / "content.md", "w", encoding="utf-8") as f:
                f.write(markdown_content)

            with open(doc_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Prepare summarized response
            return {
                "success": True,
                "doc_id": doc_id,
                "metadata": metadata,
                "markdown": markdown_content,
                "tables_count": len(table_entries),
                "storage_path": str(doc_dir),
                "extraction_mode": extraction_mode,
                "table_entries": table_entries,
            }

        except Exception as e:
            logger.error(f"Docling processing failed: {e}")
            return {"success": False, "error": str(e), "source": source}

    async def store_to_history(self, storage: Any, process_result: Dict[str, Any]):
        """Store the processed document result into HybridStorage history."""
        if not process_result.get("success"):
            return False

        metadata = process_result["metadata"]
        res = await storage.store_research(
            research_id=process_result["doc_id"],
            user_id=metadata["user_id"],
            topic=metadata["title"],
            content=process_result["markdown"],
            results={"tables_count": process_result["tables_count"]},
            metadata=metadata,
            summary=f"Processed document from {metadata['source']}.",
            keywords=[metadata["file_type"], "docling_processed"],
        )

        # Store each table as a separate history entry for finer retrieval.
        # This is intentionally best-effort: if table_entries is absent, we only store the doc-level record.
        table_entries: List[Dict[str, Any]] = process_result.get("table_entries", []) or []
        table_store_ok = True
        for te in table_entries:
            try:
                i = te.get("table_index")
                if i is None:
                    continue
                table_research_id = f"{process_result['doc_id']}_table_{i}"
                table_content = te.get("markdown") or ""
                if not table_content.strip():
                    continue

                table_results = {
                    "tables_count": 1,
                    "table_index": i,
                    "num_rows": te.get("num_rows"),
                    "num_cols": te.get("num_cols"),
                }

                table_metadata = {
                    **metadata,
                    "doc_id": process_result["doc_id"],
                    "table_index": i,
                    "provenance": te.get("provenance", {}),
                }

                await storage.store_research(
                    research_id=table_research_id,
                    user_id=metadata["user_id"],
                    topic=f"{metadata['title']} - Table {int(i) + 1}",
                    content=table_content,
                    results=table_results,
                    metadata=table_metadata,
                    summary=f"Extracted table {int(i) + 1} from {metadata['source']}.",
                    keywords=[metadata["file_type"], "docling_table"],
                )
            except Exception as e:
                logger.warning("Table store failed (doc=%s): %s", process_result.get("doc_id"), e)
                table_store_ok = False

        return res and table_store_ok
