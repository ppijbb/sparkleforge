"""Parsing/formatting helpers for MCP tool calls and results.

Extracted from the monolithic ``src/core/mcp_integration.py`` (issue #508,
Anvil Phase Sigma-1). Pure functions with no dependency on UniversalMCPHub
or the tool-execution dispatch layer, so they're safe to isolate first.
"""

import json
import logging
import os
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _parse_json_text(value: str, *, context: str) -> tuple[bool, Any]:
    """Parse a JSON string without warning for expected plain-text tool output."""
    try:
        return True, json.loads(value)
    except json.JSONDecodeError:
        logger.debug(
            "%s returned non-JSON text; preserving plain-text fallback: %r",
            context,
            value[:120],
        )
        return False, None
    except TypeError as exc:
        logger.debug("%s returned non-string JSON candidate: %s", context, exc)
        return False, None


def _parse_markdown_link_results(value: str) -> list[dict[str, str]]:
    """Parse simple numbered markdown-link search results from plain text."""
    results: list[dict[str, str]] = []
    current_result: dict[str, str] | None = None
    for line in value.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        link_match = re.match(r"^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)", line)
        if link_match:
            if current_result:
                results.append(current_result)
            current_result = {
                "title": link_match.group(1),
                "url": link_match.group(2),
                "snippet": "",
            }
        elif current_result:
            current_result["snippet"] = f"{current_result['snippet']} {line}".strip()
    if current_result:
        results.append(current_result)
    return results


def _normalize_mcp_tool_alias(tool_name: str) -> str:
    """Normalize legacy MCP tool names to SEP-986-safe aliases."""
    return tool_name.replace("::", "_").replace("-", "_").replace("/", "_").replace(" ", "_")


# 9대 혁신: ToolTrace 추적 시스템
_tool_trace_manager = None


def get_tool_trace_manager():
    """전역 ToolTraceManager 인스턴스 반환 (싱글톤 패턴)."""
    global _tool_trace_manager
    if _tool_trace_manager is None:
        from src.core.tool_trace import ToolTraceManager

        _tool_trace_manager = ToolTraceManager()
    return _tool_trace_manager


def set_tool_trace_manager(manager):
    """전역 ToolTraceManager 인스턴스 설정."""
    global _tool_trace_manager
    _tool_trace_manager = manager


def _create_tool_trace(
    tool_id: str,
    citation_id: str,
    tool_type: str,
    query: str,
    result: Dict[str, Any],
    mcp_server: str | None = None,
    mcp_tool_name: str | None = None,
) -> Any | None:
    """ToolTrace 생성 헬퍼 함수 (9대 혁신: ToolTrace 추적 시스템).

    Args:
        tool_id: Tool ID
        citation_id: Citation ID
        tool_type: Tool type
        query: Query string
        result: 도구 실행 결과
        mcp_server: MCP 서버 이름 (optional)
        mcp_tool_name: MCP 도구 이름 (optional)

    Returns:
        ToolTrace 객체 (생성 성공 시), None (실패 시)
    """
    try:
        from src.core.tool_trace import ToolTrace

        # raw_answer 생성 (result를 JSON 문자열로)
        raw_answer = json.dumps(result, ensure_ascii=False, indent=2) if result else "{}"

        # summary 생성 (간단한 요약)
        if result.get("success"):
            if isinstance(result.get("data"), dict):
                if "results" in result["data"]:
                    summary = f"Found {len(result['data']['results'])} results"
                elif "content" in result["data"]:
                    content = str(result["data"]["content"])
                    summary = (
                        f"Content: {content[:100]}..."
                        if len(content) > 100
                        else f"Content: {content}"
                    )
                else:
                    summary = "Tool executed successfully"
            else:
                summary = "Tool executed successfully"
        else:
            summary = f"Tool execution failed: {result.get('error', 'Unknown error')[:100]}"

        trace = ToolTrace.create_with_size_limit(
            tool_id=tool_id,
            citation_id=citation_id,
            tool_type=tool_type,
            query=query,
            raw_answer=raw_answer,
            summary=summary,
            mcp_server=mcp_server,
            mcp_tool_name=mcp_tool_name,
        )

        # ToolTraceManager에 추가
        try:
            manager = get_tool_trace_manager()
            manager.add_trace(trace)
        except Exception as e:
            logger.debug(f"Failed to add ToolTrace to manager: {e}")

        return trace
    except Exception as e:
        logger.debug(f"Failed to create ToolTrace: {e}")
        return None


def _infer_tool_type(tool_name: str) -> str:
    """도구 이름에서 도구 타입 추론.

    Args:
        tool_name: 도구 이름

    Returns:
        도구 타입
    """
    tool_lower = tool_name.lower()

    if "::" in tool_name:
        # MCP 도구
        return "mcp_tool"
    elif "search" in tool_lower or "google" in tool_lower or "tavily" in tool_lower:
        return "web_search"
    elif "arxiv" in tool_lower or "scholar" in tool_lower or "paper" in tool_lower:
        return "paper_search"
    elif "rag" in tool_lower or "query" in tool_lower:
        return "rag_hybrid" if "hybrid" in tool_lower else "rag_naive"
    elif "code" in tool_lower or "python" in tool_lower or "execute" in tool_lower:
        return "run_code"
    elif "browser" in tool_lower:
        return "browser"
    elif (
        "screenshot" in tool_lower
        or "mouse" in tool_lower
        or "keyboard" in tool_lower
        or "computer_use" in tool_lower
        or "key_press" in tool_lower
        or "type_text" in tool_lower
    ):
        return "computer"
    elif "generate" in tool_lower or "document" in tool_lower:
        return "document_generation"
    elif "file" in tool_lower:
        return "file_operation"
    else:
        return "unknown"


def _format_query_string(tool_name: str, parameters: Dict[str, Any]) -> str:
    """도구 파라미터를 쿼리 문자열로 포맷.

    Args:
        tool_name: 도구 이름
        parameters: 도구 파라미터

    Returns:
        포맷된 쿼리 문자열
    """
    # 주요 파라미터 추출
    query_keys = ["query", "question", "text", "input", "url", "path", "code"]
    for key in query_keys:
        if key in parameters:
            value = parameters[key]
            if isinstance(value, str):
                return value[:200]  # 최대 200자
            elif isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False)[:200]

    # 파라미터 전체를 JSON으로
    return json.dumps(parameters, ensure_ascii=False)[:200]


def _structured_tool_description(tool_config: Dict[str, Any], tool_name: str) -> str:
    """Build a stable LangChain tool description from loose config data."""
    description = str(tool_config.get("description") or tool_name).strip()
    category = tool_config.get("category")
    parameters = tool_config.get("parameters") or {}
    if not parameters:
        return description

    param_bits = []
    if isinstance(parameters, dict):
        for key, spec in parameters.items():
            if isinstance(spec, dict):
                type_name = spec.get("type", "any")
                required = "required" if spec.get("required") else "optional"
                param_bits.append(f"{key}: {type_name} ({required})")
            else:
                param_bits.append(str(key))

    suffix = "; ".join(param_bits[:8])
    category_text = f"Category: {category}. " if category else ""
    return f"{description}\n\n{category_text}Parameters: {suffix}".strip()


def _actionable_error_message(tool_name: str, error: Any) -> str:
    """Normalize tool errors so callers do not crash while formatting them."""
    raw = str(error or "Unknown error").strip()
    if not raw:
        raw = "Unknown error"
    raw = raw.replace("\n", " ")
    if len(raw) > 500:
        raw = raw[:497] + "..."
    return f"{tool_name} failed: {raw}"


def _cap_tool_result_for_context(
    result: Dict[str, Any], tool_name: str, max_chars: int | None = None
) -> Dict[str, Any]:
    """Cap very large successful tool payloads while preserving result shape."""
    if not isinstance(result, dict):
        return {
            "success": False,
            "data": None,
            "error": f"{tool_name} returned non-dict result",
        }

    limit = max_chars or int(os.getenv("MCP_TOOL_RESULT_MAX_CHARS", "12000"))
    try:
        data = result.get("data")
        rendered = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        if rendered and len(rendered) > limit:
            capped = rendered[:limit] + "\n...[truncated]"
            return {
                **result,
                "data": {
                    "preview": capped,
                    "truncated": True,
                    "original_type": type(data).__name__,
                },
            }
    except Exception as e:
        logger.debug("Tool result capping skipped for %s: %s", tool_name, e)
    return result


def _normalize_mcp_call_params(tool_def: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap params for FastMCP tools whose schema exposes a single input model."""

    def _normalize_aliases(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize common caller aliases to embedded MCP server field names."""
        normalized = dict(raw or {})
        if "max_results" in normalized and "num_results" not in normalized:
            normalized["num_results"] = normalized.pop("max_results")
        return normalized

    if isinstance(tool_def, dict):
        schema = tool_def.get("inputSchema") or tool_def.get("input_schema")
    else:
        schema = getattr(tool_def, "inputSchema", None) or getattr(tool_def, "input_schema", None)
    params = _normalize_aliases(params)
    if not isinstance(schema, dict):
        return params

    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    if set(properties.keys()) == {"input"} and "input" in required and "input" not in params:
        return {"input": params}
    if "input" in params and isinstance(params["input"], dict):
        return {"input": _normalize_aliases(params["input"])}
    return params


# Centralized Tool Registry imports
