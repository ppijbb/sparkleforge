"""Scratch Pad Manager - Filesystem Context (Agent-Skills-for-Context-Engineering)

대용량 도구 출력을 파일로 오프로드하여 컨텍스트 윈도우를 절약합니다.
반환값: 요약 + 파일 경로 참조만 컨텍스트에 넣고, 전체 내용은 파일에서 조회.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

# 프로젝트 루트 기준 scratch 디렉터리
_project_root: Path | None = None


def _get_workspace_scratch_root() -> Path:
    """workspace/scratch 디렉터리 경로. 없으면 생성."""
    global _project_root
    if _project_root is None:
        _project_root = Path(__file__).resolve().parent.parent.parent
    scratch = _project_root / "workspace" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def _estimate_tokens(text: str) -> int:
    """대략적인 토큰 수 추정 (영어 기준 단어*1.3)."""
    return int(len(text.split()) * 1.3)


def _extract_summary(data: Any, max_chars: int = 500) -> str:
    """결과에서 요약용 짧은 문자열 추출."""
    if data is None:
        return "(no output)"
    if isinstance(data, str):
        return (data[:max_chars] + "..." if len(data) > max_chars else data).strip()
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            n = len(data["results"])
            first = data["results"][0] if data["results"] else {}
            preview = json.dumps(first, ensure_ascii=False)[:200]
            return f"Found {n} results. First: {preview}..."
        if "content" in data:
            c = data["content"]
            return (str(c)[:max_chars] + "..." if len(str(c)) > max_chars else str(c)).strip()
        return (json.dumps(data, ensure_ascii=False)[:max_chars] + "...").strip()
    if isinstance(data, list):
        return f"List of {len(data)} items. First: {str(data[0])[:150]}..." if data else "(empty list)"
    return str(data)[:max_chars]


def write_tool_output(
    tool_name: str,
    output: Any,
    *,
    threshold_chars: int = 8000,
    threshold_tokens: int = 2000,
) -> Tuple[str | None, str]:
    """대용량 도구 출력을 scratch 파일에 쓰고, 참조와 요약만 반환.

    Args:
        tool_name: 도구 이름 (파일명 접두사)
        output: 원본 출력 (dict 등, JSON 직렬화 가능)
        threshold_chars: 이 문자 수 초과 시 오프로드
        threshold_tokens: 이 토큰 수 초과 시 오프로드 (chars와 둘 다 확인)

    Returns:
        (file_path, summary) 오프로드한 경우; (None, summary) 오프로드하지 않은 경우.
        summary는 항상 반환 (컨텍스트에 넣을 짧은 요약).
    """
    try:
        if isinstance(output, dict) and "data" in output:
            payload = output.get("data")
        else:
            payload = output

        if payload is None:
            return None, "(no output)"

        raw = json.dumps(payload, ensure_ascii=False, indent=2)
        summary = _extract_summary(payload)

        num_chars = len(raw)
        num_tokens = _estimate_tokens(raw)

        if num_chars <= threshold_chars and num_tokens <= threshold_tokens:
            return None, summary

        scratch_root = _get_workspace_scratch_root()
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in tool_name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{safe_name}_{ts}.json"
        file_path = scratch_root / fname

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(raw)

        logger.info(
            "ScratchPad: offloaded %s chars to %s (summary in context)",
            num_chars, file_path.name,
        )
        return str(file_path), summary
    except Exception as e:
        logger.warning("ScratchPad write failed: %s", e)
        summary = _extract_summary(output)
        return None, summary


def build_result_with_scratch_ref(
    original_result: Dict[str, Any],
    scratch_path: str | None,
    summary: str,
) -> Dict[str, Any]:
    """원본 result dict에서 data를 scratch 참조로 교체한 새 dict 반환."""
    out = dict(original_result)
    if scratch_path and out.get("success"):
        out["data"] = {
            "_scratch_ref": scratch_path,
            "summary": summary,
            "instruction": "Use read_file or grep on the path above to retrieve full content when needed.",
        }
    return out
