"""Response-parsing helpers shared by the CI gate agents."""

from __future__ import annotations

import json


def _strip_fenced_response(raw_content: str) -> str:
    raw_content = raw_content.strip()
    if not raw_content.startswith("```"):
        return raw_content
    lines = raw_content.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_triage_response(raw_content: str) -> dict[str, object]:
    raw_content = _strip_fenced_response(raw_content)
    candidates = [raw_content]
    start = raw_content.find("{")
    end = raw_content.rfind("}")
    if 0 <= start < end:
        candidates.append(raw_content[start : end + 1])

    saw_json_shape = raw_content.startswith("{") or start >= 0
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            raise ValueError("Triage response JSON must be an object.")
        required = {"should_create_issue", "title", "body"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(
                "Triage response JSON missing required field(s): "
                + ", ".join(missing)
            )
        if isinstance(data.get("should_create_issue"), bool):
            return data
        raise ValueError("Triage response field should_create_issue must be boolean.")

    if saw_json_shape:
        raise ValueError("Triage response looked like JSON but could not be parsed.")
    return {"should_create_issue": False, "title": "", "body": ""}
