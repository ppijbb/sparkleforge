#!/usr/bin/env python3
"""Minimal security reminder hook for SparkleForge. Exit 0 to allow tool; exit 2 to block."""
import json
import sys

def main() -> None:
    try:
        raw = sys.stdin.read()
        if raw:
            data = json.loads(raw)
            tool_name = data.get("tool_name", "")
            if tool_name in ("Edit", "Write", "MultiEdit", "write_file", "edit_file"):
                tool_input = data.get("tool_input") or {}
                content = tool_input.get("content") or tool_input.get("new_string") or ""
                if "eval(" in content or "exec(" in content:
                    print("Security: eval/exec in edited content is discouraged.", file=sys.stderr)
                    sys.exit(2)
    except (json.JSONDecodeError, KeyError):
        pass
    sys.exit(0)

if __name__ == "__main__":
    main()
