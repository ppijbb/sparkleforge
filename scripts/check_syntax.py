#!/usr/bin/env python3
"""Python file syntax check script. Used for CI health check regression detection."""

import ast
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
src_dir = project_root / "src"

errors = []
checked = 0

for py_file in src_dir.rglob("*.py"):
    if "__pycache__" in str(py_file):
        continue

    try:
        with open(py_file, encoding="utf-8") as f:
            content = f.read()
        checked += 1

        # Try to parse
        ast.parse(content, filename=str(py_file))
    except SyntaxError as e:
        errors.append(
            {
                "file": str(py_file.relative_to(project_root)),
                "line": e.lineno,
                "message": e.msg,
                "text": e.text,
            }
        )
    except Exception as e:
        errors.append(
            {
                "file": str(py_file.relative_to(project_root)),
                "line": 0,
                "message": str(e),
                "text": None,
            }
        )

if errors:
    print(f"❌ {len(errors)} syntax errors found:\n")
    for err in errors:
        print(f"File: {err['file']}")
        print(f"  Line: {err['line']}")
        print(f"  Error: {err['message']}")
        if err["text"]:
            print(f"  Code: {err['text'].strip()}")
        print()
    sys.exit(1)
else:
    print(f"✅ No syntax errors in Python files ({checked} files checked)")
    sys.exit(0)
