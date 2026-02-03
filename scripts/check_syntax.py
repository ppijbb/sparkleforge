#!/usr/bin/env python3
"""
Python 파일 문법 오류 점검 스크립트
"""

import sys
import ast
from pathlib import Path

project_root = Path(__file__).parent.parent
src_dir = project_root / "src"

errors = []

for py_file in src_dir.rglob("*.py"):
    if "__pycache__" in str(py_file):
        continue
    
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse
        ast.parse(content, filename=str(py_file))
    except SyntaxError as e:
        errors.append({
            'file': str(py_file.relative_to(project_root)),
            'line': e.lineno,
            'message': e.msg,
            'text': e.text
        })
    except Exception as e:
        errors.append({
            'file': str(py_file.relative_to(project_root)),
            'line': 0,
            'message': str(e),
            'text': None
        })

if errors:
    print(f"❌ {len(errors)}개 문법 오류 발견:\n")
    for err in errors:
        print(f"파일: {err['file']}")
        print(f"  라인: {err['line']}")
        print(f"  오류: {err['message']}")
        if err['text']:
            print(f"  코드: {err['text'].strip()}")
        print()
    sys.exit(1)
else:
    print("✅ 모든 Python 파일 문법 오류 없음")
    sys.exit(0)

