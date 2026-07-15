#!/usr/bin/env python3
"""Prevent hardcoded Supabase credentials in frontend assets.

Scans .html and .js files for literal ``supabaseKey``/``supabase_url`` style
assignments and fails the build if any are found. Replacement tokens such as
``__SUPABASE_URL__`` and ``__SUPABASE_ANON_KEY__`` are allowed because they are
substituted at build time from environment variables.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_TOKENS = {"__SUPABASE_URL__", "__SUPABASE_ANON_KEY__"}

KEY_PATTERN = re.compile(
    r"(?i)\b(supabase(?:_?url|_?key|_?anon_?key))\s*[:=]\s*['\"](?!__SUPABASE)"
)


def scan_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    findings: list[str] = []
    for line_num, line in enumerate(text.splitlines(), 1):
        if KEY_PATTERN.search(line):
            findings.append(f"{path.relative_to(ROOT)}:{line_num}: {line.strip()}")
    return findings


def main() -> int:
    targets = list(ROOT.glob("src/web/**/*.html")) + list(ROOT.glob("src/web/**/*.js"))
    findings: list[str] = []
    for path in targets:
        findings.extend(scan_file(path))

    if findings:
        print("Hardcoded Supabase credentials detected in frontend assets:")
        for item in findings:
            print(f"  - {item}")
        print("Inject credentials via build-time tokens like __SUPABASE_URL__ instead.")
        return 1

    print("No hardcoded Supabase credentials found in frontend assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
