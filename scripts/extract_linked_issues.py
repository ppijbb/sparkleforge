#!/usr/bin/env python3
"""Extract linked issue numbers from closing/refs keywords in PR/issue text.

Moved out of gemini-assistant.yml, where this exact grep pipeline was
copy-pasted 3x (route-gemini-review-feedback, auto-merge-ready-fix-prs,
post-pr-merge-cleanup jobs). Kept as a small stdlib-only script rather than
a `sparkleforge ci` subcommand deliberately: this is mechanical extraction
(the earlier audit's own classification), not agent judgment, and all three
call sites are otherwise-lightweight jobs that don't already pay the `uv
sync` cost -- adding the full CLI's dependency bootstrap just for this
would cost more than the dedup saves. See also scripts/validate_commit_messages.py
for the same reasoning (it backs a local git hook that needs to stay fast).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ISSUE_REF_RE = re.compile(r"(?:refs|close[sd]?|fix(?:e[sd])?|resolve[sd]?) #(?P<number>\d+)", re.IGNORECASE)


def extract_linked_issue_numbers(text: str) -> list[int]:
    numbers = {int(m.group("number")) for m in _ISSUE_REF_RE.finditer(text)}
    return sorted(numbers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, help="Read text from this file instead of stdin")
    args = parser.parse_args()

    text = args.text_file.read_text(encoding="utf-8") if args.text_file else sys.stdin.read()
    for number in extract_linked_issue_numbers(text):
        print(number)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
