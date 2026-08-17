"""Derive a Conventional Commit type + subject from an issue title.

Moved verbatim from opencode-auto-fix.yml's two sequential, independent
matches: a leading `[tag]` bracket first, then a bare/emoji `type:` prefix
second (checked against whatever the bracket step produced, or the original
title if there was no bracket).

Known pre-existing quirk, preserved as-is (relocation, not a rewrite): the
second prefix match only recognizes fix/feat/refactor/chore (plus their
emoji/capitalized variants) -- not docs/test/perf/ci/build/style. So a
bracket tag of e.g. `[docs]` produces normalized title "docs: rest", which
the second match doesn't recognize, falling through to its default branch
that leaves the "docs: " prefix embedded in the subject unstripped. That was
already the workflow's behavior before this move.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BRACKET_TAG_TYPES = {
    "feat": "feat",
    "feature": "feat",
    "fix": "fix",
    "refactor": "refactor",
    "chore": "chore",
    "docs": "docs",
    "test": "test",
    "perf": "perf",
    "ci": "ci",
    "build": "build",
    "style": "style",
}

# (prefix, type override or None to keep whatever type is already set) --
# order matters, first match wins, exactly mirroring the original case
# statement's branch order.
_PREFIX_MAP: list[tuple[str, str | None]] = [
    ("🔧 fix:", None),
    ("✨ feature:", "feat"),
    ("♻️ refactor:", "refactor"),
    ("🧹 chore:", "chore"),
    ("fix:", None),
    ("Fix:", None),
    ("feat:", "feat"),
    ("Feat:", "feat"),
    ("feature:", "feat"),
    ("Feature:", "feat"),
    ("refactor:", "refactor"),
    ("Refactor:", "refactor"),
    ("chore:", "chore"),
    ("Chore:", "chore"),
]


@dataclass
class ConventionalCommit:
    type: str
    subject: str


def classify_conventional_commit(issue_title: str) -> ConventionalCommit:
    normalized_title = issue_title.strip()
    conventional_type = "fix"

    bracket_match = re.match(r"^\[([^\]]+)\]", normalized_title)
    if bracket_match:
        bracket_tag = bracket_match.group(1)
        rest = re.sub(r"^\[[^\]]+\]\s*", "", normalized_title)
        conventional_type = _BRACKET_TAG_TYPES.get(bracket_tag, "chore")
        normalized_title = f"{conventional_type}: {rest}"

    raw_subject = None
    for prefix, mapped_type in _PREFIX_MAP:
        if normalized_title.startswith(prefix):
            if mapped_type is not None:
                conventional_type = mapped_type
            raw_subject = normalized_title[len(prefix):]
            break
    if raw_subject is None:
        raw_subject = normalized_title

    subject = raw_subject.strip()
    subject = re.sub(r"^\d{4}-\d{2}-\d{2}\s*-\s*", "", subject)
    subject = re.sub(r"\s*\(#\d+\)$", "", subject)
    subject = re.sub(r"\s+", " ", subject)
    subject = subject.rstrip().lower()

    return ConventionalCommit(type=conventional_type, subject=subject)
