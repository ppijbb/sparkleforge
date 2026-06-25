#!/usr/bin/env python3
"""Validate SparkleForge commit subjects."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ALLOWED_TYPES = "feat|fix|chore|docs|refactor|test|ci|build|perf|style|revert"
HEADER_RE = re.compile(
    rf"^(?P<type>{ALLOWED_TYPES})(?:\([a-z0-9._-]+\))?!?: (?P<subject>.+)$"
)
TITLE_PREFIX_RE = re.compile(
    rf"^\s*(?:[^\w\s]+(?:\s+)*)?"
    rf"(?P<type>{ALLOWED_TYPES}|feature)(?:\([a-z0-9._-]+\))?!?:\s*(?P<subject>.*)$",
    re.IGNORECASE,
)

EMOJI_RE = re.compile(r"[\u2600-\u27BF\U0001F300-\U0001FAFF]")
ISO_DATE_RE = re.compile(r"\b20[0-9]{2}-[0-9]{2}-[0-9]{2}\b")
ISSUE_REF_RE = re.compile(r"#\d+")
UPPERCASE_RE = re.compile(r"[A-Z]")

BANNED_EXACT = {
    "fix: apply opencode changes",
    "chore: apply opencode changes",
    "fix: restore daily roadmap cli",
    "fix: restore daily roadmap generation",
    "fix: update automated fix",
    "fix: update automated issue fix",
    "chore: update automated issue fix",
    "fix: daily roadmap generation failure",
    "feat: daily roadmap generation failure",
}

BANNED_PATTERNS = [
    re.compile(r"^(fix|chore): address issue\b"),
    re.compile(r"^(fix|chore): apply .*\bchanges$"),
    re.compile(r"^(fix|chore): update automated\b"),
    re.compile(r"^(fix|chore): restore daily roadmap\b"),
    re.compile(r"^(fix|feat): daily roadmap generation failure$"),
]


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def commit_subjects(revision_range: str) -> list[tuple[str, str]]:
    output = run_git(["log", "--format=%H%x09%s", revision_range])
    commits: list[tuple[str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        commit_hash, subject = line.split("\t", 1)
        commits.append((commit_hash, subject))
    return commits


def subject_from_commit_msg_file(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def normalize_title(title: str) -> str:
    value = re.sub(r"\s+", " ", title.strip())
    commit_type = "fix"
    match = TITLE_PREFIX_RE.match(value)
    if match:
        raw_type = match.group("type").lower()
        commit_type = "feat" if raw_type == "feature" else raw_type
        subject = match.group("subject")
    else:
        subject = value

    subject = re.sub(r"^\s*20[0-9]{2}-[0-9]{2}-[0-9]{2}\s*-\s*", "", subject)
    subject = re.sub(r"\s*\(#\d+\)\s*$", "", subject)
    subject = re.sub(r"\s+", " ", subject).strip().lower()
    normalized = f"{commit_type}: {subject}" if subject else ""
    errors = validate_subject(normalized, "normalized title")
    if errors:
        raise ValueError("\n".join(errors))
    return normalized


def validate_subject(subject: str, label: str) -> list[str]:
    errors: list[str] = []
    lower_subject = subject.lower()

    if not subject:
        errors.append(f"{label}: empty commit subject is not allowed")
        return errors

    if subject.startswith("Merge "):
        errors.append(f"{label}: merge commit subjects are not allowed")
    if EMOJI_RE.search(subject):
        errors.append(f"{label}: emoji prefixes are not allowed")
    if ISO_DATE_RE.search(subject):
        errors.append(f"{label}: ISO dates are not allowed in commit subjects")
    if ISSUE_REF_RE.search(subject):
        errors.append(f"{label}: issue or PR numbers are not allowed in commit subjects")
    if "(hotfix)" in lower_subject:
        errors.append(f"{label}: hotfix labels are not allowed in commit subjects")
    if lower_subject in BANNED_EXACT or any(
        pattern.search(lower_subject) for pattern in BANNED_PATTERNS
    ):
        errors.append(f"{label}: generic automated commit subject is not allowed")

    match = HEADER_RE.match(subject)
    if not match:
        errors.append(
            f"{label}: subject must match '<type>: <lowercase specific summary>' "
            f"using one of: {ALLOWED_TYPES.replace('|', ', ')}"
        )
        return errors

    summary = match.group("subject")
    if UPPERCASE_RE.search(summary):
        errors.append(f"{label}: summary must be lowercase")
    if not re.match(r"[a-z0-9`_(-]", summary):
        errors.append(f"{label}: summary must start with a lowercase word or concrete token")
    if summary.endswith("."):
        errors.append(f"{label}: summary must not end with a period")
    if len(summary) < 12:
        errors.append(f"{label}: summary is too short to be specific")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", dest="revision_range")
    parser.add_argument("--message")
    parser.add_argument("--commit-msg-file", type=Path)
    parser.add_argument("--normalize-title")
    args = parser.parse_args()

    modes = [
        args.revision_range is not None,
        args.message is not None,
        args.commit_msg_file is not None,
        args.normalize_title is not None,
    ]
    if sum(modes) != 1:
        parser.error("choose exactly one of --range, --message, --commit-msg-file, or --normalize-title")

    if args.normalize_title is not None:
        try:
            print(normalize_title(args.normalize_title))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0

    if args.message is not None:
        items = [("message", args.message.strip())]
    elif args.commit_msg_file is not None:
        items = [(str(args.commit_msg_file), subject_from_commit_msg_file(args.commit_msg_file))]
    else:
        items = commit_subjects(args.revision_range)

    all_errors: list[str] = []
    for label, subject in items:
        all_errors.extend(validate_subject(subject, label))

    if all_errors:
        print("Commit message policy failed:", file=sys.stderr)
        for error in all_errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Commit message policy passed for {len(items)} subject(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
