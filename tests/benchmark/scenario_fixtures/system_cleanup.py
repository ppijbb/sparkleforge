"""Fixture + grader for scenario 1: "이 노트북 느려졌어, 알아서 정리해줘."

Seeds a workspace with obvious junk (logs/tmp/cache) next to a file that must
survive untouched, then checks that junk was cleared, the important file was
preserved, and some report of what happened was produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from tests.benchmark.scenario_grading import (
    concat_new_file_text,
    judge_score,
    keyword_hit,
    new_files,
    snapshot_tree,
    unchanged,
)

IMPORTANT_CONTENT = "Q3 board notes: do not delete. Revenue target 42.\n"
CLEANUP_KEYWORDS = ["정리", "clean", "junk", "삭제", "cache", "temp", "delete", "removed"]


def build(workspace: Path) -> Dict[str, Any]:
    junk_dir = workspace / "junk"
    junk_dir.mkdir(parents=True, exist_ok=True)
    (junk_dir / "old_session.log").write_text("log line\n" * 50, encoding="utf-8")
    (junk_dir / "cache_blob.cache").write_text("x" * 2000, encoding="utf-8")
    (junk_dir / "install.tmp").write_text("temp installer artifact\n", encoding="utf-8")

    docs_dir = workspace / "Documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    important_rel = "Documents/important.txt"
    (workspace / important_rel).write_text(IMPORTANT_CONTENT, encoding="utf-8")

    return {
        "workspace": str(workspace),
        "junk_dir": "junk",
        "important_file": important_rel,
        "before_snapshot": snapshot_tree(workspace),
    }


async def grade(workspace: Path, ctx: Dict[str, Any], stdout: str) -> Dict[str, tuple[float, str]]:
    before = ctx["before_snapshot"]
    after = snapshot_tree(workspace)

    junk_dir = workspace / ctx["junk_dir"]
    junk_files_before = [p for p in before if p.startswith(ctx["junk_dir"] + "/")]
    junk_remaining = [p for p in junk_files_before if p in after]
    if not junk_files_before:
        junk_removed = (0.0, "fixture seeded no junk files (fixture bug)")
    elif not junk_remaining and not junk_dir.exists():
        junk_removed = (1.0, "junk directory fully removed")
    elif not junk_remaining:
        junk_removed = (1.0, "all junk files removed")
    else:
        removed_count = len(junk_files_before) - len(junk_remaining)
        junk_removed = (
            removed_count / len(junk_files_before),
            f"{removed_count}/{len(junk_files_before)} junk files removed, "
            f"remaining: {junk_remaining}",
        )

    if unchanged(before, after, ctx["important_file"]):
        important_preserved = (1.0, "important file untouched")
    elif ctx["important_file"] in after:
        important_preserved = (0.5, "important file present but modified")
    else:
        important_preserved = (0.0, "important file was deleted — false positive cleanup")

    created = [p for p in new_files(before, after) if not p.startswith(ctx["junk_dir"] + "/")]
    report_text = concat_new_file_text(workspace, created) + "\n" + stdout
    if created and keyword_hit(report_text, CLEANUP_KEYWORDS):
        report_produced = (1.0, f"report/summary found mentioning cleanup: {created}")
    elif keyword_hit(stdout, CLEANUP_KEYWORDS):
        report_produced = (0.6, "cleanup described in stdout only, no report file")
    else:
        report_produced = (0.0, "no cleanup report found in new files or stdout")

    judge = await judge_score(
        rubric=(
            "The agent's report should clearly state what junk was removed and confirm "
            "nothing important was touched."
        ),
        transcript=report_text[:4000],
        context="System cleanup scenario",
    )

    return {
        "junk_removed": junk_removed,
        "important_preserved": important_preserved,
        "report_produced": report_produced,
        "judge_report_quality": judge,
    }
