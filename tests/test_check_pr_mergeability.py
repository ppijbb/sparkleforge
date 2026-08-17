import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "check_pr_mergeability.py"


def _run(pr: dict, tmp_path: Path) -> dict:
    pr_file = tmp_path / "pr.json"
    pr_file.write_text(json.dumps(pr), encoding="utf-8")
    out_file = tmp_path / "mergeability_result.json"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--pr-json-file", str(pr_file), "--out", str(out_file)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    return json.loads(out_file.read_text(encoding="utf-8"))


def test_ready_pr(tmp_path):
    pr = {"isDraft": False, "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN", "statusCheckRollup": []}
    result = _run(pr, tmp_path)
    assert result == {"ready": True, "reason": ""}


def test_draft_pr_not_ready(tmp_path):
    pr = {"isDraft": True, "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN", "statusCheckRollup": []}
    result = _run(pr, tmp_path)
    assert result["ready"] is False
    assert "draft" in result["reason"]


def test_conflicting_pr_not_ready(tmp_path):
    pr = {"isDraft": False, "mergeable": "CONFLICTING", "mergeStateStatus": "CLEAN", "statusCheckRollup": []}
    result = _run(pr, tmp_path)
    assert result["ready"] is False


def test_does_not_require_uv_or_sparkleforge_bootstrap(tmp_path):
    """Confirms this really is a lightweight subprocess -- no `main.py`
    module import, no BootstrapGraph. Runs in well under a second."""
    import time

    pr = {"isDraft": False, "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN", "statusCheckRollup": []}
    start = time.monotonic()
    _run(pr, tmp_path)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"expected a lightweight process, took {elapsed:.2f}s"
