import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from extract_linked_issues import extract_linked_issue_numbers  # noqa: E402

SCRIPT = PROJECT_ROOT / "scripts" / "extract_linked_issues.py"


def test_extracts_single_closes_reference():
    assert extract_linked_issue_numbers("This PR Closes #12 for good.") == [12]


def test_extracts_multiple_and_dedups():
    text = "refs #3, fixes #4 and also fixes #4 again, resolves #5"
    assert extract_linked_issue_numbers(text) == [3, 4, 5]


def test_case_insensitive_and_variants():
    text = "FIX #1, Fixed #2, Resolve #3, Resolved #4, Close #5, Closed #6"
    assert extract_linked_issue_numbers(text) == [1, 2, 3, 4, 5, 6]


def test_no_matches_returns_empty():
    assert extract_linked_issue_numbers("Just a plain PR body with #42 mentioned but no keyword.") == []


def test_cli_reads_from_text_file(tmp_path):
    text_file = tmp_path / "pr-body.txt"
    text_file.write_text("Closes #7 and refs #9.\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--text-file", str(text_file)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == ["7", "9"]


def test_cli_reads_from_stdin():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input="Fixes #11.\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == ["11"]
