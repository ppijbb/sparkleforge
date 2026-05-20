import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import cli_result_succeeded, extract_cli_result_content


def test_extract_cli_result_content_reads_final_synthesis_payload():
    result = {
        "current_step": "completed",
        "final_synthesis": {"content": "# Daily roadmap\n\nConcrete plan."},
    }

    content = extract_cli_result_content(result)

    assert content == "# Daily roadmap\n\nConcrete plan."
    assert cli_result_succeeded(result, content) is True


def test_empty_completed_payload_is_not_successful():
    result = {
        "current_step": "completed",
        "final_synthesis": {"content": "   "},
    }

    content = extract_cli_result_content(result)

    assert content == ""
    assert cli_result_succeeded(result, content) is False


def test_explicit_failure_flag_wins_over_content():
    result = {
        "success": False,
        "content": "partial output",
        "error": "verification failed",
    }

    content = extract_cli_result_content(result)

    assert content == "partial output"
    assert cli_result_succeeded(result, content) is False
