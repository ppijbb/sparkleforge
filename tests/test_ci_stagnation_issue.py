from src.core.ci.stagnation_issue import (
    EXEMPT_LABELS,
    STAGNATION_LABEL,
    build_stagnation_issue,
    count_inconclusive,
    load_history,
    lowest_breakdown,
)


def test_no_stagnation_returns_none():
    report = {"stagnation_detected": False}
    assert build_stagnation_issue(report, []) is None


def test_missing_stagnation_key_returns_none():
    assert build_stagnation_issue({}, []) is None


def test_stagnation_builds_issue_with_labels():
    report = {
        "stagnation_detected": True,
        "scenarios": {
            "s1": {
                "breakdown": {
                    "correctness": {"score": 0.4, "inconclusive": False},
                    "style": {"score": 0.9, "inconclusive": False},
                }
            }
        },
    }
    issue = build_stagnation_issue(report, [])
    assert issue is not None
    assert issue.labels == [STAGNATION_LABEL, *EXEMPT_LABELS]
    assert "s1.correctness (score=0.400)" in issue.body
    assert "(no history)" in issue.body


def test_lowest_breakdown_ignores_loop_engineering_metrics_and_inconclusive():
    report = {
        "scenarios": {
            "s1": {
                "breakdown": {
                    "loop_engineering_metrics": {"score": 0.0, "inconclusive": False},
                    "correctness": {"score": 0.6, "inconclusive": False},
                    "flaky_axis": {"score": 0.1, "inconclusive": True},
                }
            }
        }
    }
    assert lowest_breakdown(report) == ("s1", "correctness", 0.6)


def test_count_inconclusive_across_scenarios():
    report = {
        "scenarios": {
            "s1": {"breakdown": {"a": {"inconclusive": True}, "b": {"inconclusive": False}}},
            "s2": {"breakdown": {"c": {"inconclusive": True}}},
        }
    }
    inconclusive, total = count_inconclusive(report)
    assert inconclusive == 2
    assert total == 3


def test_dead_judgment_axis_warning_appears_when_majority_inconclusive():
    report = {
        "stagnation_detected": True,
        "scenarios": {
            "s1": {
                "breakdown": {
                    "a": {"score": 0.5, "inconclusive": True},
                    "b": {"score": 0.5, "inconclusive": True},
                    "c": {"score": 0.5, "inconclusive": False},
                }
            }
        },
    }
    issue = build_stagnation_issue(report, [])
    assert "Dead judgment axis warning" in issue.body


def test_load_history_returns_last_n_entries(tmp_path):
    history_path = tmp_path / "history.jsonl"
    lines = [f'{{"generated_at": "2026-01-0{i}", "overall_score_adjusted": 0.{i}}}' for i in range(1, 8)]
    history_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    entries = load_history(history_path, n=5)
    assert len(entries) == 5
    assert entries[-1]["generated_at"] == "2026-01-07"


def test_load_history_missing_file_returns_empty():
    from pathlib import Path

    assert load_history(Path("/nonexistent/history.jsonl")) == []
