"""Tests for the scenario-eval harness's inconclusive/failed distinction and
history-based trend tracking.

Background: every judge_* check in tests/benchmark/run_scenarios.py silently
scored 0.0 whenever the LLM judge itself couldn't run (provider quota
exhausted, timeout) -- indistinguishable from the agent actually earning a
0.0. That corrupted overall_score and made the CI regression gate compare
"infra broke" against "capability regressed" as if they were the same thing.
Separately, the CI regression gate's baseline file used a schema
(`capability_scenarios.scenarios`) that the checked-in baseline JSON never
actually had (`scenarios` at top level) -- the comparison loop always hit its
"no prior baseline" branch and regression detection was silently a no-op.
"""

import asyncio
import json

import pytest

from tests.benchmark.scenario_grading import INCONCLUSIVE_MARKER, judge_score, weighted_total
from tests.benchmark.run_scenarios import (
    _compare_scenarios,
    append_history,
    compare_to_baseline,
    compare_to_history,
    print_trend,
    update_baseline,
)


class TestWeightedTotal:
    def test_no_inconclusive_checks_adjusted_equals_total(self):
        scores = {"a": (1.0, "ok"), "b": (0.5, "ok")}
        weights = {"a": 0.5, "b": 0.5}

        result = weighted_total(scores, weights)

        assert result["total"] == pytest.approx(0.75)
        assert result["adjusted_total"] == pytest.approx(0.75)
        assert result["breakdown"]["a"]["inconclusive"] is False

    def test_inconclusive_check_excluded_and_renormalized(self):
        scores = {
            "a": (1.0, "ok"),
            "b": (0.0, f"{INCONCLUSIVE_MARKER}judge unavailable: boom"),
        }
        weights = {"a": 0.5, "b": 0.5}

        result = weighted_total(scores, weights)

        # total keeps the conservative (unadjusted) semantics: 1.0*0.5 + 0.0*0.5
        assert result["total"] == pytest.approx(0.5)
        # adjusted_total renormalizes over only the conclusive check ("a")
        assert result["adjusted_total"] == pytest.approx(1.0)
        assert result["breakdown"]["b"]["inconclusive"] is True
        # the marker prefix must not leak into the displayed reason
        assert result["breakdown"]["b"]["reason"] == "judge unavailable: boom"

    def test_all_checks_inconclusive_yields_none_adjusted_total(self):
        scores = {"a": (0.0, f"{INCONCLUSIVE_MARKER}no models available")}
        weights = {"a": 1.0}

        result = weighted_total(scores, weights)

        assert result["total"] == 0.0
        assert result["adjusted_total"] is None

    def test_missing_check_is_not_inconclusive(self):
        """A check that never ran at all (KeyError-equivalent) is a real 0.0, not infra noise."""
        result = weighted_total({}, {"a": 1.0})

        assert result["breakdown"]["a"]["inconclusive"] is False
        assert result["adjusted_total"] == 0.0


class TestJudgeScore:
    @pytest.mark.asyncio
    async def test_no_rubric_is_not_inconclusive(self):
        score, reason = await judge_score(rubric="", transcript="something happened")
        assert score == 0.0
        assert not reason.startswith(INCONCLUSIVE_MARKER)

    @pytest.mark.asyncio
    async def test_empty_transcript_is_not_inconclusive(self):
        score, reason = await judge_score(rubric="grade this", transcript="   ")
        assert score == 0.0
        assert not reason.startswith(INCONCLUSIVE_MARKER)

    @pytest.mark.asyncio
    async def test_judge_exception_is_inconclusive(self, monkeypatch):
        import tests.benchmark.scenario_grading as scenario_grading

        async def _boom(rubric, transcript, context):
            raise RuntimeError("All fallback models failed. No available models.")

        monkeypatch.setattr(scenario_grading, "_call_judge", _boom)

        score, reason = await judge_score(rubric="grade this", transcript="agent output")

        assert score == 0.0
        assert reason.startswith(INCONCLUSIVE_MARKER)
        assert "All fallback models failed" in reason

    @pytest.mark.asyncio
    async def test_judge_timeout_is_inconclusive(self, monkeypatch):
        import tests.benchmark.scenario_grading as scenario_grading

        async def _hangs(rubric, transcript, context):
            raise asyncio.TimeoutError()

        monkeypatch.setattr(scenario_grading, "_call_judge", _hangs)
        monkeypatch.setattr(scenario_grading, "JUDGE_TIMEOUT_S", 0.01)

        score, reason = await judge_score(rubric="grade this", transcript="agent output")

        assert score == 0.0
        assert reason.startswith(INCONCLUSIVE_MARKER)


def _make_report(scenario_id: str, total: float, adjusted_total, breakdown=None, generated_at="2026-01-01T00:00:00Z"):
    return {
        "generated_at": generated_at,
        "overall_score": total,
        "overall_score_adjusted": adjusted_total,
        "inconclusive_checks": 0,
        "scenarios": {
            scenario_id: {
                "id": scenario_id,
                "total": total,
                "adjusted_total": adjusted_total,
                "breakdown": breakdown or {},
            }
        },
    }


class TestCompareScenarios:
    def test_first_run_scenario_is_not_a_regression(self, capsys):
        current = {"new_scenario": {"total": 0.5, "adjusted_total": 0.5, "breakdown": {}}}
        exit_code = _compare_scenarios(current, {})

        assert exit_code == 0
        assert "no prior baseline, first run" in capsys.readouterr().out

    def test_real_regression_is_detected(self):
        current = {
            "s1": {
                "total": 0.2,
                "adjusted_total": 0.2,
                "breakdown": {"check_a": {"score": 0.2, "reason": "got worse", "inconclusive": False}},
            }
        }
        baseline = {
            "s1": {
                "total": 0.9,
                "adjusted_total": 0.9,
                "breakdown": {"check_a": {"score": 0.9, "reason": "was fine"}},
            }
        }

        exit_code = _compare_scenarios(current, baseline)

        assert exit_code == 1

    def test_inconclusive_check_never_flagged_as_regression(self):
        """The exact bug this fixes: a judge outage must not look like a capability regression."""
        current = {
            "s1": {
                "total": 0.0,
                "adjusted_total": None,
                "breakdown": {
                    "judge_quality": {
                        "score": 0.0,
                        "reason": "judge unavailable: All fallback models failed.",
                        "inconclusive": True,
                    }
                },
            }
        }
        baseline = {
            "s1": {
                "total": 0.9,
                "adjusted_total": 0.9,
                "breakdown": {"judge_quality": {"score": 0.9, "reason": "was fine"}},
            }
        }

        exit_code = _compare_scenarios(current, baseline)

        assert exit_code == 0


class TestBaselineAndHistoryFiles:
    def test_update_baseline_writes_top_level_scenarios_key(self, tmp_path):
        report = _make_report("system_cleanup", 0.8, 0.8)
        baseline_path = tmp_path / "baseline.json"

        update_baseline(report, baseline_path)
        saved = json.loads(baseline_path.read_text())

        assert "system_cleanup" in saved["scenarios"]
        assert "capability_scenarios" not in saved

    def test_compare_to_baseline_reads_top_level_scenarios_key(self, tmp_path, capsys):
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(_make_report("system_cleanup", 0.9, 0.9)))

        current = _make_report(
            "system_cleanup", 0.1, 0.1,
            breakdown={"check_a": {"score": 0.1, "reason": "regressed", "inconclusive": False}},
        )
        # give the baseline a matching breakdown entry too
        baseline = json.loads(baseline_path.read_text())
        baseline["scenarios"]["system_cleanup"]["breakdown"] = {
            "check_a": {"score": 0.9, "reason": "was fine"}
        }
        baseline_path.write_text(json.dumps(baseline))

        exit_code = compare_to_baseline(current, baseline_path)

        assert exit_code == 1

    def test_compare_to_baseline_missing_file_is_not_an_error(self, tmp_path):
        exit_code = compare_to_baseline(_make_report("s1", 0.5, 0.5), tmp_path / "nope.json")
        assert exit_code == 0

    def test_append_then_compare_to_history_round_trip(self, tmp_path):
        history_path = tmp_path / "history.jsonl"
        good_run = _make_report("system_cleanup", 0.9, 0.9)
        append_history(good_run, history_path)

        bad_run = _make_report(
            "system_cleanup", 0.1, 0.1,
            breakdown={"check_a": {"score": 0.1, "reason": "regressed", "inconclusive": False}},
        )
        good_run["scenarios"]["system_cleanup"]["breakdown"] = {"check_a": {"score": 0.9, "reason": "fine"}}
        append_history(good_run, tmp_path / "history2.jsonl")  # sanity: independent file, no cross-talk

        # Re-append the good run with a breakdown so the real history file has one
        history_path.write_text("")
        append_history(good_run, history_path)

        exit_code = compare_to_history(bad_run, history_path)
        assert exit_code == 1

    def test_compare_to_history_uses_most_recent_line(self, tmp_path):
        history_path = tmp_path / "history.jsonl"
        append_history(_make_report("s1", 0.2, 0.2), history_path)
        append_history(_make_report("s1", 0.9, 0.9), history_path)  # most recent = the good one

        regressed = _make_report(
            "s1", 0.1, 0.1,
            breakdown={"c": {"score": 0.1, "reason": "bad", "inconclusive": False}},
        )
        # patch the last-written entry to have a matching breakdown
        lines = history_path.read_text().splitlines()
        last = json.loads(lines[-1])
        last["scenarios"]["s1"]["breakdown"] = {"c": {"score": 0.9, "reason": "good"}}
        history_path.write_text("\n".join(lines[:-1] + [json.dumps(last)]) + "\n")

        exit_code = compare_to_history(regressed, history_path)
        assert exit_code == 1

    def test_print_trend_missing_file_returns_error(self, tmp_path, capsys):
        exit_code = print_trend(tmp_path / "nope.jsonl")
        assert exit_code == 1

    def test_print_trend_prints_a_row_per_history_entry(self, tmp_path, capsys):
        history_path = tmp_path / "history.jsonl"
        append_history(_make_report("s1", 0.2, 0.2, generated_at="2026-01-01T00:00:00Z"), history_path)
        append_history(_make_report("s1", 0.4, 0.4, generated_at="2026-01-02T00:00:00Z"), history_path)

        exit_code = print_trend(history_path)
        out = capsys.readouterr().out

        assert exit_code == 0
        assert "2026-01-01T00:00:00Z" in out
        assert "2026-01-02T00:00:00Z" in out
        assert "+0.200" in out  # delta between the two adjusted scores
