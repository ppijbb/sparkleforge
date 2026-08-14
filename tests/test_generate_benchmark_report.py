"""Tests for scripts/generate_benchmark_report.py (Anvil Phase Mu-4/Mu-5, issues #1220/#1221).

Covers the fabricated "Research Pass Rate 100.0%" figure this replaces, the
CI drift check, and the Mu-5 momentum-divergence check (internal score
improving while external resolve rate or cost does not).
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_benchmark_report.py"
_spec = importlib.util.spec_from_file_location("generate_benchmark_report", SCRIPT_PATH)
gbr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gbr)


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


class TestLoadLatestScenarioEntry:
    def test_missing_file_returns_none(self, tmp_path):
        assert gbr.load_latest_scenario_entry(tmp_path / "nope.jsonl") is None

    def test_empty_file_returns_none(self, tmp_path):
        path = tmp_path / "history.jsonl"
        path.write_text("", encoding="utf-8")
        assert gbr.load_latest_scenario_entry(path) is None

    def test_prefers_most_recent_conclusive_entry(self, tmp_path):
        path = tmp_path / "history.jsonl"
        _write_jsonl(
            path,
            [
                {"generated_at": "1", "inconclusive_checks": 0, "overall_score_adjusted": 0.1},
                {"generated_at": "2", "inconclusive_checks": 3, "overall_score_adjusted": 0.9},
                {"generated_at": "3", "inconclusive_checks": 0, "overall_score_adjusted": 0.24},
            ],
        )
        entry = gbr.load_latest_scenario_entry(path)
        assert entry["generated_at"] == "3"

    def test_falls_back_to_latest_entry_when_all_inconclusive(self, tmp_path):
        path = tmp_path / "history.jsonl"
        _write_jsonl(
            path,
            [
                {"generated_at": "1", "inconclusive_checks": 5},
                {"generated_at": "2", "inconclusive_checks": 5},
            ],
        )
        entry = gbr.load_latest_scenario_entry(path)
        assert entry["generated_at"] == "2"


class TestLoadSwebenchRuns:
    def test_missing_file_returns_empty(self, tmp_path):
        assert gbr.load_swebench_runs(tmp_path / "nope.md") == []

    def test_parses_runs_in_order(self, tmp_path):
        path = tmp_path / "SWEBENCH_REPORT.md"
        path.write_text(
            "# SWE-bench Lite Report\n\n"
            "## 2026-07-26 — run `ci-20260726-abc`\n\n"
            "- Resolved: **0 / 6** submitted (6 instances)\n\n"
            "## 2026-08-02 — run `ci-20260802-def`\n\n"
            "- Resolved: **2 / 6** submitted (6 instances)\n",
            encoding="utf-8",
        )
        runs = gbr.load_swebench_runs(path)
        assert [r["run_id"] for r in runs] == ["ci-20260726-abc", "ci-20260802-def"]
        assert runs[-1]["resolved"] == "2"
        assert runs[-1]["submitted"] == "6"


class TestComputeMomentum:
    def _entry(self, generated_at: str, adjusted: float, duration: float) -> dict:
        return {
            "generated_at": generated_at,
            "inconclusive_checks": 0,
            "overall_score_adjusted": adjusted,
            "scenarios": {"s1": {"duration_s": duration}},
        }

    def test_insufficient_data_yields_no_trends_and_no_divergence(self):
        momentum = gbr.compute_momentum([], [])
        assert all(axis["trend"] is None for axis in momentum["axes"].values())
        assert momentum["divergence"] is False

    def test_all_axes_improving_together_is_not_divergence(self):
        entries = [self._entry("1", 0.2, 100.0), self._entry("2", 0.3, 80.0)]
        runs = [
            {"date": "1", "run_id": "a", "resolved": "0", "submitted": "6"},
            {"date": "2", "run_id": "b", "resolved": "2", "submitted": "6"},
        ]
        momentum = gbr.compute_momentum(entries, runs)
        assert momentum["axes"]["internal"]["trend"] == "improving"
        assert momentum["axes"]["external"]["trend"] == "improving"
        assert momentum["axes"]["cost"]["trend"] == "improving"
        assert momentum["divergence"] is False

    def test_internal_improving_external_flat_is_divergence(self):
        entries = [self._entry("1", 0.2, 100.0), self._entry("2", 0.3, 90.0)]
        runs = [
            {"date": "1", "run_id": "a", "resolved": "2", "submitted": "6"},
            {"date": "2", "run_id": "b", "resolved": "2", "submitted": "6"},
        ]
        momentum = gbr.compute_momentum(entries, runs)
        assert momentum["divergence"] is True
        assert "external" in momentum["opposing_axes"]

    def test_internal_improving_cost_worse_is_divergence(self):
        entries = [self._entry("1", 0.2, 100.0), self._entry("2", 0.3, 150.0)]
        momentum = gbr.compute_momentum(entries, [])
        assert momentum["axes"]["cost"]["trend"] == "declining"
        assert momentum["divergence"] is True
        assert "cost" in momentum["opposing_axes"]

    def test_internal_declining_is_never_divergence_regardless_of_other_axes(self):
        entries = [self._entry("1", 0.3, 100.0), self._entry("2", 0.2, 150.0)]
        runs = [
            {"date": "1", "run_id": "a", "resolved": "3", "submitted": "6"},
            {"date": "2", "run_id": "b", "resolved": "0", "submitted": "6"},
        ]
        momentum = gbr.compute_momentum(entries, runs)
        assert momentum["axes"]["internal"]["trend"] == "declining"
        assert momentum["divergence"] is False


class TestRenderMomentum:
    def test_divergence_renders_warning(self):
        momentum = {
            "axes": {
                "internal": {"trend": "improving", "delta": 0.1},
                "external": {"trend": "flat", "delta": 0.0},
                "cost": {"trend": "improving", "delta": -5.0},
            },
            "divergence": True,
            "opposing_axes": ["external"],
        }
        block = gbr.render_block(None, [], momentum)
        assert "DIVERGENCE DETECTED" in block

    def test_no_divergence_renders_clean(self):
        momentum = {
            "axes": {
                "internal": {"trend": "flat", "delta": 0.0},
                "external": {"trend": None, "delta": None},
                "cost": {"trend": None, "delta": None},
            },
            "divergence": False,
            "opposing_axes": [],
        }
        block = gbr.render_block(None, [], momentum)
        assert "DIVERGENCE DETECTED" not in block
        assert "No divergence detected" in block


class TestRenderAndSplice:
    def test_render_block_includes_percentage_and_swebench(self):
        entry = {
            "generated_at": "2026-08-09T00:00:00Z",
            "overall_score_adjusted": 0.24,
            "inconclusive_checks": 0,
            "scenarios": {"system_cleanup": {"total": 0.35, "adjusted_total": 0.35}},
        }
        runs = [{"date": "2026-08-02", "run_id": "ci-abc", "resolved": "0", "submitted": "6"}]

        block = gbr.render_block(entry, runs)

        assert "24.0%" in block
        assert "system_cleanup" in block
        assert "0 / 6" in block
        assert block.startswith(gbr.BEGIN_MARKER)
        assert block.endswith(gbr.END_MARKER)

    def test_render_block_handles_no_data(self):
        block = gbr.render_block(None, [])
        assert "no recorded runs yet" in block

    def test_splice_replaces_only_marked_region(self):
        existing = f"before\n{gbr.BEGIN_MARKER}\nstale content\n{gbr.END_MARKER}\nafter"
        result = gbr.splice(existing, f"{gbr.BEGIN_MARKER}\nfresh\n{gbr.END_MARKER}")
        assert result == "before\n" + gbr.BEGIN_MARKER + "\nfresh\n" + gbr.END_MARKER + "\nafter"

    def test_splice_raises_without_markers(self):
        with pytest.raises(SystemExit):
            gbr.splice("no markers here", "block")

    def test_regenerating_twice_is_idempotent(self, tmp_path, monkeypatch):
        report = tmp_path / "BENCHMARK_REPORT.md"
        history = tmp_path / "scenario_history.jsonl"
        swebench = tmp_path / "SWEBENCH_REPORT.md"
        report.write_text(f"# doc\n{gbr.BEGIN_MARKER}\n{gbr.END_MARKER}\n", encoding="utf-8")
        _write_jsonl(history, [{"generated_at": "1", "inconclusive_checks": 0, "overall_score_adjusted": 0.24, "scenarios": {}}])
        swebench.write_text("", encoding="utf-8")

        monkeypatch.setattr(gbr, "REPORT_PATH", report)
        monkeypatch.setattr(gbr, "HISTORY_PATH", history)
        monkeypatch.setattr(gbr, "SWEBENCH_PATH", swebench)
        monkeypatch.setattr("sys.argv", ["generate_benchmark_report.py"])

        assert gbr.main() == 0
        first = report.read_text(encoding="utf-8")
        assert gbr.main() == 0
        assert report.read_text(encoding="utf-8") == first


class TestCheckMode:
    def test_check_fails_on_stale_content(self, tmp_path, monkeypatch):
        report = tmp_path / "BENCHMARK_REPORT.md"
        history = tmp_path / "scenario_history.jsonl"
        swebench = tmp_path / "SWEBENCH_REPORT.md"
        report.write_text(f"# doc\n{gbr.BEGIN_MARKER}\nold hand-edited number: 0.775\n{gbr.END_MARKER}\n", encoding="utf-8")
        _write_jsonl(history, [{"generated_at": "1", "inconclusive_checks": 0, "overall_score_adjusted": 0.24, "scenarios": {}}])
        swebench.write_text("", encoding="utf-8")

        monkeypatch.setattr(gbr, "REPORT_PATH", report)
        monkeypatch.setattr(gbr, "HISTORY_PATH", history)
        monkeypatch.setattr(gbr, "SWEBENCH_PATH", swebench)
        monkeypatch.setattr("sys.argv", ["generate_benchmark_report.py", "--check"])

        assert gbr.main() == 1
        assert "old hand-edited number" in report.read_text(encoding="utf-8")  # --check must not write

    def test_check_passes_on_fresh_content(self, tmp_path, monkeypatch):
        report = tmp_path / "BENCHMARK_REPORT.md"
        history = tmp_path / "scenario_history.jsonl"
        swebench = tmp_path / "SWEBENCH_REPORT.md"
        report.write_text(f"# doc\n{gbr.BEGIN_MARKER}\n{gbr.END_MARKER}\n", encoding="utf-8")
        _write_jsonl(history, [{"generated_at": "1", "inconclusive_checks": 0, "overall_score_adjusted": 0.24, "scenarios": {}}])
        swebench.write_text("", encoding="utf-8")

        monkeypatch.setattr(gbr, "REPORT_PATH", report)
        monkeypatch.setattr(gbr, "HISTORY_PATH", history)
        monkeypatch.setattr(gbr, "SWEBENCH_PATH", swebench)
        monkeypatch.setattr("sys.argv", ["generate_benchmark_report.py"])
        assert gbr.main() == 0

        monkeypatch.setattr("sys.argv", ["generate_benchmark_report.py", "--check"])
        assert gbr.main() == 0
