"""Tests for the intervention telemetry & preference learning daemon (#1068)."""

import os
import tempfile

import pytest

from src.core.intervention_telemetry import InterventionTelemetryDaemon


@pytest.fixture()
def daemon(tmp_path):
    return InterventionTelemetryDaemon(
        store_path=str(tmp_path / "telemetry.json"),
        smoothing=0.5,
    )


def test_record_updates_stats(daemon):
    daemon.record("after_planning", "approve")
    daemon.record("after_planning", "revise")
    daemon.record("after_planning", "abort")
    stats = daemon.stats("after_planning")["after_planning"]
    assert stats.total == 3
    assert stats.approvals == 1
    assert stats.revisions == 1
    assert stats.rejections == 1
    assert stats.approval_rate == pytest.approx(1 / 3)


def test_threshold_moves_down_on_rejection(daemon):
    start = daemon.recommend_threshold("after_planning")
    for _ in range(10):
        daemon.record("after_planning", "abort")
    end = daemon.recommend_threshold("after_planning")
    assert end < start


def test_threshold_moves_up_on_approval(daemon):
    start = daemon.recommend_threshold("after_planning")
    for _ in range(10):
        daemon.record("after_planning", "approve")
    end = daemon.recommend_threshold("after_planning")
    assert end > start


def test_threshold_clamped(daemon):
    for _ in range(50):
        daemon.record("after_planning", "abort")
    assert daemon.recommend_threshold("after_planning") >= daemon.min_threshold
    for _ in range(50):
        daemon.record("before_final_report", "approve")
    assert daemon.recommend_threshold("before_final_report") <= daemon.max_threshold


def test_persistence_round_trip(tmp_path):
    path = str(tmp_path / "telemetry.json")
    d1 = InterventionTelemetryDaemon(store_path=path, smoothing=0.5)
    d1.record("after_planning", "approve")
    d2 = InterventionTelemetryDaemon(store_path=path, smoothing=0.5)
    assert d2.stats("after_planning")["after_planning"].approvals == 1
