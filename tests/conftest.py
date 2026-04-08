"""Shared pytest fixtures for all test modules."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return the project root directory path."""
    return Path(__file__).parent.parent


@pytest.fixture
def runner(project_root):
    """Create a BenchmarkRunner instance for benchmark tests."""
    import sys
    sys.path.insert(0, str(project_root / "tests" / "benchmark"))
    from benchmark_runner import BenchmarkRunner
    config_path = project_root / "tests" / "benchmark" / "benchmark_config.yaml"
    thresholds_path = project_root / "tests" / "benchmark" / "benchmark_thresholds.yaml"
    return BenchmarkRunner(
        str(project_root), str(config_path), str(thresholds_path)
    )
