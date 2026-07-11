"""Shared pytest fixtures for all test modules."""

import os
import shutil
import tempfile
import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return the project root directory path."""
    return Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def isolate_storage_env(monkeypatch):
    """Isolate storage env by redirecting MEMORI_STORAGE_PATH to a temp directory."""
    temp_dir = tempfile.mkdtemp()
    monkeypatch.setenv("MEMORI_STORAGE_PATH", os.path.join(temp_dir, "memori"))
    # Also reset the global shared memory instance before and after test
    import src.core.shared_memory
    src.core.shared_memory._shared_memory = None
    yield temp_dir
    src.core.shared_memory._shared_memory = None
    shutil.rmtree(temp_dir, ignore_errors=True)


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
