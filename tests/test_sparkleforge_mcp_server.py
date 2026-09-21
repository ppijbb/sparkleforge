"""Anvil Phase B-1: SparkleForge exposed as an MCP server via src/sdk.py."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.core.mcp_servers import sparkleforge_server


class _FakeJobStatus:
    def __init__(self, status, error=None):
        self.status = status
        self.error = error


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_run_task_wraps_sdk_run_success():
    fake_result = {"success": True, "content": "42"}
    with patch("src.sdk.run", new=AsyncMock(return_value=fake_result)):
        raw = await sparkleforge_server.run_task(prompt="what is the answer?")

    assert json.loads(raw) == fake_result


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_run_task_returns_structured_error_instead_of_raising():
    with patch("src.sdk.run", new=AsyncMock(side_effect=RuntimeError("boom"))):
        raw = await sparkleforge_server.run_task(prompt="anything")

    parsed = json.loads(raw)
    assert parsed["success"] is False
    assert "boom" in parsed["error"]


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_start_research_returns_job_id():
    with patch("src.sdk.submit_job", new=AsyncMock(return_value="job-123")):
        raw = await sparkleforge_server.start_research(query="what is the answer?")

    assert json.loads(raw) == {"job_id": "job-123"}


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_start_research_returns_structured_error_instead_of_raising():
    with patch("src.sdk.submit_job", new=AsyncMock(side_effect=RuntimeError("boom"))):
        raw = await sparkleforge_server.start_research(query="anything")

    parsed = json.loads(raw)
    assert parsed["success"] is False
    assert "boom" in parsed["error"]


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_get_report_running_has_no_result_yet():
    with patch(
        "src.sdk.get_job_status", new=AsyncMock(return_value=_FakeJobStatus("running"))
    ):
        raw = await sparkleforge_server.get_report(job_id="job-123")

    parsed = json.loads(raw)
    assert parsed == {"job_id": "job-123", "status": "running"}


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_get_report_completed_includes_result():
    with patch(
        "src.sdk.get_job_status", new=AsyncMock(return_value=_FakeJobStatus("completed"))
    ), patch("src.sdk.get_report", new=AsyncMock(return_value={"content": "42"})):
        raw = await sparkleforge_server.get_report(job_id="job-123")

    parsed = json.loads(raw)
    assert parsed == {"job_id": "job-123", "status": "completed", "result": {"content": "42"}}


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_get_report_failed_includes_error():
    with patch(
        "src.sdk.get_job_status",
        new=AsyncMock(return_value=_FakeJobStatus("failed", error="LLM timeout")),
    ):
        raw = await sparkleforge_server.get_report(job_id="job-123")

    parsed = json.loads(raw)
    assert parsed == {"job_id": "job-123", "status": "failed", "error": "LLM timeout"}


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_get_report_unknown_job_returns_structured_error():
    with patch(
        "src.sdk.get_job_status", new=AsyncMock(side_effect=ValueError("Job not found: x"))
    ):
        raw = await sparkleforge_server.get_report(job_id="x")

    parsed = json.loads(raw)
    assert parsed["success"] is False
    assert "Job not found" in parsed["error"]


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_nightwelding_status_lists_tracked_items(tmp_path, monkeypatch):
    from src.core.nightwelding.models import (
        NightweldingItem,
        NightweldingQueue,
        NightweldingStatus,
    )

    queue = NightweldingQueue(storage_path=tmp_path)
    queue.upsert(
        NightweldingItem(
            issue_number=42, status=NightweldingStatus.DRAFT_OPENED, pr_url="https://x"
        )
    )
    monkeypatch.setattr(
        "src.core.nightwelding.models.NightweldingQueue",
        lambda: queue,
    )

    raw = await sparkleforge_server.nightwelding_status()

    parsed = json.loads(raw)
    assert len(parsed) == 1
    assert parsed[0]["issue_number"] == 42
    assert parsed[0]["status"] == "draft_opened"


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_search_skills_returns_backend_results(monkeypatch):
    class _FakeBackend:
        def search(self, query):
            return [{"name": "foo", "description": "does foo", "score": 1.0}]

    monkeypatch.setattr(sparkleforge_server, "_skill_share_backend", lambda: _FakeBackend())

    raw = await sparkleforge_server.search_skills(query="foo")

    assert json.loads(raw) == [{"name": "foo", "description": "does foo", "score": 1.0}]


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_get_skill_returns_manifest_with_code(monkeypatch):
    from src.core.anvil.skill_marketplace import SkillManifest

    manifest = SkillManifest(name="foo", description="does foo", code="print('hi')")

    class _FakeBackend:
        def read_manifest(self, skill_id):
            assert skill_id == "foo"
            return manifest

    monkeypatch.setattr(sparkleforge_server, "_skill_share_backend", lambda: _FakeBackend())

    raw = await sparkleforge_server.get_skill(skill_id="foo")

    parsed = json.loads(raw)
    assert parsed["name"] == "foo"
    assert parsed["code"] == "print('hi')"


@pytest.mark.skipif(sparkleforge_server.mcp is None, reason="fastmcp not installed")
async def test_get_skill_returns_structured_error_when_missing(monkeypatch):
    class _FakeBackend:
        def read_manifest(self, skill_id):
            return None

    monkeypatch.setattr(sparkleforge_server, "_skill_share_backend", lambda: _FakeBackend())

    raw = await sparkleforge_server.get_skill(skill_id="no-such-skill")

    parsed = json.loads(raw)
    assert parsed["success"] is False
    assert "no-such-skill" in parsed["error"]


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_run_task_wraps_sdk_run_success())
    asyncio.run(test_run_task_returns_structured_error_instead_of_raising())
    print("ok")
