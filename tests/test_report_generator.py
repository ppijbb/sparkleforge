import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.core.monitoring.report_generator import generate_daily_report, get_recent_changes
from src.core.nightwelding.models import NightweldingItem, NightweldingStatus, MakerMark


@pytest.mark.asyncio
@patch("src.core.monitoring.report_generator.NightweldingQueue")
@patch("src.core.monitoring.report_generator.MakerMarkLedger")
@patch("src.core.monitoring.report_generator.execute_llm_task")
@patch("src.core.monitoring.report_generator.get_recent_changes")
async def test_generate_daily_report(
    mock_get_recent_changes,
    mock_execute_llm_task,
    mock_ledger_cls,
    mock_queue_cls,
    tmp_path
):
    # Setup mocks
    mock_get_recent_changes.return_value = "Mocked git changes"
    
    # Mock LLM result
    mock_llm_result = MagicMock()
    mock_llm_result.content = "Mocked critique"
    mock_execute_llm_task.return_value = mock_llm_result
    
    # Mock Nightwelding Queue items
    mock_queue = MagicMock()
    mock_queue.list.return_value = [
        NightweldingItem(issue_number=1, status=NightweldingStatus.DRAFT_OPENED),
        NightweldingItem(issue_number=2, status=NightweldingStatus.FAILED),
    ]
    mock_queue_cls.return_value = mock_queue
    
    # Mock Maker Mark Ledger
    mock_ledger = MagicMock()
    mock_ledger.list.return_value = [
        MakerMark(mark_id="mark-1", issue_number=1, first_attempt_success=True, human_intervention_required=False),
        MakerMark(mark_id="mark-2", issue_number=3, first_attempt_success=True, human_intervention_required=True),
    ]
    mock_ledger_cls.return_value = mock_ledger
    
    # Run report generator
    res = await generate_daily_report(tmp_path)
    
    # Assertions
    # Strict score: 100 - (1 failed_run * 20.0) - (1 human_intervention * 10.0) = 70.0
    assert res["strict_score"] == 70.0
    assert res["metrics"]["total_attempts"] == 2
    assert res["metrics"]["success_rate"] == 50.0
    assert res["metrics"]["total_marks"] == 2
    assert res["critique"] == "Mocked critique"
    
    # Check that output files were created
    reports_dir = tmp_path / "results" / "agent_reports"
    assert (reports_dir / "latest_critique.json").exists()
    assert (reports_dir / "history.json").exists()
    
    # Verify markdown file exists
    markdown_files = list(reports_dir.glob("report_*.md"))
    assert len(markdown_files) == 1
    content = markdown_files[0].read_text(encoding="utf-8")
    assert "종합 Strict Score" in content
    assert "70.0 / 100" in content
    assert "Mocked critique" in content
