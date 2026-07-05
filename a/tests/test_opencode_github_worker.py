"""Regression tests for opencode_github_worker config bootstrapping."""

from unittest import mock

import pytest

from scripts import opencode_github_worker as worker


@pytest.mark.parametrize("entrypoint", ["code_review", "issue_triage", "merge_decision"])
def test_entrypoint_calls_ensure_config_loaded(entrypoint, tmp_path):
    """Each orchestrator entrypoint must bootstrap config before instantiation."""
    review_file = tmp_path / "review_result.txt"
    review_file.write_text("sample review", encoding="utf-8")
    pr_meta_file = tmp_path / "pr_meta.json"
    pr_meta_file.write_text("{}", encoding="utf-8")
    diff_file = tmp_path / "diff.txt"
    diff_file.write_text("sample diff", encoding="utf-8")

    with mock.patch.object(worker, "ensure_config_loaded") as mock_ensure, mock.patch(
        "src.core.llm_manager.MultiModelOrchestrator"
    ) as mock_orchestrator:
        instance = mock_orchestrator.return_value
        instance.execute_with_model = mock.AsyncMock(
            return_value=mock.Mock(content='{"should_merge": false, "reason": "test"}')
        )

        import asyncio

        if entrypoint == "code_review":
            asyncio.run(worker.code_review(diff_file))
        elif entrypoint == "issue_triage":
            asyncio.run(worker.issue_triage(review_file))
        else:
            asyncio.run(worker.merge_decision(pr_meta_file, review_file))

    mock_ensure.assert_called_once()
