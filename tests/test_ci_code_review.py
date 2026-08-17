from src.core.ci.code_review import code_review


async def test_code_review_soft_fails_when_all_model_providers_are_unavailable(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    diff_path = tmp_path / "diff.txt"
    diff_path.write_text("diff --git a/foo.py b/foo.py\n+pass\n", encoding="utf-8")

    from src.core.llm_manager import MultiModelOrchestrator

    async def _raise_all_failed(self, *args, **kwargs):
        raise RuntimeError("All fallback models failed. No available models.")

    monkeypatch.setattr(MultiModelOrchestrator, "execute_with_model", _raise_all_failed)

    result = await code_review(diff_path)

    assert result == 0
    review_text = (tmp_path / "review_result.txt").read_text(encoding="utf-8")
    assert "unavailable" in review_text.lower()
