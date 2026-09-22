"""#1620: reports.sources entries carry agent_name/timestamp always, and
agent_log_id only when the producing execution result actually set one."""

import asyncio

import src.utils.supabase_exporter as supabase_exporter
from src.storage.hybrid_storage import HybridStorage


def test_sources_always_get_agent_name_and_timestamp(tmp_path, monkeypatch):
    captured = {}

    async def fake_publish_report(**kwargs):
        captured.update(kwargs)
        return {"id": "r1"}

    monkeypatch.setattr(supabase_exporter, "publish_report", fake_publish_report)

    storage = HybridStorage(storage_path=str(tmp_path))
    state = {
        "objective_id": "obj1",
        "user_request": "topic",
        "final_synthesis": {"content": "the report body"},
        "execution_results": [
            {"url": "https://example.com/a", "title": "A", "tool_used": "cdp"},
        ],
    }

    assert asyncio.run(storage.save_research_result(state)) is True

    sources = captured["sources"]
    assert len(sources) == 1
    assert sources[0]["agent_name"] == "cdp"
    assert "timestamp" in sources[0]
    assert "agent_log_id" not in sources[0]


def test_sources_pass_through_agent_log_id_when_present(tmp_path, monkeypatch):
    captured = {}

    async def fake_publish_report(**kwargs):
        captured.update(kwargs)
        return {"id": "r1"}

    monkeypatch.setattr(supabase_exporter, "publish_report", fake_publish_report)

    storage = HybridStorage(storage_path=str(tmp_path))
    state = {
        "objective_id": "obj2",
        "user_request": "topic",
        "final_synthesis": {"content": "the report body"},
        "execution_results": [
            {
                "url": "https://example.com/b",
                "title": "B",
                "tool_used": "hermes_agent_loop",
                "agent_log_id": "11111111-1111-1111-1111-111111111111",
            },
        ],
    }

    assert asyncio.run(storage.save_research_result(state)) is True

    sources = captured["sources"]
    assert sources[0]["agent_log_id"] == "11111111-1111-1111-1111-111111111111"


def test_sources_default_agent_name_when_tool_used_missing(tmp_path, monkeypatch):
    captured = {}

    async def fake_publish_report(**kwargs):
        captured.update(kwargs)
        return {"id": "r1"}

    monkeypatch.setattr(supabase_exporter, "publish_report", fake_publish_report)

    storage = HybridStorage(storage_path=str(tmp_path))
    state = {
        "objective_id": "obj3",
        "user_request": "topic",
        "final_synthesis": {"content": "the report body"},
        "execution_results": [{"url": "https://example.com/c"}],
    }

    assert asyncio.run(storage.save_research_result(state)) is True

    assert captured["sources"][0]["agent_name"] == "unknown"
