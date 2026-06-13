from pathlib import Path

from src.core.mcp_servers.registry import get_server_info
from src.core.system_control.inventory_store import InventoryStore
from src.core.system_control.resource_locator import (
    find_project_directories,
    standard_directories,
)


def test_inventory_store_uses_fresh_cache(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "inventory.json"
    store = InventoryStore(cache, ttl_seconds=3600)
    cached = {"collected_at": 9_999_999_999, "hardware": {"cpu": "cached"}}
    store.save(cached)

    monkeypatch.setattr(
        "src.core.system_control.inventory_store.collect_inventory",
        lambda: {"collected_at": 1, "hardware": {"cpu": "fresh"}},
    )

    assert store.get() == cached
    assert store.get(force_refresh=True)["hardware"]["cpu"] == "fresh"


def test_find_project_directories_by_marker(tmp_path) -> None:
    project = tmp_path / "demo"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    matches = find_project_directories([tmp_path])

    assert matches == [{"path": str(project), "markers": ["pyproject.toml"]}]


def test_standard_directories_reports_home(tmp_path) -> None:
    dirs = standard_directories(Path(tmp_path))

    assert dirs["home"] == str(tmp_path)


def test_system_control_server_registered() -> None:
    info = get_server_info("system-control")

    assert info is not None
    assert "get_system_inventory" in info["tools"]
