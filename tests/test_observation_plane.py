import asyncio
from src.core.observe.event_bus import EventBus
from src.core.observe.observation_plane import ObservationPlane
from src.core.observe.system_collector import SystemCollector
from src.core.bootstrap_graph import BootstrapGraph


def test_event_bus_pub_sub():
    async def run_test():
        bus = EventBus()
        received_data = []

        async def callback(data):
            received_data.append(data)

        sub_id = bus.subscribe("test_event", callback)
        assert sub_id.startswith("sub_")

        await bus.publish("test_event", {"payload": "data"})
        assert received_data == [{"payload": "data"}]

        assert bus.unsubscribe(sub_id) is True
        await bus.publish("test_event", {"payload": "data2"})
        assert len(received_data) == 1

    asyncio.run(run_test())


def test_system_collector_metrics():
    async def run_test():
        op = ObservationPlane()
        metrics = await op.system.get_all_metrics()

        assert "cpu" in metrics
        assert "memory" in metrics
        assert "disk" in metrics
        assert "network" in metrics
        assert "battery" in metrics
        assert "temperature" in metrics

    asyncio.run(run_test())


def test_check_thresholds_flags_exceeded_metrics():
    collector = SystemCollector(thresholds={"cpu_percent": 50.0, "memory_percent": 50.0, "disk_percent": 50.0})
    warnings = collector.check_thresholds(
        {
            "cpu": {"percent": 95.0},
            "memory": {"percent": 30.0},
            "disk": {"percent": 60.0},
        }
    )
    assert len(warnings) == 2
    assert any("CPU" in w for w in warnings)
    assert any("Disk" in w for w in warnings)
    assert not any("Memory" in w for w in warnings)


def test_check_thresholds_ignores_errored_metrics():
    collector = SystemCollector()
    warnings = collector.check_thresholds(
        {
            "cpu": {"error": "boom"},
            "memory": {"error": "boom"},
            "disk": {"error": "boom"},
        }
    )
    assert warnings == []


def test_integrated_state_includes_resource_warnings():
    async def run_test():
        op = ObservationPlane()
        state = await op.get_integrated_state()
        assert "resource_warnings" in state
        assert isinstance(state["resource_warnings"], list)

    asyncio.run(run_test())


def test_snapshot_api_retrieval():
    async def run_test():
        op = ObservationPlane()
        snapshot = await op.snapshot.get_system_snapshot()

        assert "processes" in snapshot
        assert "ports" in snapshot
        assert "sessions" in snapshot
        assert "services" in snapshot

    asyncio.run(run_test())


def test_package_inventory_command():
    async def run_test():
        op = ObservationPlane()
        inventory = await op.packages.get_unified_inventory()

        assert "apt" in inventory
        assert "brew" in inventory
        assert "pipx" in inventory
        assert "npm" in inventory

    asyncio.run(run_test())


def test_window_tracker():
    async def run_test():
        op = ObservationPlane()
        window = await op.windows.get_active_window()
        assert window is not None
        assert "title" in window

        window_list = await op.windows.get_window_list()
        assert isinstance(window_list, list)

    asyncio.run(run_test())


def test_bootstrap_graph_integration():
    async def run_test():
        graph = BootstrapGraph(runtime_mode="local")
        res = await graph.run()

        assert res.ok is True
        assert "observation_plane" in res.values
        obs_payload = res.values["observation_plane"]
        assert "observation_plane" in obs_payload
        assert "metrics_available" in obs_payload

    asyncio.run(run_test())
