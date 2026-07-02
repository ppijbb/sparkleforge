import asyncio
from src.core.observe.event_bus import EventBus
from src.core.observe.observation_plane import ObservationPlane
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
