import asyncio
from unittest.mock import AsyncMock
import pytest
from src.core.bootstrap_graph import BootstrapGraph
from src.core.automation.automation_engine import AutomationEngine
from src.core.scheduler import get_scheduler, ScheduleStatus
from src.core.observe.event_bus import EventBus


@pytest.fixture(autouse=True)
def clean_scheduler():
    # Reset scheduler state between tests
    scheduler = get_scheduler()
    scheduler.schedules.clear()
    scheduler.executions.clear()
    scheduler.running_tasks.clear()
    # Detach callback
    scheduler.set_execution_callback(None)
    # Detach any coordinator/timeout override left on the singleton by a previous test
    if AutomationEngine._instance is not None:
        AutomationEngine._instance.coordinator = None
        AutomationEngine._instance.__dict__.pop("DEFAULT_ROUTED_TIMEOUT", None)
    yield
    scheduler.schedules.clear()
    scheduler.executions.clear()
    if AutomationEngine._instance is not None:
        AutomationEngine._instance.coordinator = None
        AutomationEngine._instance.__dict__.pop("DEFAULT_ROUTED_TIMEOUT", None)


def make_mock_coordinator(delegate_result: bool = True) -> AsyncMock:
    coordinator = AsyncMock()
    coordinator.active_workers = {"remote-worker": object()}
    coordinator.delegate_task.return_value = delegate_result
    return coordinator


@pytest.mark.asyncio
async def test_automation_engine_creation_and_delegation():
    engine = AutomationEngine()
    assert engine is not None
    assert engine.scheduler == get_scheduler()

    # Create dummy cron automation
    auto = engine.create_automation(
        name="cron_test",
        user_query="test query",
        trigger_type="cron",
        cron_expression="0 9 * * *"
    )
    assert auto.name == "cron_test"
    assert auto.cron_expression == "0 9 * * *"
    assert auto.metadata["trigger_type"] == "cron"
    assert auto.enabled is True

    # Create dummy event automation
    auto_event = engine.create_automation(
        name="event_test",
        user_query="event query",
        trigger_type="event",
        event_type="metrics_alert"
    )
    assert auto_event.cron_expression == "0 0 1 1 *" # placeholder
    assert auto_event.metadata["trigger_type"] == "event"
    assert auto_event.metadata["event_type"] == "metrics_alert"
    assert auto_event.enabled is False
    assert auto_event.status == ScheduleStatus.DISABLED


@pytest.mark.asyncio
async def test_event_trigger():
    event_bus = EventBus()
    engine = AutomationEngine(event_bus=event_bus)
    
    triggered_query = None
    async def mock_callback(query, session_id):
        nonlocal triggered_query
        triggered_query = query
        return {"status": "ok"}
        
    engine._orig_callback = mock_callback

    # Register event automation
    engine.create_automation(
        name="event_alert",
        user_query="process alert",
        trigger_type="event",
        event_type="cpu_spike"
    )

    # Publish event
    await event_bus.publish("cpu_spike", {"cpu_percent": 99})
    await asyncio.sleep(0.5) # Allow event-bus execution

    assert triggered_query == "process alert"


@pytest.mark.asyncio
async def test_webhook_trigger():
    engine = AutomationEngine()

    triggered_query = None
    async def mock_callback(query, session_id):
        nonlocal triggered_query
        triggered_query = query
        return {"status": "ok"}

    engine._orig_callback = mock_callback

    # Register webhook automation
    auto = engine.create_automation(
        name="webhook_auto",
        user_query="process webhook",
        trigger_type="webhook",
        webhook_id="github_pr_opened"
    )

    # Trigger webhook
    execs = await engine.trigger_webhook("github_pr_opened", {"pr_number": 12})
    assert len(execs) == 1
    assert execs[0].schedule_id == auto.schedule_id

    # Wait for execution loop
    await asyncio.sleep(0.2)
    assert triggered_query == "process webhook"


@pytest.mark.asyncio
async def test_chain_trigger():
    engine = AutomationEngine()

    triggered_queries = []
    async def mock_callback(query, session_id):
        triggered_queries.append(query)
        return {"status": "ok"}

    engine._orig_callback = mock_callback

    # Register parent cron/manual automation
    parent = engine.create_automation(
        name="parent_task",
        user_query="run parent",
        trigger_type="cron",
        cron_expression="0 9 * * *"
    )

    # Register child chain automation
    child = engine.create_automation(
        name="child_task",
        user_query="run child",
        trigger_type="chain",
        parent_id=parent.schedule_id
    )

    # Run parent manually
    await get_scheduler().run_now(parent.schedule_id)
    await asyncio.sleep(0.5) # Wait for parent and child execution

    # Verify both ran in sequence
    assert "run parent" in triggered_queries
    assert "run child" in triggered_queries
    # Child should run after parent
    assert triggered_queries[0] == "run parent"
    assert triggered_queries[1] == "run child"


@pytest.mark.asyncio
async def test_multi_agent_routing():
    engine = AutomationEngine()

    routed_queries = []
    async def mock_callback(query, session_id):
        routed_queries.append(query)
        return {"status": "ok"}

    engine._orig_callback = mock_callback

    # Register automation with agent expertise tag
    auto = engine.create_automation(
        name="agent_gui",
        user_query="click submit",
        trigger_type="cron",
        cron_expression="0 9 * * *",
        metadata={"agent_expertise": "gui"}
    )

    await get_scheduler().run_now(auto.schedule_id)
    await asyncio.sleep(0.2)

    assert len(routed_queries) == 1
    assert routed_queries[0] == "[Agent: gui] click submit"


# --- Cross-node AutomationEngine Routing Tests (#313) ---

@pytest.mark.asyncio
async def test_cross_node_delegation_via_engine():
    engine = AutomationEngine(coordinator=make_mock_coordinator(delegate_result=True))

    local_calls = []
    async def mock_callback(query, session_id):
        local_calls.append(query)
        return {"status": "ok"}

    engine._orig_callback = mock_callback

    auto = engine.create_automation(
        name="remote_job",
        user_query="collect metrics",
        trigger_type="cron",
        cron_expression="0 9 * * *",
        metadata={"agent_expertise": "ops"},
    )

    await get_scheduler().run_now(auto.schedule_id)
    await asyncio.sleep(0.2)

    # Delegated cross-node with the routed query; local callback bypassed
    task_id, payload = engine.coordinator.delegate_task.call_args[0]
    assert task_id == auto.schedule_id
    assert payload["command"] == "[Agent: ops] collect metrics"
    assert local_calls == []

    execution = get_scheduler().executions[-1]
    assert execution.status == "completed"
    assert execution.result["message"] == "Delegated execution succeeded."


@pytest.mark.asyncio
async def test_cross_node_delegation_falls_back_to_local():
    engine = AutomationEngine(coordinator=make_mock_coordinator(delegate_result=False))

    local_calls = []
    async def mock_callback(query, session_id):
        local_calls.append(query)
        return {"status": "ok"}

    engine._orig_callback = mock_callback

    auto = engine.create_automation(
        name="auto_job",
        user_query="cleanup temp",
        trigger_type="cron",
        cron_expression="0 9 * * *",
    )

    await get_scheduler().run_now(auto.schedule_id)
    await asyncio.sleep(0.2)

    # Delegation was attempted, failed, and execution fell back to local
    engine.coordinator.delegate_task.assert_called_once()
    assert local_calls == ["cleanup temp"]
    assert get_scheduler().executions[-1].status == "completed"


@pytest.mark.asyncio
async def test_execution_target_remote_fails_without_local_fallback():
    engine = AutomationEngine(coordinator=make_mock_coordinator(delegate_result=False))

    local_calls = []
    async def mock_callback(query, session_id):
        local_calls.append(query)
        return {"status": "ok"}

    engine._orig_callback = mock_callback

    auto = engine.create_automation(
        name="remote_only_job",
        user_query="gpu benchmark",
        trigger_type="cron",
        cron_expression="0 9 * * *",
        execution_target="remote",
    )

    await get_scheduler().run_now(auto.schedule_id)
    await asyncio.sleep(0.2)

    assert local_calls == []
    execution = get_scheduler().executions[-1]
    assert execution.status == "failed"
    assert "delegation failed" in execution.error


@pytest.mark.asyncio
async def test_execution_target_local_skips_delegation():
    engine = AutomationEngine(coordinator=make_mock_coordinator(delegate_result=True))

    local_calls = []
    async def mock_callback(query, session_id):
        local_calls.append(query)
        return {"status": "ok"}

    engine._orig_callback = mock_callback

    auto = engine.create_automation(
        name="local_job",
        user_query="rotate logs",
        trigger_type="cron",
        cron_expression="0 9 * * *",
        execution_target="local",
    )

    await get_scheduler().run_now(auto.schedule_id)
    await asyncio.sleep(0.2)

    engine.coordinator.delegate_task.assert_not_called()
    assert local_calls == ["rotate logs"]


@pytest.mark.asyncio
async def test_chain_triggers_after_cross_node_delegation():
    engine = AutomationEngine(coordinator=make_mock_coordinator(delegate_result=True))
    engine._orig_callback = None

    parent = engine.create_automation(
        name="remote_parent",
        user_query="run remote parent",
        trigger_type="cron",
        cron_expression="0 9 * * *",
    )
    child = engine.create_automation(
        name="remote_child",
        user_query="run remote child",
        trigger_type="chain",
        parent_id=parent.schedule_id,
    )

    await get_scheduler().run_now(parent.schedule_id)
    await asyncio.sleep(0.5)

    # Both parent and downstream chained child were delegated cross-node
    delegated_ids = [call.args[0] for call in engine.coordinator.delegate_task.call_args_list]
    assert parent.schedule_id in delegated_ids
    assert child.schedule_id in delegated_ids


@pytest.mark.asyncio
async def test_cross_node_delegation_hard_timeout_falls_back_to_local():
    """A hung coordinator.delegate_task must not block the engine forever (#316)."""
    coordinator = AsyncMock()
    coordinator.active_workers = {"remote-worker": object()}

    async def hang_forever(task_id, payload):
        await asyncio.sleep(10)
        return True

    coordinator.delegate_task.side_effect = hang_forever
    engine = AutomationEngine(coordinator=coordinator)
    # Isolate the engine-internal timeout boundary: leave schedule.timeout_seconds
    # unset so the scheduler's own outer asyncio.wait_for doesn't also fire and
    # race with the one under test.
    engine.DEFAULT_ROUTED_TIMEOUT = 0.05

    local_calls = []
    async def mock_callback(query, session_id):
        local_calls.append(query)
        return {"status": "ok"}

    engine._orig_callback = mock_callback

    auto = engine.create_automation(
        name="hung_auto_job",
        user_query="auto job",
        trigger_type="cron",
        cron_expression="0 9 * * *",
    )

    await asyncio.wait_for(get_scheduler().run_now(auto.schedule_id), timeout=2.0)

    # Timed out, target is "auto" so it fell back to local execution instead of hanging
    assert local_calls == ["auto job"]


@pytest.mark.asyncio
async def test_cross_node_delegation_hard_timeout_fatal_for_remote_target():
    coordinator = AsyncMock()
    coordinator.active_workers = {"remote-worker": object()}

    async def hang_forever(task_id, payload):
        await asyncio.sleep(10)
        return True

    coordinator.delegate_task.side_effect = hang_forever
    engine = AutomationEngine(coordinator=coordinator)
    engine.DEFAULT_ROUTED_TIMEOUT = 0.05
    engine._orig_callback = None

    auto = engine.create_automation(
        name="hung_remote_job",
        user_query="remote job",
        trigger_type="cron",
        cron_expression="0 9 * * *",
        execution_target="remote",
    )

    await asyncio.wait_for(get_scheduler().run_now(auto.schedule_id), timeout=2.0)

    execution = get_scheduler().executions[-1]
    assert execution.status == "failed"
    assert "timed out" in execution.error


@pytest.mark.asyncio
async def test_automation_engine_bootstrap():
    graph = BootstrapGraph()
    res = await graph.run()
    assert res.ok is True

    stages = [s.name for s in res.stages]
    assert "automation_engine" in stages

    stage_res = next(s for s in res.stages if s.name == "automation_engine")
    assert stage_res.ok is True
    assert stage_res.payload["initialized"] is True
    assert isinstance(stage_res.payload["automation_engine"], AutomationEngine)
