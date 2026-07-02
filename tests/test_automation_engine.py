import asyncio
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
    yield
    scheduler.schedules.clear()
    scheduler.executions.clear()


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
