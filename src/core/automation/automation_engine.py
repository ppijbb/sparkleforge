import asyncio
import logging
import os
import time
from datetime import datetime, time as dtime
from typing import Any, Callable, Dict, List, Optional

from src.core.anvil.mode_controller import ExecutionMode
from src.core.observe.event_bus import EventBus
from src.core.scheduler import (
    ScheduleConfig,
    ScheduleExecution,
    Scheduler,
    ScheduleStatus,
    get_scheduler,
)

logger = logging.getLogger(__name__)


class AutomationEngine:
    """Orchestrates system automation triggers (cron, event, webhook, chains) and routes tasks to expert agents."""

    DEFAULT_ROUTED_TIMEOUT = 30.0

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        scheduler: Scheduler | None = None,
        event_bus: EventBus | None = None,
        coordinator: Any = None,
    ):
        self.scheduler = scheduler or get_scheduler()
        self.event_bus = event_bus or EventBus()
        if coordinator is not None:
            self.coordinator = coordinator
        elif not hasattr(self, "coordinator"):
            self.coordinator = getattr(self.scheduler, "coordinator", None)

        if hasattr(self, "_initialized") and self._initialized:
            # Re-apply callback wrapping in case testing fixtures reset scheduler callbacks
            self.scheduler.set_execution_callback(self._wrapped_execution_callback)
            return

        logger.info("Initializing Automation Engine...")
        self._initialized = True
        self._event_subscriptions = {}
        
        # Heartbeat daemon state
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_running = False
        self._heartbeat_interval = 10.0  # seconds
        self._active_hours_start = dtime(8, 0)
        self._active_hours_end = dtime(22, 0)
        self._active_hours_enforced = True

        # Wrap scheduler callback for chains and multi-agent routing
        self._orig_callback = self.scheduler.execution_callback
        self.scheduler.set_execution_callback(self._wrapped_execution_callback)
        
        # Start heartbeat daemon
        self.start_heartbeat_daemon()
        logger.info("✅ Automation Engine successfully initialized and hooked into Scheduler")

    def start_heartbeat_daemon(self, interval_seconds: float = 10.0):
        """Start the background heartbeat daemon checking HEARTBEAT.md and active hours."""
        if self._heartbeat_running:
            return
        self._heartbeat_interval = interval_seconds
        self._heartbeat_running = True
        try:
            loop = asyncio.get_running_loop()
            self._heartbeat_task = loop.create_task(self._heartbeat_loop())
        except RuntimeError:
            # No running event loop yet
            pass

    def stop_heartbeat_daemon(self):
        """Stop the background heartbeat daemon."""
        self._heartbeat_running = False
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

    def set_active_hours(self, start_hour: int, start_min: int, end_hour: int, end_min: int, enforce: bool = True):
        """Configure active hours for quiet-window enforcement."""
        self._active_hours_start = dtime(start_hour, start_min)
        self._active_hours_end = dtime(end_hour, end_min)
        self._active_hours_enforced = enforce

    def is_within_active_hours(self) -> bool:
        """Check if current time is within active hours."""
        if not self._active_hours_enforced:
            return True
        now_time = datetime.now().time()
        if self._active_hours_start <= self._active_hours_end:
            return self._active_hours_start <= now_time <= self._active_hours_end
        else:
            # Over-midnight range
            return now_time >= self._active_hours_start or now_time <= self._active_hours_end

    async def _heartbeat_loop(self):
        """Background loop executing HEARTBEAT.md checklist tasks respecting activeHours."""
        logger.info("AutomationEngine: Heartbeat daemon started.")
        while self._heartbeat_running:
            try:
                if self.is_within_active_hours():
                    await self.tick_heartbeat()
                else:
                    logger.debug("AutomationEngine: Outside activeHours. Heartbeat muted.")
            except Exception as e:
                logger.error(f"AutomationEngine: Error in heartbeat loop: {e}")
            await asyncio.sleep(self._heartbeat_interval)

    async def tick_heartbeat(self) -> Dict[str, Any]:
        """Parse HEARTBEAT.md if present and run pending checklist items."""
        heartbeat_path = "HEARTBEAT.md"
        checklist_items = []
        if os.path.exists(heartbeat_path):
            try:
                with open(heartbeat_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_s = line.strip()
                        if line_s.startswith("- [ ]") or line_s.startswith("* [ ]"):
                            checklist_items.append(line_s[5:].strip())
            except Exception as e:
                logger.warning(f"Failed to parse {heartbeat_path}: {e}")

        results = []
        for item in checklist_items:
            logger.info(f"Heartbeat checklist item running: {item}")
            try:
                res = await self._wrapped_execution_callback(f"Heartbeat task: {item}", "heartbeat-session")
                results.append({"item": item, "status": "success", "result": res})
            except Exception as e:
                logger.error(f"Heartbeat task failed [{item}]: {e}")
                results.append({"item": item, "status": "failed", "error": str(e)})

        return {"checked": len(checklist_items), "results": results}

    def set_execution_callback(self, callback: Callable[[str, str], Any]):
        """Set the original executor query callback."""
        self._orig_callback = callback
        self.scheduler.set_execution_callback(self._wrapped_execution_callback)

    async def _wrapped_execution_callback(self, user_query: str, session_id: str) -> Any:
        """Internal callback wrapper to run multi-agent routing, execute query, and trigger downstream chains."""
        # Resolve the active schedule configuration for this execution
        schedule = None
        for s in self.scheduler.schedules.values():
            # If the query matches (simple heuristic)
            if s.user_query == user_query:
                schedule = s
                break

        metadata = schedule.metadata if schedule else {}

        # Nightwelding sweep: keyed off the resolved schedule's metadata (not the
        # query string, which route_task() below may rewrite) so a recurring
        # Nightwelding schedule ticks the overnight backlog sweep instead of
        # being treated as a research query.
        if schedule and metadata.get("nightwelding_sweep"):
            from src.core.nightwelding.runner import run_nightwelding_sweep

            return await run_nightwelding_sweep()

        # 1. Multi-agent Routing
        routed_query = self.route_task(user_query, metadata)

        # 2. Execution (cross-node delegation when eligible, otherwise local)
        result = await self._execute_routed(routed_query, session_id, schedule, metadata)

        # 3. Chain Triggering (downstream tasks)
        if schedule:
            asyncio.create_task(self._trigger_chain(schedule.schedule_id))

        return result

    async def _execute_routed(
        self,
        routed_query: str,
        session_id: str,
        schedule: ScheduleConfig | None,
        metadata: Dict[str, Any],
    ) -> Any:
        """Execute a routed task, delegating cross-node via the coordinator when eligible.

        ``execution_target`` metadata controls placement: "local" forces local
        execution, "remote" requires a worker node (failure is fatal), and
        "auto" (default) prefers a worker but falls back to local execution.
        """
        target = metadata.get("execution_target", "auto")
        if target != "local" and self.coordinator is not None and getattr(self.coordinator, "active_workers", None):
            task_id = schedule.schedule_id if schedule else session_id
            timeout = float(
                schedule.timeout_seconds if schedule and schedule.timeout_seconds else self.DEFAULT_ROUTED_TIMEOUT
            )
            payload = {"command": routed_query, "timeout": timeout}
            logger.info(f"AutomationEngine [Cross-node]: Delegating task '{task_id}' via coordinator.")
            try:
                ok = await asyncio.wait_for(self.coordinator.delegate_task(task_id, payload), timeout=timeout)
            except TimeoutError:
                # Hard boundary: never block the scheduler loop indefinitely on a hung
                # coordinator/worker, even though the timeout is also passed down to
                # the session layer as an (advisory, implementation-dependent) hint.
                logger.error(
                    f"AutomationEngine [Cross-node]: Delegation for '{task_id}' timed out after {timeout}s."
                )
                if target == "remote":
                    raise RuntimeError(
                        f"Cross-node delegation for task '{task_id}' timed out after {timeout}s."
                    )
                ok = False
            if ok:
                return {
                    "status": "success",
                    "message": "Delegated execution succeeded.",
                    "routed_query": routed_query,
                }
            if target == "remote":
                raise RuntimeError(
                    f"Cross-node delegation failed for task '{task_id}' and execution_target is 'remote'."
                )
            logger.warning(
                f"AutomationEngine [Cross-node]: Delegation failed for '{task_id}'; falling back to local execution."
            )

        if self._orig_callback:
            return await self._orig_callback(routed_query, session_id)
        logger.warning("AutomationEngine: No original scheduler callback configured. Execution skipped.")
        return {"status": "skipped", "reason": "no callback"}

    def route_task(self, query: str, metadata: Dict[str, Any]) -> str:
        """Route task to specialized agent based on routing metadata or tags."""
        expertise = metadata.get("agent_expertise")
        if expertise:
            logger.info(f"AutomationEngine [Routing]: Routing query '{query}' to expert agent [{expertise.upper()}]")
            # In a full system, this would delegate to a specific AgentOrchestrator instance.
            # We append metadata routing for system context.
            return f"[Agent: {expertise}] {query}"
        return query

    def create_automation(
        self,
        name: str,
        user_query: str,
        trigger_type: str = "cron",  # cron, event, webhook, chain
        cron_expression: str | None = None,
        event_type: str | None = None,
        webhook_id: str | None = None,
        parent_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
        tags: List[str] | None = None,
        execution_target: str | None = None,  # local, remote, auto
    ) -> ScheduleConfig:
        """Create an automation task ruleset."""
        metadata = metadata or {}
        tags = tags or []

        metadata["trigger_type"] = trigger_type
        if execution_target:
            metadata["execution_target"] = execution_target
        if event_type:
            metadata["event_type"] = event_type
        if webhook_id:
            metadata["webhook_id"] = webhook_id
        if parent_id:
            metadata["parent_id"] = parent_id

        # Non-cron automations use a placeholder cron and are disabled for cron-loop runner
        # so they can only be explicitly triggered via run_now()
        is_cron = trigger_type.lower() == "cron"
        cron_expr = cron_expression if is_cron else "0 0 1 1 *" # dummy cron
        enabled = is_cron

        schedule = self.scheduler.create_schedule(
            name=name,
            cron_expression=cron_expr,
            user_query=user_query,
            enabled=enabled,
            metadata=metadata,
            tags=tags
        )
        
        # If disabled for cron, make sure the status represents disabled/paused
        if not is_cron:
            schedule.status = ScheduleStatus.DISABLED

        # Setup event trigger hook
        if trigger_type.lower() == "event" and event_type:
            self._setup_event_hook(event_type, schedule.schedule_id)

        return schedule

    def _setup_event_hook(self, event_type: str, schedule_id: str):
        """Subscribe to the event bus to trigger this automation on events."""
        if event_type not in self._event_subscriptions:
            async def event_callback(data):
                logger.info(f"AutomationEngine: Event '{event_type}' received. Triggering automations.")
                await self.trigger_event(event_type, data)

            sub_id = self.event_bus.subscribe(event_type, event_callback)
            self._event_subscriptions[event_type] = sub_id

    async def trigger_event(self, event_type: str, data: Any) -> List[ScheduleExecution]:
        """Trigger all event-type automations registered for this event."""
        executions = []
        for schedule in list(self.scheduler.schedules.values()):
            meta = schedule.metadata
            if meta.get("trigger_type") == "event" and meta.get("event_type") == event_type:
                logger.info(f"AutomationEngine: Event-triggering automation: {schedule.schedule_id}")
                try:
                    exec_record = await self.scheduler.run_now(schedule.schedule_id)
                    executions.append(exec_record)
                except Exception as e:
                    logger.error(f"AutomationEngine: Failed to run event automation {schedule.schedule_id}: {e}")
        return executions

    async def trigger_webhook(self, webhook_id: str, payload: Dict[str, Any]) -> List[ScheduleExecution]:
        """Manually trigger webhook-based automations."""
        executions = []
        for schedule in list(self.scheduler.schedules.values()):
            meta = schedule.metadata
            if meta.get("trigger_type") == "webhook" and meta.get("webhook_id") == webhook_id:
                logger.info(f"AutomationEngine: Webhook-triggering automation: {schedule.schedule_id}")
                # We can store the payload in metadata before running
                schedule.metadata["last_webhook_payload"] = payload
                try:
                    exec_record = await self.scheduler.run_now(schedule.schedule_id)
                    executions.append(exec_record)
                except Exception as e:
                    logger.error(f"AutomationEngine: Failed to run webhook automation {schedule.schedule_id}: {e}")
        return executions

    async def _trigger_chain(self, parent_schedule_id: str) -> List[ScheduleExecution]:
        """Trigger downstream chained tasks when a parent task completes."""
        executions = []
        for schedule in list(self.scheduler.schedules.values()):
            meta = schedule.metadata
            if meta.get("trigger_type") == "chain" and meta.get("parent_id") == parent_schedule_id:
                logger.info(f"AutomationEngine: Chain-triggering downstream automation: {schedule.schedule_id} (parent: {parent_schedule_id})")
                try:
                    exec_record = await self.scheduler.run_now(schedule.schedule_id)
                    executions.append(exec_record)
                except Exception as e:
                    logger.error(f"AutomationEngine: Failed to run chained automation {schedule.schedule_id}: {e}")
        return executions

    def shutdown(self):
        """Cleanup event handlers and subscriptions."""
        for event_type, sub_id in list(self._event_subscriptions.items()):
            self.event_bus.unsubscribe(sub_id)
        self._event_subscriptions.clear()
