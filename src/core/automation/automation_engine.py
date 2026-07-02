import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
from src.core.scheduler import get_scheduler, Scheduler, ScheduleConfig, ScheduleExecution, ScheduleStatus
from src.core.observe.event_bus import EventBus

logger = logging.getLogger(__name__)


class AutomationEngine:
    """Orchestrates system automation triggers (cron, event, webhook, chains) and routes tasks to expert agents."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, scheduler: Optional[Scheduler] = None, event_bus: Optional[EventBus] = None):
        self.scheduler = scheduler or get_scheduler()
        self.event_bus = event_bus or EventBus()

        if hasattr(self, "_initialized") and self._initialized:
            # Re-apply callback wrapping in case testing fixtures reset scheduler callbacks
            self.scheduler.set_execution_callback(self._wrapped_execution_callback)
            return

        logger.info("Initializing Automation Engine...")
        self._initialized = True
        self._event_subscriptions = {}
        
        # Wrap scheduler callback for chains and multi-agent routing
        self._orig_callback = self.scheduler.execution_callback
        self.scheduler.set_execution_callback(self._wrapped_execution_callback)
        logger.info("✅ Automation Engine successfully initialized and hooked into Scheduler")

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
        
        # 1. Multi-agent Routing
        routed_query = self.route_task(user_query, metadata)

        # 2. Execution
        result = None
        if self._orig_callback:
            result = await self._orig_callback(routed_query, session_id)
        else:
            logger.warning("AutomationEngine: No original scheduler callback configured. Execution skipped.")
            result = {"status": "skipped", "reason": "no callback"}

        # 3. Chain Triggering (downstream tasks)
        if schedule:
            asyncio.create_task(self._trigger_chain(schedule.schedule_id))

        return result

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
        cron_expression: Optional[str] = None,
        event_type: Optional[str] = None,
        webhook_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> ScheduleConfig:
        """Create an automation task ruleset."""
        metadata = metadata or {}
        tags = tags or []
        
        metadata["trigger_type"] = trigger_type
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
