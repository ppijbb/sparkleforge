import asyncio
import logging
from typing import Dict, List, Optional

from src.core.observe.event_bus import EventBus
from src.core.actuate.iot_device import SensorDevice

logger = logging.getLogger(__name__)


class IOTTelemetryLoop:
    """Background loop that polls registered IoT sensors and publishes telemetry to the EventBus."""

    def __init__(self, event_bus: EventBus, interval: float = 1.0):
        self.event_bus = event_bus
        self.interval = interval
        self.sensors: List[SensorDevice] = []
        self._task: Optional[asyncio.Task] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if the telemetry loop is active."""
        return self._is_running

    def register_sensor(self, sensor: SensorDevice) -> None:
        """Register a sensor device to poll."""
        if sensor not in self.sensors:
            self.sensors.append(sensor)
            logger.info(f"IOTTelemetryLoop: Registered sensor '{sensor.device_id}'")

    def start(self) -> None:
        """Start the background telemetry polling task."""
        if self._is_running:
            return
        self._is_running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("IOTTelemetryLoop: Background loop started.")

    async def stop(self) -> None:
        """Stop the background telemetry polling task."""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("IOTTelemetryLoop: Background loop stopped.")

    async def _poll_loop(self) -> None:
        while self._is_running:
            for sensor in self.sensors:
                if not sensor.is_connected:
                    continue
                try:
                    data = sensor.read()
                    payload = {
                        "device_id": sensor.device_id,
                        "metrics": data
                    }
                    await self.event_bus.publish("sensor_telemetry", payload)
                    logger.debug(f"IOTTelemetryLoop: Published telemetry for '{sensor.device_id}': {data}")
                except Exception as e:
                    logger.error(f"IOTTelemetryLoop: Error polling sensor '{sensor.device_id}': {e}")
            await asyncio.sleep(self.interval)
