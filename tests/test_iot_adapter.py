import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Mock pyautogui completely to bypass display dependency in headless test runs
sys.modules['pyautogui'] = MagicMock()

import pytest

from src.core.actuate.iot_device import (
    GPIODevice,
    SerialDevice,
    USBHIDDevice,
    RobotArmDevice,
    SensorDevice,
    CameraDevice,
)
from src.core.actuate.actuation_plane import ActuationPlane
from src.core.observe.event_bus import EventBus
from src.core.observe.iot_telemetry_loop import IOTTelemetryLoop
from src.core.guard.guard_plane import GuardPlane
from src.core.guard.capability_manager import CapabilityManager


# --- 1. Basic Adapter Tests ---

def test_gpio_device_mock():
    gpio = GPIODevice("test_gpio")
    assert gpio.connect() is True
    
    # Write pin states
    gpio.write({4: 1, 17: 0})
    assert gpio.read() == {4: 1, 17: 0}
    
    # Command interface
    res = gpio.execute_command("get_pin 4")
    assert res["status"] == "success"
    assert res["stdout"] == "1"
    
    gpio.disconnect()
    assert gpio.is_connected is False


def test_serial_device_mock():
    serial = SerialDevice("test_serial")
    assert serial.connect() is True
    
    # Command TX/RX exchange
    res = serial.execute_command("PING")
    assert res["status"] == "success"
    assert res["stdout"] == "PONG"
    
    serial.disconnect()


def test_usb_hid_device_mock():
    hid = USBHIDDevice("test_hid")
    assert hid.connect() is True
    
    res = hid.execute_command("hello")
    assert res["status"] == "success"
    assert res["stdout"] == "hello".encode().hex()
    
    hid.disconnect()


# --- 2. IoT Device Wrappers Tests ---

def test_robot_arm_device():
    arm = RobotArmDevice("test_arm")
    assert arm.connect() is True
    
    # Command move
    res = arm.execute_command("move_joint 2 90")
    assert res["status"] == "success"
    assert arm.joints[2] == 90
    assert "moved to 90" in res["stdout"]
    
    # Invalid command
    res = arm.execute_command("move_joint 5 90")
    assert res["status"] == "failed"
    
    arm.disconnect()


def test_camera_device():
    camera = CameraDevice("test_cam")
    assert camera.connect() is True
    
    res = camera.execute_command("capture")
    assert res["status"] == "success"
    assert "Captured frame" in res["stdout"]
    
    camera.disconnect()


# --- 3. Observe Telemetry Integration Tests ---

@pytest.mark.asyncio
async def test_telemetry_loop_streaming():
    event_bus = EventBus()
    tele_loop = IOTTelemetryLoop(event_bus, interval=0.01)
    
    sensor = SensorDevice("climate_sensor")
    sensor.connect()
    
    tele_loop.register_sensor(sensor)
    
    telemetry_received = []
    
    async def sensor_callback(data):
        telemetry_received.append(data)
        
    sub_id = event_bus.subscribe("sensor_telemetry", sensor_callback)
    
    tele_loop.start()
    
    # Wait for a couple of polls
    await asyncio.sleep(0.05)
    
    await tele_loop.stop()
    event_bus.unsubscribe(sub_id)
    
    assert len(telemetry_received) > 0
    assert telemetry_received[0]["device_id"] == "climate_sensor"
    assert "temperature" in telemetry_received[0]["metrics"]
    assert "humidity" in telemetry_received[0]["metrics"]


# --- 4. GuardPlane Capabilities Verification ---

@pytest.mark.asyncio
async def test_guard_plane_blocks_iot_unauthorized():
    guard = GuardPlane()
    guard.capability_manager.reset()
    
    # Reset singleton of ActuationPlane
    ActuationPlane._instance = None
    actuator = ActuationPlane()
    
    camera = CameraDevice("secure_cam")
    camera.connect()
    actuator.register_device("secure_cam", camera)
    
    # Agent lacks "iot_control" capability
    res = guard.check_and_control_device(
        agent_id="attacker_agent",
        device_id="secure_cam",
        command="capture",
        description="Try unauthorized capture",
        is_write=True,
    )
    
    assert res["ok"] is False
    assert "Missing capability" in res["error"]


@pytest.mark.asyncio
async def test_guard_plane_allows_iot_after_grant():
    guard = GuardPlane()
    guard.capability_manager.reset()
    
    # Reset singleton of ActuationPlane
    ActuationPlane._instance = None
    actuator = ActuationPlane()
    
    arm = RobotArmDevice("lab_arm")
    arm.connect()
    actuator.register_device("lab_arm", arm)
    
    # Grant control capability to the agent
    guard.capability_manager.grant_agent("operator_agent", "iot_control")
    
    # Disable HITL requirement manually for test automation (or mock hitl)
    cap = guard.capability_manager.get_capability("iot_control")
    cap.requires_hitl = False
    
    res = guard.check_and_control_device(
        agent_id="operator_agent",
        device_id="lab_arm",
        command="move_joint 1 45",
        description="Rotate base joint",
        is_write=True,
    )
    
    assert res["ok"] is True
    assert "moved to 45" in res["stdout"]
