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


# --- 1b. Hardware Backend Selection & Fallback Tests ---

def test_auto_backend_falls_back_to_mock_without_hardware():
    # No driver library / device in the test environment -> mock fallback
    serial = SerialDevice("fallback_serial", backend="auto")
    assert serial.connect() is True
    assert serial.active_backend == "mock"
    serial.disconnect()
    assert serial.active_backend is None


def test_mock_backend_forced_even_if_hardware_present():
    gpio = GPIODevice("forced_mock_gpio", backend="mock")
    assert gpio.connect() is True
    assert gpio.active_backend == "mock"
    gpio.disconnect()


def test_hardware_backend_required_fails_without_driver():
    for device in (
        GPIODevice("hw_gpio", backend="hardware"),
        SerialDevice("hw_serial", backend="hardware"),
        USBHIDDevice("hw_hid", backend="hardware"),
    ):
        assert device.connect() is False
        assert device.is_connected is False


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        GPIODevice("bad_backend", backend="quantum")


def test_serial_hardware_backend_with_driver():
    class FakeSerialPort:
        def __init__(self, port, baudrate, timeout=None):
            self.port = port
            self.baudrate = baudrate
            self.timeout = timeout
            self.written = b""
            self.closed = False

        def write(self, data):
            self.written += data

        def readline(self):
            return b"PONG\n"

        def close(self):
            self.closed = True

    fake_serial_module = MagicMock()
    fake_serial_module.Serial = FakeSerialPort

    with patch.dict(sys.modules, {"serial": fake_serial_module}):
        serial = SerialDevice("real_serial", port="/dev/ttyTEST", baudrate=115200, backend="hardware")
        assert serial.connect() is True
        assert serial.active_backend == "hardware"

        res = serial.execute_command("PING")
        assert res["status"] == "success"
        assert res["stdout"] == "PONG"
        assert serial._serial.written == b"PING\n"
        assert serial._serial.port == "/dev/ttyTEST"
        assert serial._serial.baudrate == 115200

        port = serial._serial
        serial.disconnect()
        assert port.closed is True


def test_usb_hid_hardware_backend_with_driver():
    class FakeHIDHandle:
        def __init__(self):
            self.opened_with = None
            self.written = []
            self.closed = False

        def open(self, vendor_id, product_id):
            self.opened_with = (vendor_id, product_id)

        def read(self, size, timeout_ms):
            return [0x01, 0x02]

        def write(self, data):
            self.written.append(bytes(data))

        def close(self):
            self.closed = True

    fake_hid_module = MagicMock()
    fake_hid_module.device = FakeHIDHandle

    with patch.dict(sys.modules, {"hid": fake_hid_module}):
        hid_dev = USBHIDDevice("real_hid", vendor_id=0x1234, product_id=0x5678, backend="hardware")
        assert hid_dev.connect() is True
        assert hid_dev.active_backend == "hardware"
        assert hid_dev._hid.opened_with == (0x1234, 0x5678)

        assert hid_dev.read() == b"\x01\x02"
        hid_dev.write(b"\xaa\xbb")
        assert hid_dev._hid.written == [b"\xaa\xbb"]

        handle = hid_dev._hid
        hid_dev.disconnect()
        assert handle.closed is True


def test_gpio_hardware_backend_with_driver():
    class FakeLineRequest:
        def __init__(self):
            self.values = {}
            self.released = False

        def set_value(self, pin, value):
            self.values[pin] = value

        def release(self):
            self.released = True

    class FakeChip:
        def __init__(self, path):
            self.path = path
            self.requests = []
            self.closed = False

        def request_lines(self, consumer, config):
            request = FakeLineRequest()
            self.requests.append((consumer, config, request))
            return request

        def close(self):
            self.closed = True

    fake_line_module = MagicMock()
    fake_line_module.Direction.OUTPUT = "output"
    fake_line_module.Value.ACTIVE = "active"
    fake_line_module.Value.INACTIVE = "inactive"

    fake_gpiod_module = MagicMock()
    fake_gpiod_module.Chip = FakeChip
    fake_gpiod_module.line = fake_line_module

    with patch.dict(sys.modules, {"gpiod": fake_gpiod_module, "gpiod.line": fake_line_module}):
        gpio = GPIODevice("real_gpio", backend="hardware", chip_path="/dev/gpiochip9")
        assert gpio.connect() is True
        assert gpio.active_backend == "hardware"
        assert gpio._chip.path == "/dev/gpiochip9"

        gpio.write({4: 1, 17: 0})
        assert gpio.read() == {4: 1, 17: 0}
        # Lines were requested and driven through the driver
        assert len(gpio._chip.requests) == 2
        assert gpio._line_requests[4].values[4] == "active"
        assert gpio._line_requests[17].values[17] == "inactive"

        chip = gpio._chip
        requests = list(gpio._line_requests.values())
        gpio.disconnect()
        assert chip.closed is True
        assert all(r.released for r in requests)


def test_robot_arm_passes_backend_to_serial_adapter():
    arm = RobotArmDevice("backend_arm", backend="mock")
    assert arm.connect() is True
    assert arm.active_backend == "mock"
    arm.disconnect()


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
