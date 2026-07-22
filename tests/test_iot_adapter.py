import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Mock pyautogui completely to bypass display dependency in headless test runs
sys.modules['pyautogui'] = MagicMock()

import pytest

from src.core.actuate.actuation_plane import ActuationPlane
from src.core.actuate.iot_device import (
    CameraDevice,
    GPIODevice,
    RobotArmDevice,
    SensorDevice,
    SerialDevice,
    USBHIDDevice,
)
from src.core.guard.capability_manager import CapabilityManager
from src.core.guard.guard_plane import GuardPlane, register_iot_guard_tools
from src.core.observe.event_bus import EventBus
from src.core.observe.iot_telemetry_loop import IOTTelemetryLoop
from src.core.tools.registry import registry

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
        def __init__(self, pins=None):
            self.values = {}
            self.released = False
            self.pins = pins or []

        def set_value(self, pin, value):
            self.values[pin] = value

        def get_value(self, pin=None):
            if pin is None:
                pin = self.pins[0] if self.pins else None
            return self.values.get(pin)

        def release(self):
            self.released = True

    class FakeChip:
        def __init__(self, path):
            self.path = path
            self.requests = []
            self.closed = False

        def request_lines(self, consumer, config):
            pins = list(config.keys())
            request = FakeLineRequest(pins=pins)
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


def test_gpio_hardware_read_queries_driver_not_just_cache():
    """read() must reflect the physical line state, not only the last-written cache (#319)."""

    class FakeLineRequest:
        def __init__(self, pins=None):
            self.values = {}
            self.pins = pins or []

        def set_value(self, pin, value):
            self.values[pin] = value

        def get_value(self, pin=None):
            if pin is None:
                pin = self.pins[0] if self.pins else None
            return self.values.get(pin)

        def release(self):
            pass

    class FakeChip:
        def __init__(self, path):
            pass

        def request_lines(self, consumer, config):
            return FakeLineRequest(list(config.keys()))

        def close(self):
            pass

    fake_line_module = MagicMock()
    fake_line_module.Direction.INPUT = "input"
    fake_line_module.Direction.OUTPUT = "output"
    fake_line_module.Value.ACTIVE = "active"
    fake_line_module.Value.INACTIVE = "inactive"

    fake_gpiod_module = MagicMock()
    fake_gpiod_module.Chip = FakeChip
    fake_gpiod_module.line = fake_line_module
    fake_gpiod_module.LineSettings = MagicMock()

    with patch.dict(sys.modules, {"gpiod": fake_gpiod_module, "gpiod.line": fake_line_module}):
        gpio = GPIODevice("drift_gpio", backend="hardware")
        gpio.connect()
        gpio.execute_command("configure_pin 4 input")

        # Something external forces the physical line to a different value; read() must surface that drift.
        gpio._line_requests[4].values[4] = "inactive"
        assert gpio.read() == {4: 0}
        gpio.disconnect()


def test_usb_hid_write_reports_failure_on_short_write():
    """write() must not claim success when hidapi reports a short/failed write (#319)."""

    class FlakyHIDHandle:
        def open(self, vendor_id, product_id):
            pass

        def write(self, data):
            return -1  # hidapi failure sentinel

        def close(self):
            pass

    fake_hid_module = MagicMock()
    fake_hid_module.device = FlakyHIDHandle

    with patch.dict(sys.modules, {"hid": fake_hid_module}):
        hid_dev = USBHIDDevice("flaky_hid", backend="hardware")
        assert hid_dev.connect() is True
        assert hid_dev.write(b"\xaa\xbb") is False


def test_serial_backend_is_keyword_only():
    with pytest.raises(TypeError):
        SerialDevice("id", "/dev/ttyUSB0", 9600, 2.0, "auto")  # backend passed positionally


def test_usb_hid_backend_is_keyword_only():
    with pytest.raises(TypeError):
        USBHIDDevice("id", 0x046d, 0xc077, 64, 1000, "auto")  # backend passed positionally


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


def test_sensor_hardware_backend_with_driver():
    class FakeDHT:
        def __init__(self, pin):
            self.pin = pin
            self.exited = False
            self._reads = [
                (float("nan"), float("nan")),
                (22.5, 46.0),
            ]
            self._call_count = 0

        @property
        def temperature(self):
            idx = min(self._call_count, len(self._reads) - 1)
            return self._reads[idx][0]

        @property
        def humidity(self):
            idx = min(self._call_count, len(self._reads) - 1)
            self._call_count += 1
            return self._reads[idx][1]

        def exit(self):
            self.exited = True

    fake_dht_module = MagicMock()
    fake_dht_module.DHT22 = FakeDHT

    with patch.dict(sys.modules, {"adafruit_dht": fake_dht_module}):
        sensor = SensorDevice("real_sensor", backend="hardware", pin=18)
        assert sensor.connect() is True
        assert sensor.active_backend == "hardware"
        assert sensor._dht.pin == 18

        # First read returns NaN, second succeeds
        data = sensor.read()
        assert data["temperature"] == 22.5
        assert data["humidity"] == 46.0

        dht = sensor._dht
        sensor.disconnect()
        assert dht.exited is True


def test_sensor_hardware_read_failure_falls_back():
    class FlakyDHT:
        def __init__(self, pin):
            self.pin = pin

        @property
        def temperature(self):
            raise RuntimeError("sensor not responding")

        @property
        def humidity(self):
            raise RuntimeError("sensor not responding")

        def exit(self):
            pass

    fake_dht_module = MagicMock()
    fake_dht_module.DHT22 = FlakyDHT

    with patch.dict(sys.modules, {"adafruit_dht": fake_dht_module}):
        sensor = SensorDevice("flaky_sensor", backend="hardware", pin=4)
        assert sensor.connect() is True
        with pytest.raises(RuntimeError):
            sensor.read()


def test_camera_hardware_backend_with_driver():
    class FakeVideoCapture:
        def __init__(self, source):
            self.source = source
            self.opened = True
            self.released = False

        def isOpened(self):
            return self.opened

        def read(self):
            return True, "fake-frame"

        def release(self):
            self.released = True

    fake_cv2_module = MagicMock()
    fake_cv2_module.VideoCapture = FakeVideoCapture
    fake_cv2_module.imencode.return_value = (True, bytearray(b"\xff\xd8\xff\xe0\x00\x10JFIF"))

    with patch.dict(sys.modules, {"cv2": fake_cv2_module}):
        camera = CameraDevice("real_cam", backend="hardware", source=1)
        assert camera.connect() is True
        assert camera.active_backend == "hardware"
        assert camera._cap.source == 1

        frame = camera.read()
        assert frame.startswith(b"\xff\xd8\xff\xe0")

        cap = camera._cap
        camera.disconnect()
        assert cap.released is True


def test_camera_hardware_open_failure():
    class ClosedVideoCapture:
        def __init__(self, source):
            self.source = source

        def isOpened(self):
            return False

    fake_cv2_module = MagicMock()
    fake_cv2_module.VideoCapture = ClosedVideoCapture

    with patch.dict(sys.modules, {"cv2": fake_cv2_module}):
        camera = CameraDevice("closed_cam", backend="hardware", source=0)
        assert camera.connect() is False
        assert camera.is_connected is False


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
    res = await guard.check_and_control_device(
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
    
    res = await guard.check_and_control_device(
        agent_id="operator_agent",
        device_id="lab_arm",
        command="move_joint 1 45",
        description="Rotate base joint",
        is_write=True,
    )
    
    assert res["ok"] is True
    assert "moved to 45" in res["stdout"]


# --- 5. Production tool wiring (issue #783) ---


def test_control_iot_device_tool_is_registered():
    # check_and_control_device used to be reachable only from tests -- no
    # agent-callable tool exposed it in production. register_iot_guard_tools
    # should make it discoverable through the shared tool registry.
    register_iot_guard_tools()

    assert "control_iot_device" in registry.get_all_tool_names()
    assert registry.tool_sources["control_iot_device"] == "local"


@pytest.mark.asyncio
async def test_control_iot_device_tool_enforces_capability_check():
    register_iot_guard_tools()
    GuardPlane().capability_manager.reset()

    ActuationPlane._instance = None
    actuator = ActuationPlane()
    camera = CameraDevice("registry_cam")
    camera.connect()
    actuator.register_device("registry_cam", camera)

    result = await registry.execute(
        "control_iot_device",
        {
            "agent_id": "attacker_agent",
            "device_id": "registry_cam",
            "command": "capture",
            "is_write": True,
        },
    )

    assert result["ok"] is False
    assert "Missing capability" in result["error"]


@pytest.mark.asyncio
async def test_control_iot_device_tool_executes_after_grant():
    register_iot_guard_tools()
    guard = GuardPlane()
    guard.capability_manager.reset()

    ActuationPlane._instance = None
    actuator = ActuationPlane()
    arm = RobotArmDevice("registry_arm")
    arm.connect()
    actuator.register_device("registry_arm", arm)

    guard.capability_manager.grant_agent("operator_agent", "iot_control")
    cap = guard.capability_manager.get_capability("iot_control")
    cap.requires_hitl = False

    result = await registry.execute(
        "control_iot_device",
        {
            "agent_id": "operator_agent",
            "device_id": "registry_arm",
            "command": "move_joint 1 45",
            "is_write": True,
        },
    )

    assert result["ok"] is True
    assert "moved to 45" in result["stdout"]


def test_gpio_input_configuration_and_commands():
    """Test configure_pin, read_pin, get_pin under mock and fake hardware backends."""
    # 1. Test configure_pin in mock mode
    gpio = GPIODevice("test_gpio_conf", backend="mock")
    assert gpio.connect() is True
    
    # Configure pin 5 as input with pull_up bias and active_low
    res = gpio.execute_command("configure_pin 5 input pull_up active_low")
    assert res["status"] == "success"
    assert gpio.pin_directions[5] == "input"
    assert gpio.pin_biases[5] == "pull_up"
    assert gpio.pin_active_lows[5] is True

    # 2. Test configure_pin in fake hardware mode
    class FakeLineRequest:
        def __init__(self):
            self.settings = {}
            self.values = {}
            self.released = False

        def release(self):
            self.released = True

        def set_value(self, pin, value):
            self.values[pin] = value

        def get_value(self, pin=None):
            if pin is None:
                pin = list(self.settings.keys())[0] if self.settings else None
            return self.values.get(pin, "inactive")

    class FakeChip:
        def __init__(self, path):
            self.requests = {}

        def request_lines(self, consumer, config):
            req = FakeLineRequest()
            for pin, settings in config.items():
                req.settings[pin] = settings
                self.requests[pin] = req
            return req

        def close(self):
            pass

    fake_line_module = MagicMock()
    fake_line_module.Direction.INPUT = "input"
    fake_line_module.Direction.OUTPUT = "output"
    fake_line_module.Bias.PULL_UP = "pull_up"
    fake_line_module.Bias.PULL_DOWN = "pull_down"
    fake_line_module.Bias.DISABLED = "disabled"
    fake_line_module.Bias.AS_IS = "as_is"
    fake_line_module.Value.ACTIVE = "active"
    fake_line_module.Value.INACTIVE = "inactive"

    class FakeLineSettings:
        def __init__(self, direction=None, bias=None, active_low=False):
            self.direction = direction
            self.bias = bias
            self.active_low = active_low

    fake_gpiod_module = MagicMock()
    fake_gpiod_module.Chip = FakeChip
    fake_gpiod_module.LineSettings = FakeLineSettings
    fake_gpiod_module.line = fake_line_module

    with patch.dict(sys.modules, {"gpiod": fake_gpiod_module, "gpiod.line": fake_line_module}):
        gpio_hw = GPIODevice("hw_gpio_conf", backend="hardware")
        assert gpio_hw.connect() is True
        
        # Configure input pin
        res = gpio_hw.execute_command("configure_pin 12 input pull_down active_high")
        assert res["status"] == "success"
        req = gpio_hw._line_requests[12]
        assert req.settings[12].direction == "input"
        assert req.settings[12].bias == "pull_down"
        assert req.settings[12].active_low is False

        # Read pin (it will try to read from the fake request)
        req.set_value(12, "active")
        
        res = gpio_hw.execute_command("read_pin 12")
        assert res["status"] == "success"
        assert res["stdout"] == "1"


def test_gpio_mock_input_commands():
    """Test mock_input command under mock backend."""
    gpio = GPIODevice("mock_gpio_test", backend="mock")
    assert gpio.connect() is True
    
    # Configure input pin
    gpio.execute_command("configure_pin 14 input")
    
    # Set mock input state
    res = gpio.execute_command("mock_input 14 1")
    assert res["status"] == "success"
    
    # Read back input pin state
    read_res = gpio.execute_command("read_pin 14")
    assert read_res["status"] == "success"
    assert read_res["stdout"] == "1"
    
    # Get pin cache read
    get_res = gpio.execute_command("get_pin 14")
    assert get_res["status"] == "success"
    assert get_res["stdout"] == "1"


def test_gpio_direction_state_invariants():
    """Verify that read() doesn't silently flip output pin direction, and write() on input pin raises error."""
    class FakeLineRequest:
        def __init__(self):
            self.settings = {}
            self.values = {}
            self.released = False

        def release(self):
            self.released = True

        def set_value(self, pin, value):
            self.values[pin] = value

        def get_value(self, pin=None):
            if pin is None:
                pin = list(self.settings.keys())[0] if self.settings else None
            return self.values.get(pin, "inactive")

    class FakeChip:
        def __init__(self, path):
            self.requests = {}

        def request_lines(self, consumer, config):
            req = FakeLineRequest()
            for pin, settings in config.items():
                req.settings[pin] = settings
                self.requests[pin] = req
            return req

        def close(self):
            pass

    fake_line_module = MagicMock()
    fake_line_module.Direction.INPUT = "input"
    fake_line_module.Direction.OUTPUT = "output"
    fake_line_module.Bias.AS_IS = "as_is"
    fake_line_module.Value.ACTIVE = "active"
    fake_line_module.Value.INACTIVE = "inactive"

    class FakeLineSettings:
        def __init__(self, direction=None, bias=None, active_low=False):
            self.direction = direction
            self.bias = bias
            self.active_low = active_low

    fake_gpiod_module = MagicMock()
    fake_gpiod_module.Chip = FakeChip
    fake_gpiod_module.LineSettings = FakeLineSettings
    fake_gpiod_module.line = fake_line_module

    with patch.dict(sys.modules, {"gpiod": fake_gpiod_module, "gpiod.line": fake_line_module}):
        gpio = GPIODevice("inv_gpio", backend="hardware")
        assert gpio.connect() is True

        # Configure pin 10 as output
        gpio.execute_command("configure_pin 10 output")
        req10 = gpio._line_requests[10]
        
        # Configure pin 11 as input
        gpio.execute_command("configure_pin 11 input")
        req11 = gpio._line_requests[11]

        # 1. Calling read() should not reconfigure pin 10 (output), but should return its cached state
        gpio.pins[10] = 1
        res = gpio.read()
        assert res[10] == 1
        assert gpio._line_requests[10] is req10  # Must not re-create/re-configure
        assert gpio.pin_directions[10] == "output"

        # 2. Writing to pin 11 (input) without reconfigure=True should raise PinDirectionError
        from src.core.exceptions import PinDirectionError
        with pytest.raises(PinDirectionError):
            gpio.write({11: 1})

        # 3. Writing to pin 11 with reconfigure=True should succeed and configure it as output
        gpio.write({11: 1}, reconfigure=True)
        assert gpio.pin_directions[11] == "output"
        assert gpio._line_requests[11] is not req11  # Reconfigured, so it is a new request


