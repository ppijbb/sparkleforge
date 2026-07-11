import abc
import logging
import random
from typing import Any, Dict, Optional

from src.core.exceptions import PinDirectionError

logger = logging.getLogger(__name__)

BACKEND_AUTO = "auto"
BACKEND_HARDWARE = "hardware"
BACKEND_MOCK = "mock"


class PhysicalDevice(abc.ABC):
    """Abstract base class representing a physical or IoT device adapter.

    Adapters support three backends selected via ``backend``:
    - "auto" (default): try the real hardware driver, fall back to mock when
      the driver library or the device itself is unavailable
    - "hardware": require the real driver; ``connect`` fails without it
    - "mock": always use the in-memory mock, even when hardware is present

    ``active_backend`` reports which backend a connected device is using.
    """

    def __init__(self, device_id: str, backend: str = BACKEND_AUTO):
        if backend not in (BACKEND_AUTO, BACKEND_HARDWARE, BACKEND_MOCK):
            raise ValueError(f"Unknown backend '{backend}'")
        self.device_id = device_id
        self.backend = backend
        self.active_backend: str | None = None
        self._connected = False

    def connect(self) -> bool:
        """Establish connection to the device, honoring the backend policy."""
        if self.backend in (BACKEND_AUTO, BACKEND_HARDWARE):
            try:
                self._hw_connect()
                self.active_backend = BACKEND_HARDWARE
                self._connected = True
                logger.info(f"{type(self).__name__} '{self.device_id}' connected (hardware backend).")
                return True
            except Exception as e:
                if self.backend == BACKEND_HARDWARE:
                    logger.error(
                        f"{type(self).__name__} '{self.device_id}': hardware backend required "
                        f"but unavailable: {e}"
                    )
                    return False
                logger.info(
                    f"{type(self).__name__} '{self.device_id}': hardware unavailable ({e}); "
                    "falling back to mock backend."
                )
        self.active_backend = BACKEND_MOCK
        self._connected = True
        logger.info(f"{type(self).__name__} '{self.device_id}' connected (mock backend).")
        return True

    def disconnect(self) -> None:
        """Close connection to the device."""
        if self._connected and self.active_backend == BACKEND_HARDWARE:
            try:
                self._hw_disconnect()
            except Exception as e:
                logger.warning(f"{type(self).__name__} '{self.device_id}': error during hardware disconnect: {e}")
        self._connected = False
        self.active_backend = None
        logger.info(f"{type(self).__name__} '{self.device_id}' disconnected.")

    def _hw_connect(self) -> None:
        """Open the real hardware driver. Raise to signal unavailability."""
        raise NotImplementedError(f"{type(self).__name__} has no hardware backend")

    def _hw_disconnect(self) -> None:
        """Release the real hardware driver."""
        pass

    @abc.abstractmethod
    def read(self) -> Any:
        """Read data from the device."""
        pass

    @abc.abstractmethod
    def write(self, data: Any) -> bool:
        """Write data to the device."""
        pass

    @abc.abstractmethod
    def execute_command(self, cmd: str) -> Dict[str, Any]:
        """Send a control command to the device."""
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected


class GPIODevice(PhysicalDevice):
    """GPIO device adapter for pin-level control.

    Hardware backend drives lines through libgpiod v2 (``gpiod``
    package) on the given character device; mock backend keeps pin states
    in memory only.
    """

    def __init__(self, device_id: str, backend: str = BACKEND_AUTO, chip_path: str = "/dev/gpiochip0"):
        super().__init__(device_id, backend)
        self.chip_path = chip_path
        self.pins: Dict[int, int] = {}  # pin_number -> last driven or read state (0 or 1)
        self.pin_directions: Dict[int, str] = {}  # pin_number -> "input" or "output"
        self.pin_biases: Dict[int, str] = {}  # pin -> "pull_up", "pull_down", "disabled", "as_is"
        self.pin_active_lows: Dict[int, bool] = {}  # pin -> True/False
        self._gpiod = None
        self._chip = None
        self._line_requests: Dict[int, Any] = {}

    def _hw_connect(self) -> None:
        import gpiod
        self._gpiod = gpiod
        self._chip = gpiod.Chip(self.chip_path)

    def _hw_disconnect(self) -> None:
        for pin, request in list(self._line_requests.items()):
            try:
                request.release()
            except Exception as e:
                logger.warning(f"Failed to release pin {pin} during disconnect: {e}")
        self._line_requests.clear()
        if self._chip is not None:
            self._chip.close()
            self._chip = None

    def _hw_configure_pin(self, pin: int, direction: str, bias: str = "as_is", active_low: bool = False) -> None:
        from gpiod.line import Bias, Direction
        # Release if already requested to apply new configurations
        if pin in self._line_requests:
            try:
                self._line_requests[pin].release()
            except Exception as e:
                logger.warning(f"Failed to release pin {pin} during reconfiguration: {e}")
            del self._line_requests[pin]

        dir_val = Direction.INPUT if direction == "input" else Direction.OUTPUT
        bias_map = {
            "pull_up": Bias.PULL_UP,
            "pull_down": Bias.PULL_DOWN,
            "disabled": Bias.DISABLED,
            "as_is": Bias.AS_IS
        }
        bias_val = bias_map.get(bias, Bias.AS_IS)

        request = self._chip.request_lines(
            consumer=self.device_id,
            config={
                pin: self._gpiod.LineSettings(
                    direction=dir_val,
                    bias=bias_val,
                    active_low=active_low
                )
            }
        )
        self._line_requests[pin] = request

    def _hw_set_pin(self, pin: int, state: int, reconfigure: bool = False) -> None:
        from gpiod.line import Value
        request = self._line_requests.get(pin)
        direction = self.pin_directions.get(pin, "output")
        # Re-configure as output if not already output
        if direction == "input":
            if not reconfigure:
                raise PinDirectionError(pin, expected_direction="output", actual_direction="input")
            self.pin_directions[pin] = "output"
            self._hw_configure_pin(
                pin,
                direction="output",
                bias=self.pin_biases.get(pin, "as_is"),
                active_low=self.pin_active_lows.get(pin, False)
            )
            request = self._line_requests[pin]
        elif request is None:
            self.pin_directions[pin] = "output"
            self._hw_configure_pin(
                pin,
                direction="output",
                bias=self.pin_biases.get(pin, "as_is"),
                active_low=self.pin_active_lows.get(pin, False)
            )
            request = self._line_requests[pin]
        request.set_value(pin, Value.ACTIVE if state else Value.INACTIVE)

    def _hw_read_pin(self, pin: int) -> int:
        from gpiod.line import Value
        direction = self.pin_directions.get(pin, "input")
        if direction == "output":
            raise PinDirectionError(pin, expected_direction="input", actual_direction="output")
        request = self._line_requests.get(pin)
        # _hw_read_pin must not call _hw_configure_pin.
        if request is None:
            raise PinDirectionError(pin, expected_direction="input", actual_direction="unconfigured")
        return 1 if request.get_value(pin) == Value.ACTIVE else 0

    def read(self) -> Dict[int, int]:
        """Return current pin states.

        For pins previously driven or configured through the hardware backend, this
        queries the physical line rather than trusting the local write cache.
        """
        if not self._connected:
            raise RuntimeError("Device not connected")
        if self.active_backend == BACKEND_HARDWARE:
            for pin in list(self._line_requests.keys()):
                direction = self.pin_directions.get(pin, "input")
                if direction == "output":
                    pass
                else:
                    self.pins[pin] = self._hw_read_pin(pin)
        return dict(self.pins)

    def write(self, data: Dict[int, int], reconfigure: bool = False) -> bool:
        """Expects data as a dict of {pin_number: state}."""
        if not self._connected:
            raise RuntimeError("Device not connected")
        for pin, state in data.items():
            if state not in (0, 1):
                raise ValueError("State must be 0 or 1")
            if self.active_backend == BACKEND_HARDWARE:
                self._hw_set_pin(pin, state, reconfigure=reconfigure)
            self.pins[pin] = state
            logger.debug(f"GPIODevice '{self.device_id}': Pin {pin} set to {state}")
        return True

    def execute_command(self, cmd: str) -> Dict[str, Any]:
        """Expects format:
        - 'set_pin <pin> <0|1>'
        - 'get_pin <pin>' / 'read_pin <pin>'
        - 'configure_pin <pin> <input|output> [pull_up|pull_down|disabled|as_is] [active_low|active_high]'
        - 'mock_input <pin> <0|1>'
        """
        if not self._connected:
            return {"status": "failed", "stderr": "Device not connected"}

        parts = cmd.strip().split()
        if not parts:
            return {"status": "failed", "stderr": "Empty command"}

        action = parts[0].lower()
        try:
            if action == "set_pin":
                pin = int(parts[1])
                val = int(parts[2])
                self.write({pin: val})
                return {"status": "success", "stdout": f"Pin {pin} set to {val}"}
            elif action in ("get_pin", "read_pin"):
                pin = int(parts[1])
                if self.active_backend == BACKEND_HARDWARE:
                    val = self._hw_read_pin(pin)
                else:
                    # In mock mode, check if we have a simulated state, default to 0
                    val = self.pins.get(pin, 0)
                self.pins[pin] = val
                return {"status": "success", "stdout": str(val), "returncode": 0}
            elif action == "configure_pin":
                pin = int(parts[1])
                direction = parts[2].lower()
                if direction not in ("input", "output"):
                    raise ValueError(f"Direction must be 'input' or 'output', got '{direction}'")
                
                bias = "as_is"
                if len(parts) > 3:
                    bias = parts[3].lower()
                    if bias not in ("pull_up", "pull_down", "disabled", "as_is"):
                        raise ValueError(f"Invalid bias: {bias}")
                
                active_low = False
                if len(parts) > 4:
                    active_low_str = parts[4].lower()
                    if active_low_str in ("active_low", "true", "1"):
                        active_low = True
                    elif active_low_str in ("active_high", "false", "0"):
                        active_low = False
                    else:
                        raise ValueError(f"Invalid active-low configuration: {active_low_str}")

                self.pin_directions[pin] = direction
                self.pin_biases[pin] = bias
                self.pin_active_lows[pin] = active_low

                if self.active_backend == BACKEND_HARDWARE:
                    self._hw_configure_pin(pin, direction, bias, active_low)
                
                return {"status": "success", "stdout": f"Pin {pin} configured as {direction} ({bias}, active_low={active_low})"}
            elif action == "mock_input":
                if self.active_backend != BACKEND_MOCK:
                    return {"status": "failed", "stderr": "mock_input is only supported on mock backend"}
                pin = int(parts[1])
                val = int(parts[2])
                if val not in (0, 1):
                    raise ValueError("Value must be 0 or 1")
                self.pins[pin] = val
                return {"status": "success", "stdout": f"Mock input for pin {pin} set to {val}"}
            else:
                return {"status": "failed", "stderr": f"Unknown action: {action}"}
        except Exception as e:
            return {"status": "failed", "stderr": str(e)}


class SerialDevice(PhysicalDevice):
    """RS232/Serial device adapter.

    Hardware backend speaks to the port through pyserial; mock backend
    simulates a device echoing canned responses.
    """

    def __init__(
        self,
        device_id: str,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 9600,
        timeout: float = 2.0,
        *,
        backend: str = BACKEND_AUTO,
    ):
        super().__init__(device_id, backend)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.tx_buffer: str = ""
        self.rx_buffer: str = ""
        self._serial = None

    def _hw_connect(self) -> None:
        import serial
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    def _hw_disconnect(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def read(self) -> str:
        if not self._connected:
            raise RuntimeError("Device not connected")
        if self.active_backend == BACKEND_HARDWARE:
            return self._serial.readline().decode("utf-8", errors="replace")
        data = self.rx_buffer
        self.rx_buffer = ""
        return data

    def write(self, data: str) -> bool:
        if not self._connected:
            raise RuntimeError("Device not connected")
        self.tx_buffer += data
        logger.debug(f"SerialDevice '{self.device_id}' TX: {data}")
        if self.active_backend == BACKEND_HARDWARE:
            self._serial.write(data.encode("utf-8"))
        else:
            # Simulated echo or automatic parser
            self._simulate_rx_response(data)
        return True

    def _simulate_rx_response(self, data: str) -> None:
        # Mocking responses
        clean = data.strip().upper()
        if clean == "GET_TEMP":
            self.rx_buffer += "TEMP:22.5C\n"
        elif clean == "PING":
            self.rx_buffer += "PONG\n"
        else:
            self.rx_buffer += "ERR_UNKNOWN_CMD\n"

    def execute_command(self, cmd: str) -> Dict[str, Any]:
        if not self._connected:
            return {"status": "failed", "stderr": "Device not connected"}
        self.write(cmd + "\n")
        response = self.read().strip()
        return {"status": "success", "stdout": response, "returncode": 0}


class USBHIDDevice(PhysicalDevice):
    """USB HID device adapter.

    Hardware backend uses the hidapi bindings (``hid`` package); mock backend
    returns canned reports.
    """

    def __init__(
        self,
        device_id: str,
        vendor_id: int = 0x046d,
        product_id: int = 0xc077,
        read_size: int = 64,
        read_timeout_ms: int = 1000,
        *,
        backend: str = BACKEND_AUTO,
    ):
        super().__init__(device_id, backend)
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.read_size = read_size
        self.read_timeout_ms = read_timeout_ms
        self._hid = None

    def _hw_connect(self) -> None:
        import hid
        self._hid = hid.device()
        self._hid.open(self.vendor_id, self.product_id)

    def _hw_disconnect(self) -> None:
        if self._hid is not None:
            self._hid.close()
            self._hid = None

    def read(self) -> bytes:
        if not self._connected:
            raise RuntimeError("Device not connected")
        if self.active_backend == BACKEND_HARDWARE:
            return bytes(self._hid.read(self.read_size, self.read_timeout_ms))
        return b"\x00\x01\x02\x03"

    def write(self, data: bytes) -> bool:
        if not self._connected:
            raise RuntimeError("Device not connected")
        if self.active_backend == BACKEND_HARDWARE:
            # hidapi's write() returns bytes actually written (or a negative
            # value on failure); trusting a fixed True here would mask a
            # partial or failed write from the caller.
            written = self._hid.write(list(data))
            logger.debug(f"USBHIDDevice '{self.device_id}' wrote: {data.hex()}")
            return written == len(data)
        logger.debug(f"USBHIDDevice '{self.device_id}' wrote: {data.hex()}")
        return True

    def execute_command(self, cmd: str) -> Dict[str, Any]:
        if not self._connected:
            return {"status": "failed", "stderr": "Device not connected"}
        # Echo back bytes as hex string
        self.write(cmd.encode())
        return {"status": "success", "stdout": cmd.encode().hex(), "returncode": 0}


# --- IoT Device Implementations (using Adapters) ---

class RobotArmDevice:
    """Robot Arm device control wrapper using a Serial adapter.

    The joint protocol (``SET_JOINT_<id>:<angle>``) is spoken over the serial
    adapter, so the arm follows the adapter's backend: real serial hardware
    when available, in-memory mock otherwise.
    """

    def __init__(self, device_id: str, serial_port: str = "/dev/ttyUSB1", backend: str = BACKEND_AUTO):
        self.device_id = device_id
        self.adapter = SerialDevice(f"{device_id}_serial", port=serial_port, backend=backend)
        self.joints: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

    def connect(self) -> bool:
        return self.adapter.connect()

    def disconnect(self) -> None:
        self.adapter.disconnect()

    @property
    def is_connected(self) -> bool:
        return self.adapter.is_connected

    @property
    def active_backend(self) -> str | None:
        return self.adapter.active_backend

    def execute_command(self, cmd: str) -> Dict[str, Any]:
        """Expects format: 'move_joint <joint_id> <angle>'."""
        if not self.is_connected:
            return {"status": "failed", "stderr": "Robot arm not connected"}

        parts = cmd.strip().split()
        if len(parts) < 3 or parts[0].lower() != "move_joint":
            return {"status": "failed", "stderr": "Invalid format. Use: 'move_joint <joint_id> <angle>'"}

        try:
            joint = int(parts[1])
            angle = int(parts[2])
            if joint not in self.joints:
                return {"status": "failed", "stderr": f"Invalid joint ID: {joint}"}
            if not (-180 <= angle <= 180):
                return {"status": "failed", "stderr": "Angle must be between -180 and 180"}

            self.joints[joint] = angle
            # Send physical command over the serial adapter
            self.adapter.write(f"SET_JOINT_{joint}:{angle}\n")
            return {
                "status": "success",
                "stdout": f"Joint {joint} moved to {angle} degrees.",
                "returncode": 0,
            }
        except Exception as e:
            return {"status": "failed", "stderr": str(e)}


class SensorDevice(PhysicalDevice):
    """Telemetry sensor device returning climate data.

    Hardware backend reads temperature/humidity from a DHT11/DHT22 sensor
    through the ``adafruit_dht`` library on the configured GPIO pin; mock
    backend simulates slight variations around a baseline climate.
    """

    def __init__(self, device_id: str, backend: str = BACKEND_AUTO, pin: int = 4):
        super().__init__(device_id, backend)
        self.pin = pin
        self.temp_base = 22.0
        self.humi_base = 45.0
        self._dht = None

    def _hw_connect(self) -> None:
        import adafruit_dht
        self._dht = adafruit_dht.DHT22(self.pin)

    def _hw_disconnect(self) -> None:
        if self._dht is not None:
            try:
                self._dht.exit()
            except Exception:
                pass
            self._dht = None

    def read(self) -> Dict[str, float]:
        if not self._connected:
            raise RuntimeError("Sensor not connected")
        if self.active_backend == BACKEND_HARDWARE:
            # DHT sensors occasionally return NaN/spurious readings; retry a
            # few times before giving up so callers get real telemetry.
            for _ in range(5):
                try:
                    temp = float(self._dht.temperature)
                    humi = float(self._dht.humidity)
                    if temp == temp and humi == humi:  # reject NaN
                        return {"temperature": round(temp, 1), "humidity": round(humi, 1)}
                except Exception:
                    continue
            raise RuntimeError("DHT sensor read failed after retries")
        # Simulate slight variations
        temp = round(self.temp_base + random.uniform(-1.0, 1.0), 1)
        humi = round(self.humi_base + random.uniform(-3.0, 3.0), 1)
        return {"temperature": temp, "humidity": humi}

    def write(self, data: Any) -> bool:
        return False

    def execute_command(self, cmd: str) -> Dict[str, Any]:
        if not self._connected:
            return {"status": "failed", "stderr": "Sensor not connected"}
        if cmd.strip().lower() == "read_data":
            data = self.read()
            return {
                "status": "success",
                "stdout": f"Temp: {data['temperature']}C, Humi: {data['humidity']}%",
                "returncode": 0,
            }
        return {"status": "failed", "stderr": f"Unknown sensor command: {cmd}"}


class CameraDevice(PhysicalDevice):
    """IoT Camera device capturing images.

    Hardware backend grabs frames from a connected USB/CSI camera through
    ``opencv-python`` (``cv2.VideoCapture``); mock backend returns a canned
    JPEG header.
    """

    def __init__(self, device_id: str, backend: str = BACKEND_AUTO, source: int = 0):
        super().__init__(device_id, backend)
        self.source = source
        self.resolution = "1920x1080"
        self._cap = None

    def _hw_connect(self) -> None:
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera source {self.source}")

    def _hw_disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> bytes:
        if not self._connected:
            raise RuntimeError("Camera not connected")
        if self.active_backend == BACKEND_HARDWARE:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera frame capture failed")
            ok, buf = self._cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError("Camera frame encoding failed")
            return bytes(buf)
        # Return mock JPEG header
        return b"\xff\xd8\xff\xe0\x00\x10JFIF"

    def write(self, data: Any) -> bool:
        return False

    def execute_command(self, cmd: str) -> Dict[str, Any]:
        if not self._connected:
            return {"status": "failed", "stderr": "Camera not connected"}

        parts = cmd.strip().split()
        action = parts[0].lower() if parts else ""
        if action == "capture":
            img = self.read()
            return {
                "status": "success",
                "stdout": f"Captured frame: {img.hex()[:20]}...",
                "returncode": 0,
            }
        elif action == "set_resolution":
            if len(parts) < 2:
                return {"status": "failed", "stderr": "Resolution not specified"}
            self.resolution = parts[1]
            return {"status": "success", "stdout": f"Resolution set to {self.resolution}"}
        return {"status": "failed", "stderr": f"Unknown camera command: {cmd}"}
