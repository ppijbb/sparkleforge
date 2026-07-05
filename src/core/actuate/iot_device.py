import abc
import logging
import random
from typing import Any, Dict, Optional

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
        self.active_backend: Optional[str] = None
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

    Hardware backend drives output lines through libgpiod v2 (``gpiod``
    package) on the given character device; mock backend keeps pin states
    in memory only.
    """

    def __init__(self, device_id: str, backend: str = BACKEND_AUTO, chip_path: str = "/dev/gpiochip0"):
        super().__init__(device_id, backend)
        self.chip_path = chip_path
        self.pins: Dict[int, int] = {}  # pin_number -> last driven state (0 or 1)
        self._gpiod = None
        self._chip = None
        self._line_requests: Dict[int, Any] = {}

    def _hw_connect(self) -> None:
        import gpiod
        self._gpiod = gpiod
        self._chip = gpiod.Chip(self.chip_path)

    def _hw_disconnect(self) -> None:
        for request in self._line_requests.values():
            try:
                request.release()
            except Exception:
                pass
        self._line_requests.clear()
        if self._chip is not None:
            self._chip.close()
            self._chip = None

    def _hw_set_pin(self, pin: int, state: int) -> None:
        from gpiod.line import Direction, Value
        request = self._line_requests.get(pin)
        if request is None:
            request = self._chip.request_lines(
                consumer=self.device_id,
                config={pin: self._gpiod.LineSettings(direction=Direction.OUTPUT)},
            )
            self._line_requests[pin] = request
        request.set_value(pin, Value.ACTIVE if state else Value.INACTIVE)

    def _hw_read_pin(self, pin: int) -> int:
        from gpiod.line import Value
        request = self._line_requests[pin]
        return 1 if request.get_value(pin) == Value.ACTIVE else 0

    def read(self) -> Dict[int, int]:
        """Return current pin states.

        For pins previously driven through the hardware backend, this queries
        the physical line rather than trusting the local write cache, so a
        line forced to a different state outside this process (or a failed
        write) is reflected. Pins never requested as lines fall back to the
        cache (this adapter only requests output lines).
        """
        if not self._connected:
            raise RuntimeError("Device not connected")
        if self.active_backend == BACKEND_HARDWARE:
            for pin in self._line_requests:
                self.pins[pin] = self._hw_read_pin(pin)
        return dict(self.pins)

    def write(self, data: Dict[int, int]) -> bool:
        """Expects data as a dict of {pin_number: state}."""
        if not self._connected:
            raise RuntimeError("Device not connected")
        for pin, state in data.items():
            if state not in (0, 1):
                raise ValueError("State must be 0 or 1")
            if self.active_backend == BACKEND_HARDWARE:
                self._hw_set_pin(pin, state)
            self.pins[pin] = state
            logger.debug(f"GPIODevice '{self.device_id}': Pin {pin} set to {state}")
        return True

    def execute_command(self, cmd: str) -> Dict[str, Any]:
        """Expects format: 'set_pin <pin> <0|1>' or 'get_pin <pin>'."""
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
            elif action == "get_pin":
                pin = int(parts[1])
                val = self.pins.get(pin, 0)
                return {"status": "success", "stdout": str(val), "returncode": 0}
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
    def active_backend(self) -> Optional[str]:
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
    """Telemetry sensor device returning climate data (mock backend only)."""

    def __init__(self, device_id: str, backend: str = BACKEND_MOCK):
        super().__init__(device_id, backend)
        self.temp_base = 22.0
        self.humi_base = 45.0

    def read(self) -> Dict[str, float]:
        if not self._connected:
            raise RuntimeError("Sensor not connected")
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
    """IoT Camera device capturing images (mock backend only)."""

    def __init__(self, device_id: str, backend: str = BACKEND_MOCK):
        super().__init__(device_id, backend)
        self.resolution = "1920x1080"

    def read(self) -> bytes:
        if not self._connected:
            raise RuntimeError("Camera not connected")
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
