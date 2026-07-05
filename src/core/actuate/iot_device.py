import abc
import logging
import random
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PhysicalDevice(abc.ABC):
    """Abstract base class representing a physical or IoT device adapter."""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self._connected = False

    @abc.abstractmethod
    def connect(self) -> bool:
        """Establish connection to the hardware device."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close connection to the hardware device."""
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
    """Mock GPIO device adapter for pin-level control."""

    def __init__(self, device_id: str, backend: str = "mock"):
        super().__init__(device_id)
        self.pins: Dict[int, int] = {}  # pin_number -> state (0 or 1)

    def connect(self) -> bool:
        self._connected = True
        logger.info(f"Mock GPIODevice '{self.device_id}' connected.")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info(f"Mock GPIODevice '{self.device_id}' disconnected.")

    def read(self) -> Dict[int, int]:
        if not self._connected:
            raise RuntimeError("Device not connected")
        return self.pins

    def _hw_read(self) -> Dict[int, int]:
        """Read physical GPIO line states via libgpiod v2."""
        values = self._request.get_values()
        return {offset: 1 if values[offset] == self._Value.ACTIVE else 0 for offset in self.pins}

    def _hw_set_pin(self, pin: int, state: int) -> None:
        """Drive a single GPIO line via libgpiod v2."""
        self._request.set_value({pin: self._Value.ACTIVE if state else self._Value.INACTIVE})

    def write(self, data: Dict[int, int]) -> bool:
        """Expects data as a dict of {pin_number: state}."""
        if not self._connected:
            raise RuntimeError("Device not connected")
        for pin, state in data.items():
            if state not in (0, 1):
                raise ValueError("State must be 0 or 1")
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
    """Mock RS232/Serial device adapter."""

    def __init__(self, device_id: str, port: str = "/dev/ttyUSB0", baudrate: int = 9600):
        super().__init__(device_id)
        self.port = port
        self.baudrate = baudrate
        self.tx_buffer: str = ""
        self.rx_buffer: str = ""

    def connect(self) -> bool:
        self._connected = True
        logger.info(f"Mock SerialDevice '{self.device_id}' connected on {self.port} at {self.baudrate} baud.")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info(f"Mock SerialDevice '{self.device_id}' disconnected.")

    def read(self) -> str:
        if not self._connected:
            raise RuntimeError("Device not connected")
        data = self.rx_buffer
        self.rx_buffer = ""
        return data

    def write(self, data: str) -> bool:
        if not self._connected:
            raise RuntimeError("Device not connected")
        self.tx_buffer += data
        logger.debug(f"SerialDevice '{self.device_id}' TX: {data}")
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
    """Mock USB HID device adapter."""

    def __init__(self, device_id: str, vendor_id: int = 0x046d, product_id: int = 0xc077):
        super().__init__(device_id)
        self.vendor_id = vendor_id
        self.product_id = product_id

    def connect(self) -> bool:
        self._connected = True
        logger.info(f"Mock USBHIDDevice '{self.device_id}' (VID:{self.vendor_id:#x}, PID:{self.product_id:#x}) connected.")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info(f"Mock USBHIDDevice '{self.device_id}' disconnected.")

    def read(self) -> bytes:
        if not self._connected:
            raise RuntimeError("Device not connected")
        return b"\x00\x01\x02\x03"

    def write(self, data: bytes) -> bool:
        if not self._connected:
            raise RuntimeError("Device not connected")
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
    """Robot Arm device control wrapper using a Serial adapter."""

    def __init__(self, device_id: str, serial_port: str = "/dev/ttyUSB1"):
        self.device_id = device_id
        self.adapter = SerialDevice(f"{device_id}_serial", port=serial_port)
        self.joints: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

    def connect(self) -> bool:
        return self.adapter.connect()

    def disconnect(self) -> None:
        self.adapter.disconnect()

    @property
    def is_connected(self) -> bool:
        return self.adapter.is_connected

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
            # Send physical command over serial mock
            self.adapter.write(f"SET_JOINT_{joint}:{angle}\n")
            return {
                "status": "success",
                "stdout": f"Joint {joint} moved to {angle} degrees.",
                "returncode": 0,
            }
        except Exception as e:
            return {"status": "failed", "stderr": str(e)}


class SensorDevice(PhysicalDevice):
    """Telemetry sensor device returning climate data."""

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.temp_base = 22.0
        self.humi_base = 45.0

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

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
    """Mock IoT Camera device capturing images."""

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.resolution = "1920x1080"

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

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
