import logging
from typing import Any, Dict
from src.core.actuate.shell_executor import SecureShellExecutor
from src.core.actuate.os_control import OSControl
from src.core.actuate.semantic_fs import SemanticFS
from src.core.actuate.package_manager import UnifiedPackageManager
from src.core.actuate.audit_proxy import AuditProxy
from src.core.actuate.firmware_assistant import FirmwareAssistant

logger = logging.getLogger(__name__)


class ActuationPlane:
    """Unified Orchestrator for all system interaction, automation triggers, and actuation plane pipelines."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        logger.info("Initializing Actuation Plane...")
        self.shell = SecureShellExecutor()
        self.os = OSControl()
        self.fs = SemanticFS()
        self.packages = UnifiedPackageManager()
        self.proxy = AuditProxy()
        self.firmware = FirmwareAssistant()
        
        self.iot_devices: Dict[str, Any] = {}
        
        self._initialized = True
        logger.info("✅ Actuation Plane successfully initialized")

    def register_device(self, device_id: str, device: Any) -> None:
        """Register a physical or IoT device adapter."""
        self.iot_devices[device_id] = device
        logger.info(f"ActuationPlane: Registered IoT device '{device_id}'")

    def control_device(self, device_id: str, command: str) -> Dict[str, Any]:
        """Control a registered physical or IoT device."""
        if device_id not in self.iot_devices:
            return {"status": "failed", "stderr": f"Device '{device_id}' not registered"}
        device = self.iot_devices[device_id]
        try:
            return device.execute_command(command)
        except Exception as e:
            return {"status": "failed", "stderr": str(e)}

    def read_device(self, device_id: str, command: str) -> Dict[str, Any]:
        """Read state from a registered device (using execution callback or read method)."""
        if device_id not in self.iot_devices:
            return {"status": "failed", "stderr": f"Device '{device_id}' not registered"}
        device = self.iot_devices[device_id]
        try:
            if command:
                return device.execute_command(command)
            data = device.read()
            return {"status": "success", "stdout": str(data), "returncode": 0}
        except Exception as e:
            return {"status": "failed", "stderr": str(e)}
