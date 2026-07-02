import logging
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
        
        self._initialized = True
        logger.info("✅ Actuation Plane successfully initialized")
