import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PackageController:
    """
    Manages the lifecycle and interoperability of microservices within the SparkleForge ecosystem.
    Automatically links newly forged services into the composition graph.
    """
    def __init__(self):
        self.service_registry: Dict[str, Any] = {}
        self.composition_graph: List[Dict[str, Any]] = []

    def register_service(self, service_id: str, metadata: Dict[str, Any]):
        """Registers a new service and triggers auto-linking."""
        self.service_registry[service_id] = metadata
        self._link_service(service_id)
        logger.info(f"Service {service_id} registered and linked.")

    def _link_service(self, service_id: str):
        """Analyzes service capabilities and updates the composition graph."""
        metadata = self.service_registry.get(service_id, {})
        dependencies = metadata.get("dependencies", [])
        
        for dep in dependencies:
            if dep in self.service_registry:
                link = {
                    "source": service_id,
                    "target": dep,
                    "type": "composition"
                }
                self.composition_graph.append(link)
                logger.debug(f"Linked {service_id} to {dep}")

    def get_ecosystem_map(self) -> List[Dict[str, Any]]:
        """Returns the current state of the service composition graph."""
        return self.composition_graph
