import aiohttp
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FederationClient:
    """Lightweight HTTP protocol for peer-to-peer SparkleForge node communication."""
    
    def __init__(self, peers: List[str] = None):
        self.peers = peers or []

    async def distribute_tasks(self, query: str) -> List[Dict[str, Any]]:
        """Partition research vectors and assign sub-tasks to federated peers."""
        tasks = []
        for peer in self.peers:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{peer}/v1/research/subtask", json={"query": query}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            tasks.append(data)
            except Exception as e:
                logger.error(f"Failed to federate task to {peer}: {e}")
        return tasks

    async def aggregate_results(self, results: List[str]) -> str:
        return "\n---\n".join(results)
