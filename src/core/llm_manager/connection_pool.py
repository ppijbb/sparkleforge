"""LLM connection pooling for performance optimization.

Split out of the former monolithic llm_manager.py (issue #582).
"""

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class ConnectionPool:
    """LLM Connection Pool for performance optimization."""

    def __init__(self, pool_size: int = 5, max_idle_time: int = 300):
        """Initialize connection pool.

        Args:
            pool_size: Maximum number of connections per model
            max_idle_time: Maximum idle time before connection cleanup (seconds)
        """
        self.pool_size = pool_size
        self.max_idle_time = max_idle_time
        self.pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.connection_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"created": 0, "reused": 0}
        )
        self.last_cleanup = time.time()

    def get_connection(self, model_name: str, provider: str, create_func: Callable) -> Any:
        """Get or create a connection from the pool.

        Args:
            model_name: Model name
            provider: Provider name
            create_func: Function to create new connection

        Returns:
            Connection object
        """
        pool_key = f"{provider}:{model_name}"
        current_time = time.time()

        # Cleanup old connections periodically
        if current_time - self.last_cleanup > 60:  # Every minute
            self._cleanup_old_connections()
            self.last_cleanup = current_time

        # Check for available connection
        if pool_key in self.pools and self.pools[pool_key]:
            conn_info = self.pools[pool_key].pop()
            if current_time - conn_info["created_at"] < self.max_idle_time:
                self.connection_stats[pool_key]["reused"] += 1
                logger.debug(f"Reusing connection for {pool_key}")
                return conn_info["connection"]

        # Create new connection
        connection = create_func()
        self.connection_stats[pool_key]["created"] += 1
        logger.debug(f"Created new connection for {pool_key}")

        return connection

    def return_connection(self, model_name: str, provider: str, connection: Any) -> None:
        """Return a connection to the pool.

        Args:
            model_name: Model name
            provider: Provider name
            connection: Connection object to return
        """
        pool_key = f"{provider}:{model_name}"

        if len(self.pools[pool_key]) < self.pool_size:
            self.pools[pool_key].append({"connection": connection, "created_at": time.time()})
        # If pool is full, let connection be garbage collected

    def _cleanup_old_connections(self) -> None:
        """Remove old idle connections from all pools."""
        current_time = time.time()
        removed_count = 0

        for pool_key, connections in list(self.pools.items()):
            valid_connections = []
            for conn_info in connections:
                if current_time - conn_info["created_at"] < self.max_idle_time:
                    valid_connections.append(conn_info)
                else:
                    removed_count += 1

            if valid_connections:
                self.pools[pool_key] = valid_connections
            else:
                del self.pools[pool_key]

        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} old connections")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "pool_size": self.pool_size,
            "active_pools": len(self.pools),
            "total_connections": sum(len(conns) for conns in self.pools.values()),
            "connection_stats": dict(self.connection_stats),
        }


