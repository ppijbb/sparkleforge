import asyncio
import datetime
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BLACKLIST_DOMAINS = [
    "malicious-target.com",
    "blocked-adware-domain.net",
    "leakage-channel.org",
]


class AuditProxy:
    """Network requests proxy with built-in destination filtering, cost caching, and audit logging."""

    def __init__(
        self, 
        audit_log_path: str = "data/network_audit.log", 
        blacklist: Optional[List[str]] = None
    ):
        self.audit_log_path = audit_log_path
        self.blacklist = blacklist if blacklist is not None else BLACKLIST_DOMAINS
        self._cache: Dict[str, Any] = {}

    def _is_allowed(self, url: str) -> bool:
        """Check if URL host is in the blacklist."""
        try:
            parsed = urllib.parse.urlparse(url)
            host = parsed.hostname or ""
            for domain in self.blacklist:
                if domain.lower() in host.lower():
                    return False
        except Exception:
            pass
        return True

    def _write_audit_log(self, method: str, url: str, headers: Dict, request_size: int, status_code: int):
        """Append a structured log to the network audit log file."""
        try:
            log_dir = os.path.dirname(self.audit_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            log_line = (
                f"[{timestamp}] {method} {url} | Headers: {list(headers.keys()) if headers else []} | "
                f"ReqSize: {request_size} bytes | Status: {status_code}\n"
            )
            with open(self.audit_log_path, "a") as f:
                f.write(log_line)
        except Exception as e:
            logger.error(f"AuditProxy: Failed to write audit log: {e}")

    async def request(
        self, 
        method: str, 
        url: str, 
        headers: Optional[Dict[str, str]] = None, 
        data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """Perform a proxy network request with security audit and caching constraints."""
        method_upper = method.upper()
        headers = headers or {}
        req_size = len(data) if data else 0

        # Enforce destination blacklist
        if not self._is_allowed(url):
            logger.warning(f"AuditProxy: Blocked request to: {url}")
            self._write_audit_log(method_upper, url, headers, req_size, 403)
            return {
                "status": 403,
                "body": "",
                "headers": {},
                "error": "Request blocked by AuditProxy blacklist policy."
            }

        # Check Cache for GET requests
        if method_upper == "GET" and url in self._cache:
            logger.debug(f"AuditProxy: Cache hit for GET {url}")
            # Caching requests do not re-audit/re-fetch but we can record it
            return self._cache[url]

        # Async Wrapper for blocking urllib calls
        def _perform_http():
            req = urllib.request.Request(url, data=data, method=method_upper)
            for k, v in headers.items():
                req.add_header(k, v)
                
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    body = response.read().decode(errors="replace")
                    resp_headers = dict(response.info())
                    return {
                        "status": response.status,
                        "body": body,
                        "headers": resp_headers,
                        "error": None
                    }
            except urllib.error.HTTPError as he:
                return {
                    "status": he.code,
                    "body": he.read().decode(errors="replace") if he.fp else "",
                    "headers": dict(he.headers) if he.headers else {},
                    "error": str(he)
                }
            except Exception as e:
                return {
                    "status": -1,
                    "body": "",
                    "headers": {},
                    "error": str(e)
                }

        logger.info(f"AuditProxy: Sending request: {method_upper} {url}")
        res = await asyncio.to_thread(_perform_http)
        
        status_code = res.get("status", -1)
        self._write_audit_log(method_upper, url, headers, req_size, status_code)

        # Cache successful GET responses
        if method_upper == "GET" and status_code >= 200 and status_code < 300:
            self._cache[url] = res

        return res

    def clear_cache(self):
        """Reset the GET request cache."""
        self._cache.clear()
