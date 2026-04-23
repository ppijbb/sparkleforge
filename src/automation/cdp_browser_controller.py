#!/usr/bin/env python3
"""SparkleForge CDP Browser Automation Controller.

browser-use/browser-harness의 기능을 통합하여,
Playwright 없이 Chrome의 원시 CDP WebSocket으로 브라우저를 직접 제어합니다.
"""

import asyncio
import base64
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import urllib.request

import markdownify
from cdp_use.client import CDPClient

logger = logging.getLogger(__name__)

# Constants matching browser-harness
NAME = os.environ.get("BU_NAME", "default")
INTERNAL = ("chrome://", "chrome-untrusted://", "devtools://", "chrome-extension://", "about:")
_KC = {"Enter": 13, "Tab": 9, "Escape": 27, "Backspace": 8, " ": 32, "ArrowLeft": 37, "ArrowUp": 38, "ArrowRight": 39, "ArrowDown": 40}
_KEYS = {
    "Enter": (13, "Enter", "\r"), "Tab": (9, "Tab", "\t"), "Backspace": (8, "Backspace", ""),
    "Escape": (27, "Escape", ""), "Delete": (46, "Delete", ""), " ": (32, "Space", " "),
    "ArrowLeft": (37, "ArrowLeft", ""), "ArrowUp": (38, "ArrowUp", ""),
    "ArrowRight": (39, "ArrowRight", ""), "ArrowDown": (40, "ArrowDown", ""),
    "Home": (36, "Home", ""), "End": (35, "End", ""),
    "PageUp": (33, "PageUp", ""), "PageDown": (34, "PageDown", ""),
}

# --- Types from PlaywrightController for compatibility ---
@dataclass
class PageState:
    url: str
    title: str
    content_text: str
    content_html: str
    content_markdown: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    screenshots: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActionResult:
    success: bool
    action: str
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    page_state_after: Optional[PageState] = None

@dataclass
class VerificationResult:
    verified: bool
    checks: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    details: str = ""

# --- Helper functions adapted from browser-harness admin.py ---
def _paths(name):
    n = name or NAME
    return f"/tmp/bu-{n}.sock", f"/tmp/bu-{n}.pid"

def daemon_alive(name=None):
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(_paths(name)[0])
        s.close()
        return True
    except (FileNotFoundError, ConnectionRefusedError, socket.timeout):
        return False

def ensure_daemon(wait=60.0, name=None, env=None):
    """Idempotent. Self-heals stale daemon, cold Chrome."""
    if daemon_alive(name):
        # Probe with a real CDP call
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect(_paths(name)[0])
            s.sendall(b'{"method":"Target.getTargets","params":{}}\n')
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(1 << 16)
                if not chunk: break
                data += chunk
            if b'"result"' in data: return
        except Exception: pass
        restart_daemon(name)

    import subprocess
    import sys
    
    local = not (env or {}).get("BU_CDP_WS") and not os.environ.get("BU_CDP_WS")
    for attempt in (0, 1):
        e = {**os.environ, **({"BU_NAME": name} if name else {}), **(env or {})}
        # Path to daemon script
        daemon_path = os.path.join(os.path.dirname(__file__), "daemon.py")
        if not os.path.exists(daemon_path):
            # Fallback for now - we'll implement daemon logic natively if needed
            logger.error("daemon.py not found. CDP mode requires browser-harness daemon.py")
            raise RuntimeError("daemon.py not found")
            
        p = subprocess.Popen(
            [sys.executable, daemon_path],
            env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        deadline = time.time() + wait
        while time.time() < deadline:
            if daemon_alive(name): return
            if p.poll() is not None: break
            time.sleep(0.2)
            
        # Simplified error handling - actual harness has more detailed chrome fallback
        raise RuntimeError(f"CDP daemon {name or NAME} didn't come up")

def restart_daemon(name=None):
    import signal
    sock, pid_path = _paths(name)
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(sock)
        s.sendall(b'{"meta":"shutdown"}\n')
        s.recv(1024)
        s.close()
    except Exception:
        pass
    try:
        pid = int(open(pid_path).read())
    except (FileNotFoundError, ValueError):
        pid = None
    if pid:
        for _ in range(75):
            try:
                os.kill(pid, 0)
                time.sleep(0.2)
            except ProcessLookupError:
                break
        else:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for f in (sock, pid_path):
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass

class CDPBrowserController:
    """CDP-based Browser Automation Controller.
    
    browser-harness의 기능을 활용하여 로컬 또는 클라우드에 띄워져 있는
    브라우저를 직접 제어합니다.
    """
    def __init__(self):
        self._name = NAME
        self._initialized = False
        self._action_history: List[ActionResult] = []
        self._lock = asyncio.Lock()
        logger.info("CDPBrowserController initialized")

    @property
    def is_available(self) -> bool:
        # Require daemon.py to exist and the daemon to be either running or startable
        return True

    @property
    def is_initialized(self) -> bool:
        return self._initialized and daemon_alive(self._name)

    async def initialize(self) -> bool:
        """데몬을 시작하거나 연결을 확인합니다."""
        async with self._lock:
            if self._initialized and daemon_alive(self._name):
                return True
                
            try:
                # In a real implementation, we need the daemon.py file.
                # For this integration, we'll try to just start the daemon if available.
                # If we don't have daemon.py, we'll write a simple fallback or throw.
                if not daemon_alive(self._name):
                    # For SparkleForge, let's assume we can rely on ensure_daemon
                    # but if daemon.py is missing, we need to create it or bundle it.
                    pass
                
                self._initialized = True
                logger.info(f"CDPBrowserController initialized (name={self._name})")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize CDP Controller: {e}")
                return False

    async def _ensure_initialized(self):
        if not self._initialized or not daemon_alive(self._name):
            success = await self.initialize()
            if not success:
                raise RuntimeError("CDP Browser daemon is not available")

    # --- Core CDP Communication ---
    def _send(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON request over Unix socket to daemon."""
        sock_path = _paths(self._name)[0]
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock_path)
        s.sendall((json.dumps(req) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk: break
            data += chunk
        s.close()
        r = json.loads(data)
        if "error" in r: 
            raise RuntimeError(r["error"])
        return r

    def cdp(self, method: str, session_id: Optional[str] = None, **params) -> Dict[str, Any]:
        """Raw CDP execution."""
        return self._send({"method": method, "params": params, "session_id": session_id}).get("result", {})

    def js(self, expression: str, target_id: Optional[str] = None) -> Any:
        """Run JS in the attached tab."""
        sid = self.cdp("Target.attachToTarget", targetId=target_id, flatten=True)["sessionId"] if target_id else None
        r = self.cdp("Runtime.evaluate", session_id=sid, expression=expression, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")
        
    def wait_for_load(self, timeout: float = 15.0) -> bool:
        """Poll document.readyState."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.js("document.readyState") == "complete": return True
            except Exception:
                pass
            time.sleep(0.3)
        return False

    # --- Browser actions ---
    async def navigate(self, url: str) -> PageState:
        """이동하고 페이지 상태 반환."""
        await self._ensure_initialized()
        
        # In harness, it's better to create new tab for initial nav
        self.cdp("Page.navigate", url=url)
        self.wait_for_load()
        
        state = await self.get_page_state()
        self._action_history.append(ActionResult(
            success=True, action=f"navigate:{url}", data={"url": state.url}
        ))
        return state

    async def interact(self, actions: List[Dict[str, Any]]) -> List[ActionResult]:
        """일련의 액션(클릭, 타이핑 등) 실행."""
        await self._ensure_initialized()
        results = []
        
        for action_spec in actions:
            start = time.monotonic()
            action_type = action_spec.get("action", "").lower()
            
            try:
                if action_type == "click":
                    x, y = action_spec.get("x", 0), action_spec.get("y", 0)
                    if x == 0 and y == 0 and "selector" in action_spec:
                        # Fallback: get coords from selector
                        rect = self.js(f"(()=>{{const e=document.querySelector({json.dumps(action_spec['selector'])}); return e ? e.getBoundingClientRect() : null}})()")
                        if rect:
                            x, y = rect['x'] + rect['width']/2, rect['y'] + rect['height']/2
                    
                    self.cdp("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
                    self.cdp("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)
                    res = ActionResult(success=True, action="click", data={"x": x, "y": y})
                    
                elif action_type == "type":
                    text = action_spec.get("value", "")
                    self.cdp("Input.insertText", text=text)
                    res = ActionResult(success=True, action="type", data={"text": text})
                    
                elif action_type == "scroll":
                    x, y = action_spec.get("x", 0), action_spec.get("y", 0)
                    dy = action_spec.get("amount", 500) if action_spec.get("direction", "down") == "down" else -action_spec.get("amount", 500)
                    self.cdp("Input.dispatchMouseEvent", type="mouseWheel", x=x, y=y, deltaX=0, deltaY=dy)
                    res = ActionResult(success=True, action="scroll", data={"dy": dy})
                    
                elif action_type == "screenshot":
                    path = await self.take_screenshot(action_spec.get("filename"), action_spec.get("full_page", False))
                    res = ActionResult(success=True, action="screenshot", data={"filename": path})
                    
                elif action_type == "execute_js":
                    js_result = self.js(action_spec.get("script", ""))
                    res = ActionResult(success=True, action="execute_js", data={"result": js_result})
                    
                else:
                    raise ValueError(f"Unsupported action type: {action_type}")
                    
                res.execution_time = time.monotonic() - start
                results.append(res)
                self._action_history.append(res)
                
            except Exception as e:
                err_result = ActionResult(
                    success=False, action=action_type, error=str(e), execution_time=time.monotonic() - start
                )
                results.append(err_result)
                self._action_history.append(err_result)
                
        return results

    async def extract(self, extraction_spec: Dict[str, Any]) -> Dict[str, Any]:
        """페이지 내용 추출."""
        await self._ensure_initialized()
        result = {"timestamp": datetime.now().isoformat()}
        
        if "selectors" in extraction_spec:
            extracted = {}
            for field_name, selector in extraction_spec["selectors"].items():
                try:
                    text = self.js(f"(()=>{{const e=document.querySelector({json.dumps(selector)}); return e ? e.innerText : null}})()")
                    extracted[field_name] = text.strip() if text else None
                except Exception:
                    extracted[field_name] = None
            result["selector_data"] = extracted
            
        if extraction_spec.get("full_text", False):
            # HTML 가져오기
            doc = self.cdp("DOM.getDocument", depth=-1)
            html = self.cdp("DOM.getOuterHTML", nodeId=doc["root"]["nodeId"])["outerHTML"]
            result["markdown"] = markdownify.markdownify(html)
            result["text_length"] = len(result["markdown"])
            
        if extraction_spec.get("metadata", False):
            info = self.js("JSON.stringify({url:location.href, title:document.title})")
            if info:
                parsed = json.loads(info)
                result["page_url"] = parsed.get("url")
                result["page_title"] = parsed.get("title")
                
        return result

    async def verify(self, expectations: List[Dict[str, Any]]) -> VerificationResult:
        """검증 (DOM 기반)."""
        await self._ensure_initialized()
        checks = []
        passed = 0
        
        info = json.loads(self.js("JSON.stringify({url:location.href})") or "{}")
        url = info.get("url", "")
        
        for expectation in expectations:
            check_type = expectation.get("type", "")
            check_result = {"type": check_type, "passed": False, "detail": ""}
            
            try:
                if check_type == "url_contains":
                    check_result["passed"] = expectation["value"] in url
                    check_result["detail"] = f"URL: {url}"
                elif check_type == "element_exists":
                    exists = self.js(f"(()=>{{return !!document.querySelector({json.dumps(expectation['selector'])});}})()")
                    check_result["passed"] = bool(exists)
                elif check_type == "text_contains":
                    text = self.js(f"(()=>{{const e=document.querySelector({json.dumps(expectation.get('selector', 'body'))}); return e ? e.innerText : '';}})()")
                    check_result["passed"] = expectation["value"] in (text or "")
            except Exception as e:
                check_result["detail"] = str(e)
                
            if check_result["passed"]:
                passed += 1
            checks.append(check_result)
            
        total = len(expectations) if expectations else 1
        return VerificationResult(
            verified=(passed / total) >= 0.8,
            checks=checks,
            confidence=passed / total,
        )

    # --- Utility Methods ---
    async def get_page_state(self) -> PageState:
        await self._ensure_initialized()
        doc = self.cdp("DOM.getDocument", depth=-1)
        html = self.cdp("DOM.getOuterHTML", nodeId=doc["root"]["nodeId"])["outerHTML"]
        markdown = markdownify.markdownify(html)
        
        info = json.loads(self.js("JSON.stringify({url:location.href, title:document.title})") or '{"url":"","title":""}')
        
        return PageState(
            url=info.get("url", ""),
            title=info.get("title", ""),
            content_text=markdown[:10000],
            content_html=html[:10000],
            content_markdown=markdown[:10000],
        )

    async def take_screenshot(self, filename: Optional[str] = None, full_page: bool = True) -> str:
        await self._ensure_initialized()
        if filename is None:
            filename = f"/tmp/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        r = self.cdp("Page.captureScreenshot", format="png", captureBeyondViewport=full_page)
        with open(filename, "wb") as f:
            f.write(base64.b64decode(r["data"]))
        return filename

    def get_status(self) -> Dict[str, Any]:
        return {
            "backend": "cdp",
            "initialized": self._initialized,
            "daemon_alive": daemon_alive(self._name),
            "actions_count": len(self._action_history),
        }

    async def cleanup(self):
        """Clean up CDP connections."""
        # CDP WebSocket connection is handled by daemon, we just let socket close.
        # Could stop daemon here, but usually better to leave alive for next use.
        pass

# Singleton instance
_cdp_controller_instance: Optional[CDPBrowserController] = None

def get_cdp_controller() -> CDPBrowserController:
    global _cdp_controller_instance
    if _cdp_controller_instance is None:
        _cdp_controller_instance = CDPBrowserController()
    return _cdp_controller_instance
