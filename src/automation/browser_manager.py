#!/usr/bin/env python3
"""SparkleForge Browser Automation Controller.

Playwright 기반의 체계적 웹 제어 시스템.
navigate → interact → extract → verify 파이프라인을 제공합니다.

이 모듈은 harness의 execution 단계에서 ToolCategory.BROWSER를 통해 호출되며,
verification 단계에서 결과를 검증하는 구조적 인터페이스를 가집니다.
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import markdownify

# Playwright imports
try:
    from playwright.async_api import Browser as PlaywrightBrowser
    from playwright.async_api import BrowserContext as PlaywrightContext
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import Page as PlaywrightPage
    from playwright.async_api import (
        async_playwright,
    )

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None
    PlaywrightBrowser = None
    PlaywrightPage = None
    PlaywrightContext = None
    PlaywrightError = Exception

# browser-use imports (보조 도구)
try:
    from browser_use import Browser as BrowserUseBrowser
    from browser_use import BrowserConfig
    from browser_use.browser.context import BrowserContext, BrowserContextConfig
    from browser_use.dom.service import DomService

    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    BrowserUseBrowser = Any
    BrowserConfig = Any
    BrowserContext = Any
    BrowserContextConfig = Any
    DomService = Any

logger = logging.getLogger(__name__)


class BrowserAction(Enum):
    """지원되는 브라우저 액션 타입."""

    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"
    EXECUTE_JS = "execute_js"
    SELECT = "select"
    HOVER = "hover"


@dataclass
class PageState:
    """페이지 상태 스냅샷."""

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
    """브라우저 액션 실행 결과."""

    success: bool
    action: str
    data: Any = None
    error: str | None = None
    execution_time: float = 0.0
    page_state_after: PageState | None = None


@dataclass
class VerificationResult:
    """브라우저 결과 검증 결과."""

    verified: bool
    checks: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    details: str = ""


class PlaywrightController:
    """Playwright 기반 체계적 웹 제어 컨트롤러.

    navigate → interact → extract → verify 파이프라인을 구현합니다.
    """

    def __init__(self):
        """PlaywrightController 초기화."""
        self._playwright = None
        self._browser: PlaywrightBrowser | None = None
        self._context: PlaywrightContext | None = None
        self._page: PlaywrightPage | None = None
        self._lock = asyncio.Lock()
        self._initialized = False
        self._action_history: List[ActionResult] = []

        logger.info("PlaywrightController initialized")

    @property
    def is_available(self) -> bool:
        """Playwright 사용 가능 여부."""
        return PLAYWRIGHT_AVAILABLE

    @property
    def is_initialized(self) -> bool:
        """브라우저 초기화 여부."""
        return self._initialized and self._page is not None

    async def initialize(self) -> bool:
        """Playwright 브라우저 인스턴스를 초기화합니다."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "playwright 패키지가 필요합니다. "
                "`pip install playwright && playwright install chromium` 으로 설치하세요."
            )

        async with self._lock:
            if self._initialized:
                return True

            self._playwright = await async_playwright().start()

            # 환경 감지를 통한 headless 결정
            is_headless = (
                not hasattr(sys, "ps1")
                or "streamlit" in sys.modules
                or os.getenv("BACKGROUND_MODE", "false").lower() == "true"
            )

            self._browser = await self._playwright.chromium.launch(
                headless=is_headless,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
                ignore_default_args=["--enable-automation"],
            )

            # 컨텍스트 설정
            user_agent = os.getenv(
                "PLAYWRIGHT_USER_AGENT",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            )
            locale = os.getenv("PLAYWRIGHT_LOCALE", "ko-KR")
            accept_language = os.getenv(
                "PLAYWRIGHT_ACCEPT_LANGUAGE",
                "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            )

            self._context = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=user_agent,
                locale=locale,
                extra_http_headers={"Accept-Language": accept_language},
            )

            # anti-detection 스크립트
            await self._context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
            )

            self._page = await self._context.new_page()
            self._initialized = True
            logger.info(f"PlaywrightController initialized (headless={is_headless})")
            return True

    async def _ensure_initialized(self) -> PlaywrightPage:
        """페이지가 초기화되었는지 보장합니다."""
        if not self._initialized:
            await self.initialize()
        if self._page is None:
            raise RuntimeError("Playwright page is not available after initialization")
        return self._page

    # ============================
    # 핵심 API: navigate
    # ============================

    async def navigate(
        self, url: str, wait_until: str = "networkidle", timeout: int = 30000
    ) -> PageState:
        """URL로 이동하고 페이지 상태를 반환합니다.

        Args:
            url: 이동할 URL
            wait_until: 대기 조건 ('load', 'domcontentloaded', 'networkidle')
            timeout: 타임아웃 (밀리초)

        Returns:
            PageState: 현재 페이지의 상태 스냅샷
        """
        page = await self._ensure_initialized()

        await page.goto(url, wait_until=wait_until, timeout=timeout)
        state = await self.get_page_state()

        self._action_history.append(
            ActionResult(
                success=True,
                action=f"navigate:{url}",
                data={"url": state.url, "title": state.title},
            )
        )
        return state

    # ============================
    # 핵심 API: interact
    # ============================

    async def interact(self, actions: List[Dict[str, Any]]) -> List[ActionResult]:
        """일련의 브라우저 액션을 순차 실행합니다.

        Args:
            actions: 액션 리스트. 각 액션은 {"action": "click", "selector": "...", ...} 형태.

        Returns:
            List[ActionResult]: 각 액션의 실행 결과
        """
        page = await self._ensure_initialized()
        results = []

        for action_spec in actions:
            import time as _time

            start = _time.monotonic()
            action_type = action_spec.get("action", "").lower()

            try:
                result = await self._execute_action(page, action_type, action_spec)
                result.execution_time = _time.monotonic() - start
                results.append(result)
                self._action_history.append(result)
            except Exception as e:
                err_result = ActionResult(
                    success=False,
                    action=action_type,
                    error=str(e),
                    execution_time=_time.monotonic() - start,
                )
                results.append(err_result)
                self._action_history.append(err_result)
                logger.warning(f"Action '{action_type}' failed: {e}")

        return results

    async def _execute_action(
        self, page: PlaywrightPage, action_type: str, spec: Dict[str, Any]
    ) -> ActionResult:
        """개별 액션을 실행합니다."""
        selector = spec.get("selector", "")
        value = spec.get("value", "")
        timeout = spec.get("timeout", 10000)

        if action_type == BrowserAction.CLICK.value:
            await page.click(selector, timeout=timeout)
            return ActionResult(success=True, action="click", data={"selector": selector})

        elif action_type == BrowserAction.TYPE.value:
            await page.fill(selector, value, timeout=timeout)
            return ActionResult(
                success=True, action="type", data={"selector": selector, "value": value}
            )

        elif action_type == BrowserAction.SCROLL.value:
            direction = spec.get("direction", "down")
            amount = spec.get("amount", 500)
            delta_y = amount if direction == "down" else -amount
            await page.mouse.wheel(0, delta_y)
            return ActionResult(
                success=True, action="scroll", data={"direction": direction, "amount": amount}
            )

        elif action_type == BrowserAction.WAIT.value:
            wait_for = spec.get("wait_for", "networkidle")
            if wait_for.startswith("selector:"):
                await page.wait_for_selector(wait_for[9:], timeout=timeout)
            else:
                await page.wait_for_load_state(wait_for, timeout=timeout)
            return ActionResult(success=True, action="wait", data={"wait_for": wait_for})

        elif action_type == BrowserAction.SCREENSHOT.value:
            filename = spec.get(
                "filename", f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            full_page = spec.get("full_page", True)
            await page.screenshot(path=filename, full_page=full_page)
            return ActionResult(success=True, action="screenshot", data={"filename": filename})

        elif action_type == BrowserAction.EXECUTE_JS.value:
            script = spec.get("script", "")
            js_result = await page.evaluate(script)
            return ActionResult(success=True, action="execute_js", data={"result": js_result})

        elif action_type == BrowserAction.SELECT.value:
            await page.select_option(selector, value, timeout=timeout)
            return ActionResult(
                success=True, action="select", data={"selector": selector, "value": value}
            )

        elif action_type == BrowserAction.HOVER.value:
            await page.hover(selector, timeout=timeout)
            return ActionResult(success=True, action="hover", data={"selector": selector})

        elif action_type == BrowserAction.NAVIGATE.value:
            url = spec.get("url", "")
            await page.goto(url, wait_until=spec.get("wait_until", "networkidle"), timeout=timeout)
            return ActionResult(success=True, action="navigate", data={"url": url})

        else:
            raise ValueError(f"Unsupported action type: {action_type}")

    # ============================
    # 핵심 API: extract
    # ============================

    async def extract(
        self,
        extraction_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """페이지에서 구조화된 데이터를 추출합니다.

        Args:
            extraction_spec: 추출 명세
                - "selectors": {"field_name": "css_selector", ...} — CSS 셀렉터 기반 추출
                - "goal": "..." — LLM 기반 목표 지향 추출 (LLM은 외부에서 처리)
                - "full_text": True — 전체 텍스트 추출

        Returns:
            추출된 데이터 딕셔너리
        """
        page = await self._ensure_initialized()
        result = {"timestamp": datetime.now().isoformat()}

        # CSS 셀렉터 기반 추출
        if "selectors" in extraction_spec:
            extracted = {}
            for field_name, selector in extraction_spec["selectors"].items():
                try:
                    element = await page.query_selector(selector)
                    if element:
                        text = await element.text_content()
                        extracted[field_name] = text.strip() if text else ""
                    else:
                        extracted[field_name] = None
                except Exception as e:
                    logger.warning(f"Extraction failed for '{field_name}' ({selector}): {e}")
                    extracted[field_name] = None
            result["selector_data"] = extracted

        # 전체 텍스트 / 마크다운 추출
        if extraction_spec.get("full_text", False):
            html = await page.content()
            result["markdown"] = markdownify.markdownify(html)
            result["text_length"] = len(result["markdown"])

        # 멀티 엘리먼트 추출 (리스트)
        if "multi_selector" in extraction_spec:
            items = []
            elements = await page.query_selector_all(extraction_spec["multi_selector"])
            for el in elements:
                text = await el.text_content()
                if text and text.strip():
                    items.append(text.strip())
            result["items"] = items

        # 메타데이터 추출
        if extraction_spec.get("metadata", False):
            result["page_url"] = page.url
            result["page_title"] = await page.title()

        return result

    # ============================
    # 핵심 API: verify
    # ============================

    async def verify(self, expectations: List[Dict[str, Any]]) -> VerificationResult:
        """페이지 상태가 기대와 부합하는지 검증합니다.

        Args:
            expectations: 검증 조건 리스트
                [
                    {"type": "url_contains", "value": "example.com"},
                    {"type": "element_exists", "selector": "#main-content"},
                    {"type": "text_contains", "selector": "h1", "value": "Title"},
                    {"type": "element_count", "selector": ".item", "min": 3},
                ]

        Returns:
            VerificationResult: 검증 결과
        """
        page = await self._ensure_initialized()
        checks = []
        passed = 0

        for expectation in expectations:
            check_type = expectation.get("type", "")
            check_result = {"type": check_type, "passed": False, "detail": ""}

            try:
                if check_type == "url_contains":
                    check_result["passed"] = expectation["value"] in page.url
                    check_result["detail"] = f"URL: {page.url}"

                elif check_type == "element_exists":
                    element = await page.query_selector(expectation["selector"])
                    check_result["passed"] = element is not None
                    check_result["detail"] = f"Selector: {expectation['selector']}"

                elif check_type == "text_contains":
                    element = await page.query_selector(expectation.get("selector", "body"))
                    if element:
                        text = await element.text_content()
                        check_result["passed"] = expectation["value"] in (text or "")
                        check_result["detail"] = (
                            f"Found text match in {expectation.get('selector', 'body')}"
                        )
                    else:
                        check_result["detail"] = f"Element not found: {expectation.get('selector')}"

                elif check_type == "element_count":
                    elements = await page.query_selector_all(expectation["selector"])
                    count = len(elements)
                    min_count = expectation.get("min", 1)
                    max_count = expectation.get("max", float("inf"))
                    check_result["passed"] = min_count <= count <= max_count
                    check_result["detail"] = f"Count: {count} (expected {min_count}-{max_count})"

                elif check_type == "page_loaded":
                    check_result["passed"] = page.url != "about:blank"
                    check_result["detail"] = f"URL: {page.url}"

                elif check_type == "no_error":
                    # 페이지에 에러 메시지가 없는지 확인
                    error_selectors = [".error", "#error", "[role='alert']", ".alert-danger"]
                    has_error = False
                    for sel in error_selectors:
                        el = await page.query_selector(sel)
                        if el:
                            has_error = True
                            break
                    check_result["passed"] = not has_error
                    check_result["detail"] = (
                        "No error elements found" if not has_error else "Error element detected"
                    )

                else:
                    check_result["detail"] = f"Unknown check type: {check_type}"

            except Exception as e:
                check_result["detail"] = f"Check failed with error: {e}"

            if check_result["passed"]:
                passed += 1
            checks.append(check_result)

        total = len(expectations) if expectations else 1
        confidence = passed / total

        return VerificationResult(
            verified=confidence >= 0.8,
            checks=checks,
            confidence=confidence,
            details=f"{passed}/{total} checks passed",
        )

    # ============================
    # 유틸리티
    # ============================

    async def get_page_state(self) -> PageState:
        """현재 페이지의 전체 상태 스냅샷을 반환합니다."""
        page = await self._ensure_initialized()
        html = await page.content()
        markdown = markdownify.markdownify(html)
        title = await page.title()

        return PageState(
            url=page.url,
            title=title,
            content_text=markdown[:10000],
            content_html=html[:10000],
            content_markdown=markdown[:10000],
        )

    async def take_screenshot(self, filename: str | None = None, full_page: bool = True) -> str:
        """스크린샷을 캡처합니다."""
        page = await self._ensure_initialized()
        if filename is None:
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        await page.screenshot(path=filename, full_page=full_page)
        return filename

    async def generate_pdf(self, filename: str | None = None) -> str:
        """PDF를 생성합니다."""
        page = await self._ensure_initialized()
        if filename is None:
            filename = f"page_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        await page.pdf(path=filename, format="A4", print_background=True)
        return filename

    def get_action_history(self) -> List[Dict[str, Any]]:
        """수행된 액션 이력을 반환합니다."""
        return [
            {
                "action": r.action,
                "success": r.success,
                "error": r.error,
                "execution_time": r.execution_time,
            }
            for r in self._action_history
        ]

    async def cleanup(self):
        """리소스를 정리합니다."""
        async with self._lock:
            if self._page:
                await self._page.close()
                self._page = None
            if self._context:
                await self._context.close()
                self._context = None
            if self._browser:
                await self._browser.close()
                self._browser = None
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
            self._initialized = False
            self._action_history.clear()
            logger.info("PlaywrightController resources cleaned up")

    def get_status(self) -> Dict[str, Any]:
        """컨트롤러 상태를 반환합니다."""
        return {
            "playwright_available": PLAYWRIGHT_AVAILABLE,
            "browser_use_available": BROWSER_USE_AVAILABLE,
            "initialized": self._initialized,
            "actions_count": len(self._action_history),
        }


# ===== 모듈 수준 싱글톤 =====
_controller_instance: PlaywrightController | None = None


def get_playwright_controller() -> PlaywrightController:
    """PlaywrightController 싱글톤 인스턴스를 반환합니다."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = PlaywrightController()
    return _controller_instance


# ===== 하위 호환: BrowserManager 래퍼 및 Backend Router =====
class BrowserBackend(Enum):
    PLAYWRIGHT = "playwright"
    CDP = "cdp"
    AUTO = "auto"


def get_browser_backend(preference: BrowserBackend = BrowserBackend.AUTO) -> Any:
    """Returns the appropriate browser controller instance."""
    if preference == BrowserBackend.PLAYWRIGHT:
        return get_playwright_controller()
    elif preference == BrowserBackend.CDP:
        from src.automation.cdp_browser_controller import get_cdp_controller

        return get_cdp_controller()
    else:
        # AUTO logic: try CDP first, fallback to Playwright
        try:
            from src.automation.cdp_browser_controller import get_cdp_controller

            cdp_ctrl = get_cdp_controller()
            if cdp_ctrl.is_available:
                return cdp_ctrl
        except ImportError:
            pass
        return get_playwright_controller()


class BrowserManager:
    """하위 호환을 위한 BrowserManager 래퍼.

    내부적으로 PlaywrightController 또는 CDPBrowserController를 사용합니다.
    """

    def __init__(
        self, config_path: str | None = None, backend: BrowserBackend = BrowserBackend.AUTO
    ):
        self._controller = get_browser_backend(backend)
        self.config_path = config_path

        # 환경 감지 (legacy 호환)
        self.is_cli = not hasattr(sys, "ps1") and not hasattr(sys, "getwindowsversion")
        self.is_streamlit = "streamlit" in sys.modules
        self.is_background = os.getenv("BACKGROUND_MODE", "false").lower() == "true"

    @property
    def browser_available(self) -> bool:
        return self._controller.is_initialized

    async def initialize_browser(self) -> bool:
        return await self._controller.initialize()

    async def navigate_and_extract(
        self, url: str, extraction_goal: str, llm=None
    ) -> Dict[str, Any]:
        """URL로 이동하여 콘텐츠를 추출합니다."""
        try:
            await self._controller.navigate(url)

            # Use universal extract format which both controllers support
            extracted = await self._controller.extract({"full_text": True, "metadata": True})

            # LLM이 제공되면 콘텐츠 처리
            if llm:
                extracted_data = await self._process_content_with_llm(
                    extracted.get("markdown", ""), extraction_goal, llm
                )
            else:
                extracted_data = {"raw_content": extracted.get("markdown", "")[:2000]}

            return {
                "success": True,
                "url": url,
                "extraction_goal": extraction_goal,
                "extracted_data": extracted_data,
                "content_length": extracted.get("text_length", 0),
                "method": "playwright",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            raise RuntimeError(f"Content extraction failed for {url}: {e}")

    async def _process_content_with_llm(
        self, content: str, extraction_goal: str, llm, max_content_length: int = 2000
    ) -> Dict[str, Any]:
        """LLM으로 콘텐츠를 처리합니다."""
        try:
            prompt = f"""
            Extract content from a webpage based on the following goal.
            Respond in JSON format.
            
            Extraction goal: {extraction_goal}
            
            Page content:
            {content[:max_content_length]}
            """
            response = await asyncio.to_thread(llm.generate_content, prompt)
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {
                    "extracted_content": {
                        "text": response.text,
                        "metadata": {"extraction_goal": extraction_goal},
                    }
                }
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            raise RuntimeError(f"LLM content processing failed: {e}")

    async def search_and_extract(
        self, query: str, extraction_goal: str, max_results: int = 3, llm=None
    ) -> List[Dict[str, Any]]:
        """웹 검색 후 콘텐츠를 추출합니다."""
        from src.core.mcp_integration import execute_tool

        search_result = await execute_tool("g-search", {"query": query, "max_results": max_results})
        if not search_result.get("success", False):
            raise RuntimeError(f"Web search failed: {search_result.get('error', 'Unknown')}")

        results_list = search_result.get("data", {}).get("results", [])
        extracted_results = []

        for result in results_list[:max_results]:
            try:
                extraction_result = await self.navigate_and_extract(
                    result.get("url", ""), extraction_goal, llm
                )
                if extraction_result.get("success"):
                    extraction_result["search_result"] = result
                    extracted_results.append(extraction_result)
            except Exception as e:
                logger.warning(f"Failed to extract from {result.get('url', '')}: {e}")

        return extracted_results

    async def take_screenshot(self, url: str, filename: str | None = None) -> Dict[str, Any]:
        """스크린샷을 캡처합니다."""
        await self._controller.navigate(url)
        path = await self._controller.take_screenshot(filename)
        return {
            "success": True,
            "filename": path,
            "url": url,
            "timestamp": datetime.now().isoformat(),
        }

    async def generate_pdf(self, url: str, filename: str | None = None) -> Dict[str, Any]:
        """PDF를 생성합니다."""
        await self._controller.navigate(url)
        path = await self._controller.generate_pdf(filename)
        return {
            "success": True,
            "filename": path,
            "url": url,
            "timestamp": datetime.now().isoformat(),
        }

    async def execute_javascript(self, url: str, script: str) -> Dict[str, Any]:
        """JavaScript를 실행합니다."""
        await self._controller.navigate(url)
        results = await self._controller.interact([{"action": "execute_js", "script": script}])
        if results and results[0].success:
            return {
                "success": True,
                "result": results[0].data.get("result") if results[0].data else None,
                "url": url,
                "script": script,
                "timestamp": datetime.now().isoformat(),
            }
        raise RuntimeError(
            f"JavaScript execution failed: {results[0].error if results else 'No result'}"
        )

    async def fill_form(self, url: str, form_data: Dict[str, str]) -> Dict[str, Any]:
        """폼을 채우고 제출합니다."""
        await self._controller.navigate(url)
        actions = []
        for field_name, value in form_data.items():
            # name 또는 id 기반 셀렉터
            actions.append(
                {
                    "action": "type",
                    "selector": f'input[name="{field_name}"], input[id="{field_name}"], '
                    f'textarea[name="{field_name}"], textarea[id="{field_name}"]',
                    "value": value,
                }
            )
        results = await self._controller.interact(actions)
        return {
            "success": all(r.success for r in results),
            "url": url,
            "form_data": form_data,
            "timestamp": datetime.now().isoformat(),
        }

    async def extract_structured_data(self, url: str, schema: Dict[str, str]) -> Dict[str, Any]:
        """구조화된 데이터를 추출합니다."""
        await self._controller.navigate(url)
        extracted = await self._controller.extract({"selectors": schema, "metadata": True})
        return {
            "success": True,
            "url": url,
            "extracted_data": extracted.get("selector_data", {}),
            "timestamp": datetime.now().isoformat(),
        }

    async def monitor_page_changes(
        self, url: str, interval: int = 5, duration: int = 60
    ) -> Dict[str, Any]:
        """페이지 변경 모니터링."""
        page_state = await self._controller.navigate(url)
        prev_text = page_state.content_text
        changes = []
        start_time = datetime.now()

        while (datetime.now() - start_time).seconds < duration:
            await asyncio.sleep(interval)
            current_state = await self._controller.get_page_state()
            changed = current_state.content_text != prev_text
            changes.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "change_detected": changed,
                    "content_length": len(current_state.content_text),
                }
            )
            if changed:
                prev_text = current_state.content_text

        return {
            "success": True,
            "url": url,
            "monitoring_duration": duration,
            "changes_detected": len([c for c in changes if c["change_detected"]]),
            "changes": changes,
            "timestamp": datetime.now().isoformat(),
        }

    async def cleanup(self):
        """리소스 정리."""
        await self._controller.cleanup()

    def get_status(self) -> Dict[str, Any]:
        """상태 반환."""
        return self._controller.get_status()
