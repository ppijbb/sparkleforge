"""Browser automation tool dispatch (ToolCategory.BROWSER): Playwright-backed search/navigation with a BrowserManager fallback."""
import asyncio
import logging
import os
import time
from typing import Any, Dict

from src.core.tools.registry import ToolResult

logger = logging.getLogger(__name__)


async def _playwright_dismiss_google_consent(page: Any) -> None:
    """Google 검색 진입 시 지역·쿠키 동의 UI가 뜨면 닫기 시도."""
    candidates = [
        'button:has-text("Accept all")',
        'button:has-text("Accept All")',
        'button:has-text("I agree")',
        'button:has-text("동의")',
        'button:has-text("모두 동의")',
        '[aria-label="Accept all"]',
        'form[action*="consent"] button',
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if await loc.is_visible(timeout=1200):
                await loc.click(timeout=2500)
                await page.wait_for_timeout(400)
                break
        except Exception:
            continue


async def _execute_browser_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """브라우저 자동화 도구 실행."""
    start_time = time.time()

    try:
        from src.automation.browser_manager import BrowserManager

        # BrowserManager 인스턴스 생성 (싱글톤 패턴 고려)
        browser_manager = BrowserManager()

        # browser-use 기반 브라우저 유틸은 `browser_navigate`/`browser_extract`에서만 사용됩니다.
        # `browser_search` 등 Playwright 전용 경로에서는 browser-use가 없어도 동작해야 합니다.
        if (
            tool_name in {"browser_navigate", "browser_extract"}
            and not browser_manager.browser_available
        ):
            await browser_manager.initialize_browser()

        if tool_name == "browser_navigate":
            # URL로 이동 및 콘텐츠 추출
            url = parameters.get("url", "")
            extraction_goal = parameters.get("extraction_goal", "extract_all_content")

            if not url:
                raise ValueError("URL parameter is required for browser_navigate")

            result = await browser_manager.navigate_and_extract(url, extraction_goal)

            return ToolResult(
                success=result.get("success", False),
                data=result,
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
            )

        elif tool_name == "browser_extract":
            # 특정 목표에 맞는 콘텐츠 추출
            url = parameters.get("url", "")
            extraction_goal = parameters.get("extraction_goal", "extract_all_content")

            if not url:
                raise ValueError("URL parameter is required for browser_extract")

            result = await browser_manager.navigate_and_extract(url, extraction_goal)

            return ToolResult(
                success=result.get("success", False),
                data=result,
                execution_time=time.time() - start_time,
                confidence=0.9 if result.get("success") else 0.0,
            )

        elif tool_name == "browser_screenshot":
            # 스크린샷 캡처
            url = parameters.get("url", "")
            output_path = parameters.get("output_path", None)

            if not url:
                raise ValueError("URL parameter is required for browser_screenshot")

            # Playwright를 사용한 스크린샷
            try:
                from playwright.async_api import async_playwright

                PLAYWRIGHT_AVAILABLE = True
            except ImportError:
                PLAYWRIGHT_AVAILABLE = False

            if PLAYWRIGHT_AVAILABLE:
                from playwright.async_api import async_playwright

                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()
                    await page.goto(url, wait_until="networkidle")

                    if output_path:
                        await page.screenshot(path=output_path, full_page=True)
                    else:
                        # 임시 파일에 저장
                        import tempfile

                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            output_path = tmp.name
                            await page.screenshot(path=output_path, full_page=True)

                    await browser.close()

                    return ToolResult(
                        success=True,
                        data={"screenshot_path": output_path, "url": url},
                        execution_time=time.time() - start_time,
                        confidence=0.9,
                    )
            else:
                raise RuntimeError("Playwright not available for screenshot")

        elif tool_name == "browser_interact":
            # 버튼 클릭, 폼 작성 등 상호작용
            url = parameters.get("url", "")
            actions = parameters.get("actions", [])  # List of action dicts

            if not url:
                raise ValueError("URL parameter is required for browser_interact")

            if not actions:
                raise ValueError("actions parameter is required for browser_interact")

            # Playwright를 사용한 상호작용
            try:
                from playwright.async_api import async_playwright

                PLAYWRIGHT_AVAILABLE = True
            except ImportError:
                PLAYWRIGHT_AVAILABLE = False

            if PLAYWRIGHT_AVAILABLE:
                from playwright.async_api import async_playwright

                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()
                    await page.goto(url, wait_until="networkidle")

                    results = []
                    for action in actions:
                        action_type = action.get("type")
                        selector = action.get("selector")
                        value = action.get("value")

                        try:
                            if action_type == "click":
                                await page.click(selector)
                                results.append(
                                    {
                                        "type": "click",
                                        "selector": selector,
                                        "success": True,
                                    }
                                )
                            elif action_type == "fill":
                                await page.fill(selector, value)
                                results.append(
                                    {
                                        "type": "fill",
                                        "selector": selector,
                                        "success": True,
                                    }
                                )
                            elif action_type == "select":
                                await page.select_option(selector, value)
                                results.append(
                                    {
                                        "type": "select",
                                        "selector": selector,
                                        "success": True,
                                    }
                                )
                            elif action_type == "wait":
                                await page.wait_for_selector(selector, timeout=5000)
                                results.append(
                                    {
                                        "type": "wait",
                                        "selector": selector,
                                        "success": True,
                                    }
                                )
                            else:
                                results.append(
                                    {
                                        "type": action_type,
                                        "success": False,
                                        "error": "Unknown action type",
                                    }
                                )
                        except Exception as e:
                            results.append({"type": action_type, "success": False, "error": str(e)})

                    # 최종 페이지 콘텐츠 추출
                    final_content = await page.content()

                    await browser.close()

                    return ToolResult(
                        success=all(r.get("success", False) for r in results),
                        data={
                            "url": url,
                            "actions": results,
                            "final_content": final_content[:10000],  # 처음 10000자만
                        },
                        execution_time=time.time() - start_time,
                        confidence=0.8 if all(r.get("success", False) for r in results) else 0.5,
                    )
            else:
                raise RuntimeError("Playwright not available for browser interaction")

        elif tool_name == "browser_search":
            # Headless Playwright 검색. Wikipedia는 안정적, Google은 SERP 파싱(차단 가능).
            import urllib.parse

            query = parameters.get("query", "")
            engine = (
                (
                    parameters.get("engine")
                    or os.getenv("SPARKLEFORGE_BROWSER_SEARCH_ENGINE", "wikipedia")
                )
                .lower()
                .strip()
            )
            max_results = int(min(20, max(1, int(parameters.get("max_results", 3) or 3))))

            if not query:
                raise ValueError("query parameter is required for browser_search")

            if engine not in {"wikipedia", "google", "bing", "duckduckgo"}:
                raise ValueError(
                    f"Unsupported browser_search engine: {engine}. "
                    "Use 'wikipedia', 'google', 'bing', or 'duckduckgo'."
                )

            from src.automation.browser_manager import BrowserManager

            browser_manager = BrowserManager()
            await browser_manager.initialize_playwright()
            if not browser_manager.playwright_page:
                raise RuntimeError("Playwright page not initialized for browser_search")

            page = browser_manager.playwright_page

            async def _wikipedia_search() -> ToolResult:
                """Playwright로 Wikipedia 검색을 수행하고 결과를 ToolResult로 반환."""
                q_encoded = urllib.parse.quote(query)
                url = (
                    f"https://en.wikipedia.org/w/index.php?search={q_encoded}"
                    f"&title=Special:Search&ns0=1"
                )
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(1200)
                wiki_results = await page.evaluate(
                    """
                    (maxResults) => {
                        const clean = (s) => (s || '').toString().trim();
                        const pageUrl = window.location.href;
                        const resultEls = Array.from(
                            document.querySelectorAll(
                                '#mw-content-text .mw-search-result, #mw-content-text li.mw-search-result'
                            )
                        );
                        const out = [];
                        for (const el of resultEls) {
                            const a = el.querySelector('a');
                            if (!a) continue;
                            const title = clean(a.textContent);
                            let href = a.href || a.getAttribute('href') || '';
                            if (href && href.startsWith('/')) {
                                href = new URL(href, location.origin).href;
                            }
                            const snippetEl =
                                el.querySelector('.searchresult') ||
                                el.querySelector('.mw-search-result-data') ||
                                el.querySelector('p') ||
                                el;
                            const snippet = clean(snippetEl.textContent);
                            if (title && href) {
                                out.push({
                                    title,
                                    url: href,
                                    snippet: snippet.slice(0, 500),
                                    source: 'wikipedia',
                                });
                            }
                            if (out.length >= maxResults) break;
                        }
                        if (out.length) return out.slice(0, maxResults);
                        const h1 = document.querySelector('h1');
                        const title = clean(h1 ? h1.innerText : document.title);
                        const ps = Array.from(
                            document.querySelectorAll('#mw-content-text .mw-parser-output p')
                        );
                        let snippet = '';
                        for (const p of ps) {
                            const t = clean(p.innerText);
                            if (t && t.length >= 30) {
                                snippet = t;
                                break;
                            }
                        }
                        return [{
                            title,
                            url: pageUrl,
                            snippet: clean(snippet).slice(0, 500),
                            source: 'wikipedia',
                        }].slice(0, maxResults);
                    }
                    """,
                    max_results,
                )

                if not isinstance(wiki_results, list) or len(wiki_results) == 0:
                    return ToolResult(
                        success=False,
                        data={"results": [], "query": query, "engine": "wikipedia"},
                        execution_time=time.time() - start_time,
                        confidence=0.0,
                        error="wikipedia returned no results",
                    )

                return ToolResult(
                    success=True,
                    data={
                        "results": wiki_results,
                        "query": query,
                        "engine": "wikipedia",
                    },
                    execution_time=time.time() - start_time,
                    confidence=0.9,
                )

            if engine == "wikipedia":
                return await _wikipedia_search()

            elif engine == "google":
                hl = os.getenv("BROWSER_SEARCH_GOOGLE_HL", "ko")
                gl = os.getenv("BROWSER_SEARCH_GOOGLE_GL", "kr")
                num = min(max_results, 15)
                q_enc = urllib.parse.quote(query)
                g_url = (
                    f"https://www.google.com/search?q={q_enc}"
                    f"&hl={urllib.parse.quote(hl)}&gl={urllib.parse.quote(gl)}"
                    f"&num={num}&pws=0"
                )
                await page.goto(g_url, wait_until="domcontentloaded", timeout=45000)
                await _playwright_dismiss_google_consent(page)
                await page.wait_for_timeout(800)
                try:
                    await page.wait_for_selector(
                        "#search, #rso, form#captcha-form, div#recaptcha",
                        timeout=15000,
                    )
                except Exception:
                    pass
                if await page.query_selector("form#captcha-form"):
                    return await _wikipedia_search()

                body_lower = ((await page.content()) or "")[:120000].lower()
                if (
                    "detected unusual traffic" in body_lower
                    or "unusual traffic from your computer network" in body_lower
                    or "/recaptcha/" in body_lower
                ):
                    return await _wikipedia_search()

                results = await page.evaluate(
                    """
                    (maxResults) => {
                        const clean = (s) => (s || '').toString().replace(/\\s+/g, ' ').trim();
                        const out = [];
                        const seen = new Set();
                        const skipUrl = (u) => {
                            if (!u || !u.startsWith('http')) return true;
                            try {
                                const h = new URL(u).hostname.toLowerCase();
                                if (h === 'google.com' || h.endsWith('.google.com')) return true;
                                if (h.includes('gstatic.com')) return true;
                                if (h.includes('youtube.com')) return true;
                            } catch (e) { return true; }
                            return false;
                        };
                        let nodes = document.querySelectorAll('#search a h3');
                        if (!nodes.length) nodes = document.querySelectorAll('#rso a h3');
                        if (!nodes.length) nodes = document.querySelectorAll('div[data-hveid] a h3');
                        for (const h3 of nodes) {
                            const a = h3.closest('a');
                            if (!a || !a.href) continue;
                            let href = a.href;
                            if (href.startsWith('/url?')) {
                                try {
                                    const sp = new URL(href, location.origin).searchParams;
                                    href = sp.get('q') || sp.get('url') || href;
                                } catch (e) {}
                            }
                            if (href.startsWith('/')) {
                                try { href = new URL(href, location.origin).href; } catch (e) {}
                            }
                            if (skipUrl(href)) continue;
                            const title = clean(h3.textContent);
                            if (!title || seen.has(href)) continue;
                            seen.add(href);
                            let snippet = '';
                            const block =
                                a.closest('div[data-sokoban-container]') ||
                                a.closest('div.Gx5Zad') ||
                                a.closest('div.g') ||
                                a.closest('div');
                            if (block) {
                                const st = clean(block.innerText || '');
                                if (st.length > title.length + 8) snippet = st.slice(0, 500);
                            }
                            out.push({ title, url: href, snippet, source: 'google' });
                            if (out.length >= maxResults) break;
                        }
                        return out.slice(0, maxResults);
                    }
                    """,
                    max_results,
                )

                if not isinstance(results, list) or len(results) == 0:
                    return await _wikipedia_search()

                return ToolResult(
                    success=True,
                    data={"results": results, "query": query, "engine": engine},
                    execution_time=time.time() - start_time,
                    confidence=0.85,
                )

            elif engine == "bing":
                q_enc = urllib.parse.quote(query)
                b_url = (
                    f"https://www.bing.com/search?q={q_enc}" f"&setlang=en-US&cc=US&form=QBLH&sp=-1"
                )
                await page.goto(b_url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(1000)
                results = await page.evaluate(
                    """
                    (maxResults) => {
                        const clean = (s) => (s || '').toString().replace(/\\s+/g, ' ').trim();
                        const out = [];
                        const nodes = document.querySelectorAll('#b_results .b_algo h2 a');
                        for (let i = 0; i < nodes.length && out.length < maxResults; i++) {
                            const a = nodes[i];
                            const title = clean(a.textContent);
                            const href = a.href || '';
                            let snippet = '';
                            const li = a.closest('li') || a.parentElement;
                            if (li) {
                                const p = li.querySelector('p');
                                if (p) snippet = clean(p.textContent);
                                else {
                                    const cap = li.querySelector('.b_caption p');
                                    if (cap) snippet = clean(cap.textContent);
                                }
                            }
                            if (title && href) {
                                out.push({ title, url: href, snippet: snippet.slice(0, 500), source: 'bing' });
                            }
                        }
                        return out;
                    }
                    """,
                    max_results,
                )
                if not isinstance(results, list) or len(results) == 0:
                    return await _wikipedia_search()
                return ToolResult(
                    success=True,
                    data={"results": results, "query": query, "engine": engine},
                    execution_time=time.time() - start_time,
                    confidence=0.8,
                )

            elif engine == "duckduckgo":
                q_enc = urllib.parse.quote(query)
                ddg_url = f"https://duckduckgo.com/html/?q={q_enc}&kl=us-en&kp=1"
                await page.goto(ddg_url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(1200)
                results = await page.evaluate(
                    """
                    (maxResults) => {
                        const clean = (s) => (s || '').toString().replace(/\\s+/g, ' ').trim();
                        const out = [];
                        const blocks = Array.from(document.querySelectorAll('.result'));
                        for (const b of blocks) {
                            const a = b.querySelector('a.result__a');
                            if (!a) continue;
                            const title = clean(a.textContent);
                            let href = a.href || b.querySelector('a')?.getAttribute('href') || '';
                            if (!href || !href.startsWith('http')) {
                                try {
                                    href = new URL(href, location.origin).href;
                                } catch (e) {}
                            }
                            const sn = b.querySelector('.result__snippet');
                            const snippet = sn ? clean(sn.textContent) : '';
                            if (title && href) {
                                out.push({ title, url: href, snippet: snippet.slice(0, 500), source: 'duckduckgo' });
                            }
                            if (out.length >= maxResults) break;
                        }
                        return out;
                    }
                    """,
                    max_results,
                )
                if not isinstance(results, list) or len(results) == 0:
                    return await _wikipedia_search()
                return ToolResult(
                    success=True,
                    data={"results": results, "query": query, "engine": engine},
                    execution_time=time.time() - start_time,
                    confidence=0.75,
                )

        else:
            raise ValueError(f"Unknown browser tool: {tool_name}")

    except Exception as e:
        logger.error(f"Browser tool execution failed: {tool_name} - {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=f"Browser tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0,
        )
