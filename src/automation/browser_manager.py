#!/usr/bin/env python3
"""
Enhanced Browser Manager for Local Researcher

This module provides robust browser automation capabilities optimized for
CLI, background, and Streamlit environments with comprehensive error handling.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import requests
from bs4 import BeautifulSoup
import markdownify

# Browser automation imports
try:
    from browser_use import Browser as BrowserUseBrowser
    from browser_use import BrowserConfig
    from browser_use.browser.context import BrowserContext, BrowserContextConfig
    from browser_use.dom.service import DomService
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    BrowserUseBrowser = None
    BrowserConfig = None
    BrowserContext = None
    BrowserContextConfig = None
    DomService = None

# Playwright imports for advanced features
try:
    from playwright.async_api import async_playwright, Browser as PlaywrightBrowser, Page as PlaywrightPage
    from playwright.async_api import BrowserContext as PlaywrightContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None
    PlaywrightBrowser = None
    PlaywrightPage = None
    PlaywrightContext = None

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

logger = setup_logger("browser_manager", log_level="INFO")


class BrowserManager:
    """Enhanced browser manager with robust error handling."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the browser manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        
        # Browser components
        self.browser: Optional[BrowserUseBrowser] = None
        self.browser_context: Optional[BrowserContext] = None
        self.dom_service: Optional[DomService] = None
        self.browser_lock = asyncio.Lock()
        
        # Playwright components for advanced features
        self.playwright_browser: Optional[PlaywrightBrowser] = None
        self.playwright_context: Optional[PlaywrightContext] = None
        self.playwright_page: Optional[PlaywrightPage] = None
        self.playwright_lock = asyncio.Lock()
        
        # Environment detection
        self.is_cli = not hasattr(sys, 'ps1') and not hasattr(sys, 'getwindowsversion')
        self.is_streamlit = 'streamlit' in sys.modules
        self.is_background = os.getenv('BACKGROUND_MODE', 'false').lower() == 'true'
        
        # Browser status
        self.browser_available = False
        
        logger.info(f"Browser Manager initialized - CLI: {self.is_cli}, Streamlit: {self.is_streamlit}, Background: {self.is_background}")
    
    async def initialize_browser(self) -> bool:
        """Initialize browser with enhanced error handling and environment detection."""
        try:
            if not BROWSER_USE_AVAILABLE:
                raise RuntimeError("browser-use package is required but not available. Please install it with: pip install browser-use")
            
            async with self.browser_lock:
                if self.browser is None:
                    browser_config = self.config_manager.get_browser_config()
                    browser_config_kwargs = self._get_optimized_browser_config(browser_config)
                    
                    # Initialize browser with retry mechanism
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))
                            logger.info(f"Browser initialized successfully (attempt {attempt + 1})")
                            break
                        except Exception as e:
                            logger.warning(f"Browser initialization attempt {attempt + 1} failed: {e}")
                            if attempt == max_retries - 1:
                                logger.error("All browser initialization attempts failed")
                                raise RuntimeError("Browser initialization failed after multiple attempts")
                                return False
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                if self.browser_context is None:
                    context_config = self._get_optimized_context_config()
                    self.browser_context = await self.browser.new_context(context_config)
                    self.dom_service = DomService(await self.browser_context.get_current_page())
                    logger.info("Browser context initialized successfully")
                
                self.browser_available = True
                return True
                
        except Exception as e:
            logger.error(f"Browser initialization failed: {e}")
            raise RuntimeError("Browser initialization failed")
            return False
    
    def _get_optimized_browser_config(self, browser_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimized browser configuration for different environments."""
        # Default to headless for CLI/background/Streamlit environments
        headless = browser_config.get("headless", self.is_cli or self.is_streamlit or self.is_background)
        
        config = {
            "headless": headless,
            "disable_security": browser_config.get("disable_security", True),
            "disable_images": browser_config.get("disable_images", True),
            "disable_javascript": browser_config.get("disable_javascript", False),
            "disable_css": browser_config.get("disable_css", False),
            "disable_plugins": browser_config.get("disable_plugins", True),
            "disable_extensions": browser_config.get("disable_extensions", True),
            "disable_dev_shm_usage": True,
            "no_sandbox": True,
            "disable_gpu": True,
            "disable_web_security": True,
            "disable_features": "VizDisplayCompositor",
            "window_size": browser_config.get("window_size", (1920, 1080)),
            "user_agent": browser_config.get("user_agent", 
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        }
        
        # Streamlit-specific optimizations
        if self.is_streamlit:
            config.update({
                "disable_images": True,
                "disable_css": True,
                "disable_plugins": True,
                "disable_extensions": True,
                "disable_gpu": True,
                "disable_dev_shm_usage": True,
                "no_sandbox": True,
                "headless": True
            })
        
        logger.info(f"Browser config optimized for environment: CLI={self.is_cli}, Streamlit={self.is_streamlit}, Background={self.is_background}")
        return config
    
    def _get_optimized_context_config(self) -> BrowserContextConfig:
        """Get optimized browser context configuration."""
        return BrowserContextConfig(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
        )
    
    async def navigate_and_extract(self, url: str, extraction_goal: str, llm=None) -> Dict[str, Any]:
        """Navigate to URL and extract content using MCP tools only.
        
        Args:
            url: URL to navigate to
            extraction_goal: Specific goal for content extraction
            llm: LLM instance for content processing
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            # Use MCP tools for content extraction
            if self.browser_available:
                return await self._browser_extract(url, extraction_goal, llm)
            else:
                raise RuntimeError("Browser not available. Please ensure browser-use is properly installed and initialized.")
                
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return {
                "success": False,
                "url": url,
                "extraction_goal": extraction_goal,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _browser_extract(self, url: str, extraction_goal: str, llm=None) -> Dict[str, Any]:
        """Extract content using browser automation."""
        try:
            if not self.browser_available:
                raise Exception("Browser not available")
            
            context = await self._ensure_browser_initialized()
            if context is None:
                raise Exception("Browser context not available")
            
            page = await context.get_current_page()
            
            # Enhanced navigation with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await page.goto(url, timeout=30000)
                    await page.wait_for_load_state("networkidle", timeout=10000)
                    break
                except Exception as e:
                    logger.warning(f"Browser navigation attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
            
            # Extract content
            content = markdownify.markdownify(await page.content())
            max_content_length = self.config_manager.get_browser_config().get("max_content_length", 2000)
            
            # Process content with LLM if available
            if llm:
                extracted_data = await self._process_content_with_llm(content, extraction_goal, llm, max_content_length)
            else:
                extracted_data = {"raw_content": content[:max_content_length]}
            
            return {
                "success": True,
                "url": url,
                "extraction_goal": extraction_goal,
                "extracted_data": extracted_data,
                "content_length": len(content),
                "method": "browser",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Browser extraction failed: {e}")
            raise RuntimeError(f"Browser extraction failed: {e}")
    
    
    async def _process_content_with_llm(self, content: str, extraction_goal: str, llm, max_content_length: int) -> Dict[str, Any]:
        """Process content using LLM."""
        try:
            prompt = f"""
            Your task is to extract content from a webpage based on a specific goal.
            Extract all relevant information around this goal from the page.
            If the goal is vague, provide a comprehensive summary.
            Respond in JSON format.
            
            Extraction goal: {extraction_goal}
            
            Page content:
            {content[:max_content_length]}
            """
            
            response = await asyncio.to_thread(llm.generate_content, prompt)
            
            # Parse LLM response
            try:
                extracted_data = json.loads(response.text)
            except json.JSONDecodeError:
                extracted_data = {
                    "extracted_content": {
                        "text": response.text,
                        "metadata": {
                            "extraction_goal": extraction_goal
                        }
                    }
                }
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return {"raw_content": content[:max_content_length], "error": str(e)}
    
    async def _ensure_browser_initialized(self) -> Optional[BrowserContext]:
        """Ensure browser is initialized."""
        if not self.browser_available:
            return None
        
        try:
            if self.browser_context is None:
                await self.initialize_browser()
            
            return self.browser_context
            
        except Exception as e:
            logger.error(f"Browser context initialization failed: {e}")
            return None
    
    async def search_and_extract(self, query: str, extraction_goal: str, max_results: int = 3, llm=None) -> List[Dict[str, Any]]:
        """Perform web search and extract content from results."""
        try:
            # First perform web search
            search_results = await self._perform_web_search(query, max_results)
            
            if not search_results:
                return []
            
            extracted_results = []
            
            for result in search_results[:max_results]:
                try:
                    # Extract content from each search result
                    extraction_result = await self.navigate_and_extract(
                        result.get('url', ''), 
                        extraction_goal,
                        llm
                    )
                    
                    if extraction_result.get('success'):
                        extraction_result['search_result'] = result
                        extracted_results.append(extraction_result)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract from {result.get('url', '')}: {e}")
                    continue
            
            return extracted_results
            
        except Exception as e:
            logger.error(f"Search and extract failed: {e}")
            return []
    
    async def _perform_web_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform web search using MCP tools."""
        try:
            # Use MCP tools for web search
            from src.core.mcp_integration import execute_tool
            
            search_result = await execute_tool("g-search", {
                "query": query,
                "max_results": max_results
            })
            
            if search_result.get('success', False):
                return search_result.get('data', {}).get('results', [])
            else:
                logger.error(f"Web search failed: {search_result.get('error', 'Unknown error')}")
                return []
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def cleanup(self):
        """Clean up browser resources."""
        try:
            async with self.browser_lock:
                if self.browser_context:
                    await self.browser_context.close()
                    self.browser_context = None
                    self.dom_service = None
                
                if self.browser:
                    await self.browser.close()
                    self.browser = None
                
                self.browser_available = False
                logger.info("Browser resources cleaned up")
            
            # Cleanup Playwright resources
            async with self.playwright_lock:
                if self.playwright_page:
                    await self.playwright_page.close()
                    self.playwright_page = None
                
                if self.playwright_context:
                    await self.playwright_context.close()
                    self.playwright_context = None
                
                if self.playwright_browser:
                    await self.playwright_browser.close()
                    self.playwright_browser = None
                    
        except Exception as e:
            logger.error(f"Browser cleanup failed: {e}")
    
    # ==================== ADVANCED BROWSER FEATURES ====================
    
    async def initialize_playwright(self) -> bool:
        """Initialize Playwright for advanced browser features."""
        try:
            if not PLAYWRIGHT_AVAILABLE:
                logger.warning("Playwright not available. Advanced features disabled.")
                return False
            
            async with self.playwright_lock:
                if self.playwright_browser is None:
                    self.playwright = await async_playwright().start()
                    
                    # Choose browser type based on environment
                    if self.is_cli or self.is_background:
                        self.playwright_browser = await self.playwright.chromium.launch(
                            headless=True,
                            args=['--no-sandbox', '--disable-dev-shm-usage']
                        )
                    else:
                        self.playwright_browser = await self.playwright.chromium.launch(
                            headless=False
                        )
                    
                    # Create context with optimized settings
                    self.playwright_context = await self.playwright_browser.new_context(
                        viewport={'width': 1920, 'height': 1080},
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    )
                    
                    self.playwright_page = await self.playwright_context.new_page()
                    
                    logger.info("Playwright initialized successfully")
                    return True
                    
        except Exception as e:
            logger.error(f"Playwright initialization failed: {e}")
            return False
    
    async def take_screenshot(self, url: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Take a screenshot of a webpage."""
        try:
            if not await self.initialize_playwright():
                return {'success': False, 'error': 'Playwright not available'}
            
            await self.playwright_page.goto(url, wait_until='networkidle')
            
            if filename is None:
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            await self.playwright_page.screenshot(path=filename, full_page=True)
            
            return {
                'success': True,
                'filename': filename,
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def generate_pdf(self, url: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Generate PDF from a webpage."""
        try:
            if not await self.initialize_playwright():
                return {'success': False, 'error': 'Playwright not available'}
            
            await self.playwright_page.goto(url, wait_until='networkidle')
            
            if filename is None:
                filename = f"page_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            await self.playwright_page.pdf(
                path=filename,
                format='A4',
                print_background=True
            )
            
            return {
                'success': True,
                'filename': filename,
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_javascript(self, url: str, script: str) -> Dict[str, Any]:
        """Execute JavaScript on a webpage."""
        try:
            if not await self.initialize_playwright():
                return {'success': False, 'error': 'Playwright not available'}
            
            await self.playwright_page.goto(url, wait_until='networkidle')
            
            result = await self.playwright_page.evaluate(script)
            
            return {
                'success': True,
                'result': result,
                'url': url,
                'script': script,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"JavaScript execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def fill_form(self, url: str, form_data: Dict[str, str]) -> Dict[str, Any]:
        """Fill and submit a form on a webpage."""
        try:
            if not await self.initialize_playwright():
                return {'success': False, 'error': 'Playwright not available'}
            
            await self.playwright_page.goto(url, wait_until='networkidle')
            
            # Fill form fields
            for field_name, value in form_data.items():
                try:
                    # Try different selectors
                    selectors = [
                        f'input[name="{field_name}"]',
                        f'input[id="{field_name}"]',
                        f'textarea[name="{field_name}"]',
                        f'textarea[id="{field_name}"]',
                        f'select[name="{field_name}"]',
                        f'select[id="{field_name}"]'
                    ]
                    
                    for selector in selectors:
                        try:
                            element = await self.playwright_page.query_selector(selector)
                            if element:
                                await element.fill(value)
                                break
                        except:
                            continue
                            
                except Exception as e:
                    logger.warning(f"Failed to fill field {field_name}: {e}")
                    continue
            
            # Try to submit the form
            try:
                submit_button = await self.playwright_page.query_selector('input[type="submit"], button[type="submit"], button:has-text("Submit")')
                if submit_button:
                    await submit_button.click()
                    await self.playwright_page.wait_for_load_state('networkidle')
            except:
                pass  # Form might not have a submit button
            
            return {
                'success': True,
                'url': url,
                'form_data': form_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Form filling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def extract_structured_data(self, url: str, schema: Dict[str, str]) -> Dict[str, Any]:
        """Extract structured data from a webpage using CSS selectors."""
        try:
            if not await self.initialize_playwright():
                return {'success': False, 'error': 'Playwright not available'}
            
            await self.playwright_page.goto(url, wait_until='networkidle')
            
            extracted_data = {}
            
            for field_name, selector in schema.items():
                try:
                    element = await self.playwright_page.query_selector(selector)
                    if element:
                        text_content = await element.text_content()
                        extracted_data[field_name] = text_content.strip() if text_content else ''
                    else:
                        extracted_data[field_name] = ''
                except Exception as e:
                    logger.warning(f"Failed to extract {field_name}: {e}")
                    extracted_data[field_name] = ''
            
            return {
                'success': True,
                'url': url,
                'extracted_data': extracted_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def monitor_page_changes(self, url: str, interval: int = 5, duration: int = 60) -> Dict[str, Any]:
        """Monitor a webpage for changes over time."""
        try:
            if not await self.initialize_playwright():
                return {'success': False, 'error': 'Playwright not available'}
            
            await self.playwright_page.goto(url, wait_until='networkidle')
            
            initial_content = await self.playwright_page.content()
            changes = []
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < duration:
                await asyncio.sleep(interval)
                
                current_content = await self.playwright_page.content()
                if current_content != initial_content:
                    changes.append({
                        'timestamp': datetime.now().isoformat(),
                        'change_detected': True,
                        'content_length': len(current_content)
                    })
                    initial_content = current_content
                else:
                    changes.append({
                        'timestamp': datetime.now().isoformat(),
                        'change_detected': False
                    })
            
            return {
                'success': True,
                'url': url,
                'monitoring_duration': duration,
                'changes_detected': len([c for c in changes if c['change_detected']]),
                'changes': changes,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Page monitoring failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get browser manager status."""
        return {
            "browser_available": self.browser_available,
            "is_cli": self.is_cli,
            "is_streamlit": self.is_streamlit,
            "is_background": self.is_background,
            "browser_use_available": BROWSER_USE_AVAILABLE
        }
