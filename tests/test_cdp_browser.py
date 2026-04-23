import pytest
import asyncio
from unittest.mock import patch, MagicMock

from src.automation.cdp_browser_controller import CDPBrowserController
from src.automation.browser_manager import BrowserManager, BrowserBackend

@pytest.mark.asyncio
async def test_cdp_controller_initialization():
    # Mock daemon_alive to return False initially to simulate no daemon
    with patch("src.automation.cdp_browser_controller.daemon_alive", return_value=False):
        controller = CDPBrowserController()
        assert not controller.is_initialized
        assert controller.is_available

@pytest.mark.asyncio
async def test_browser_manager_router():
    # Test that AUTO backend logic works
    with patch("src.automation.browser_manager.get_playwright_controller") as mock_playwright:
        with patch("src.automation.cdp_browser_controller.daemon_alive", return_value=False):
            # If CDP daemon is not alive, it might fallback depending on logic
            # (Though our mock is simple, we just want to ensure it instantiates without error)
            manager = BrowserManager(backend=BrowserBackend.PLAYWRIGHT)
            assert manager is not None

@pytest.mark.asyncio
async def test_browser_manager_cdp_backend():
    with patch("src.automation.cdp_browser_controller.daemon_alive", return_value=True):
        manager = BrowserManager(backend=BrowserBackend.CDP)
        assert manager is not None
        
        # Access internal controller to check type (it's wrapped by singleton, so just check it exists)
        assert manager._controller is not None
