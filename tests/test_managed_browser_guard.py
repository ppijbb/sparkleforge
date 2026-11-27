"""
test_managed_browser_guard.py — Verifies GuardPlane policies on ManagedBrowserRuntime
and complete audit coverage in ActionJournal.
"""
import asyncio
import pytest

from src.core.actuate.managed_browser import ManagedBrowserRuntime
from src.core.guard.action_journal import ActionJournal
from src.core.guard.capability_manager import CapabilityManager


@pytest.mark.asyncio
async def test_disallowed_origin_blocking():
    browser = ManagedBrowserRuntime(allowed_domains=["example.com", "trusted.org"])
    
    # 1. Allowed domain navigation succeeds
    res_ok = await browser.navigate("https://example.com/page")
    assert res_ok["success"] is True

    # 2. Disallowed domain navigation is 100% blocked
    res_blocked = await browser.navigate("https://malicious-phishing.io/login")
    assert res_blocked["success"] is False
    assert "blocked" in res_blocked["error"].lower()


@pytest.mark.asyncio
async def test_payment_and_credential_hard_blocks():
    browser = ManagedBrowserRuntime(allowed_domains=["example.com"])
    
    # Payment URL navigation block
    res_pay = await browser.navigate("https://example.com/checkout/billing")
    assert res_pay["success"] is False
    assert "blocked" in res_pay["error"].lower()

    # Sensitive form submission block
    res_form = await browser.submit_form("#login-form", {"password": "secret_password"})
    assert res_form["success"] is False
    assert "blocked" in res_form["error"].lower()


@pytest.mark.asyncio
async def test_action_journal_audit_coverage():
    journal = ActionJournal(_force_new=True)
    journal.reset()

    browser = ManagedBrowserRuntime(allowed_domains=["example.com"])
    await browser.navigate("https://example.com/dashboard")
    await browser.click_element("#submit-btn", "Click dashboard submit")

    entries = journal.recent(limit=10)
    actions = [e.action for e in entries]
    assert "navigate" in actions
    assert "click" in actions
