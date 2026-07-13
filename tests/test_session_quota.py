import pytest
from src.core.session_control import SessionControl

def test_session_quota_initialization():
    """
    Verify that active sessions are initialized with a quota schema.
    This test fails if SessionControl does not track per-session quotas.
    """
    controller = SessionControl()
    session_id = "test_session_123"
    
    # Register a session
    controller.register_active_session(session_id)
    
    # Check if quota exists in the session state
    session_data = controller.get_session_state(session_id)
    assert "quota" in session_data, "Session state must contain a 'quota' field"
    assert session_data["quota"].get("token_limit") is not None, "Quota should have a token_limit"
