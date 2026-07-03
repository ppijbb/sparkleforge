import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.core.trust_gate import TrustContext, TrustLevel
from src.core.session.remote_session import WebSocketRemoteSession, SSHRemoteSession
from src.core.actuate.shell_executor import SecureShellExecutor


# --- 1. TrustContext Serialization Tests ---

def test_trust_context_serialization():
    original = TrustContext(
        level=TrustLevel.PARTIAL,
        deny_names=frozenset(["rm", "chmod"]),
        deny_prefixes=("sudo",),
        allowed_mcp_servers=frozenset(["server-1"]),
    )
    serialized = original.to_dict()
    assert serialized["level"] == "partial"
    assert "rm" in serialized["deny_names"]
    assert "sudo" in serialized["deny_prefixes"]
    assert "server-1" in serialized["allowed_mcp_servers"]

    deserialized = TrustContext.from_dict(serialized)
    assert deserialized.level == TrustLevel.PARTIAL
    assert deserialized.deny_names == frozenset(["rm", "chmod"])
    assert deserialized.deny_prefixes == ("sudo",)
    assert deserialized.allowed_mcp_servers == frozenset(["server-1"])


# --- 2. WebSocketRemoteSession Tests ---

@pytest.mark.asyncio
async def test_websocket_session_connect_and_execute():
    url = "ws://localhost:8765"
    mock_ws = AsyncMock()
    mock_ws.closed = False
    
    # Mocking response
    mock_response = {
        "stdout": "hello remote",
        "stderr": "",
        "returncode": 0,
        "status": "success"
    }
    mock_ws.recv.return_value = json.dumps(mock_response)

    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_ws
        
        session = WebSocketRemoteSession(url)
        connected = await session.connect()
        
        assert connected is True
        assert session.is_connected is True
        
        # Execute command
        result = await session.execute("echo 'hello remote'")
        assert result["status"] == "success"
        assert result["stdout"] == "hello remote"
        
        # Verify sent payload
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["action"] == "execute"
        assert sent_data["command"] == "echo 'hello remote'"
        assert "level" in sent_data["trust_context"]
        
        await session.disconnect()


@pytest.mark.asyncio
async def test_websocket_session_backoff_reconnect():
    url = "ws://localhost:8765"
    mock_ws = AsyncMock()
    mock_ws.closed = False
    
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        # First connection fails, second succeeds
        mock_connect.side_effect = [Exception("Connection refused"), mock_ws]
        
        # Speed up retry backoff for testing
        session = WebSocketRemoteSession(
            url,
            reconnect_interval_base=0.01,
            reconnect_backoff_factor=1.5,
        )
        
        # First connection should return False
        connected = await session.connect()
        assert connected is False
        assert session.is_connected is False
        
        # Give background reconnection monitor a bit of time to run
        await asyncio.sleep(0.05)
        
        # Reconnect task should run in background and connect successfully on second attempt
        assert session.is_connected is True
        
        await session.disconnect()


# --- 3. SSHRemoteSession Tests ---

@pytest.mark.asyncio
async def test_ssh_session_execution_building():
    session = SSHRemoteSession(
        host="remote-node",
        username="admin",
        key_path="/path/to/key",
    )
    
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"ssh output", b"")
    mock_proc.returncode = 0
    
    with patch("asyncio.create_subprocess_exec", return_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_proc
        
        result = await session.execute("ls -la")
        assert result["status"] == "success"
        assert result["stdout"] == "ssh output"
        
        # Verify subprocess exec args
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        
        # Check targets
        assert args[0] == "ssh"
        assert "-i" in args
        assert "/path/to/key" in args
        assert "admin@remote-node" in args
        
        # Check command wrapping with trust context environment variable
        wrapped_command = args[-1]
        assert "export SPARKLEFORGE_TRUST_CONTEXT=" in wrapped_command
        assert "ls -la" in wrapped_command


@pytest.mark.asyncio
async def test_teleport_ssh_session_execution():
    session = SSHRemoteSession(
        host="tele-host",
        use_teleport=True,
        teleport_node="tele-node-name",
    )
    
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"tsh output", b"")
    mock_proc.returncode = 0
    
    with patch("asyncio.create_subprocess_exec", return_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_proc
        
        result = await session.execute("whoami")
        assert result["status"] == "success"
        assert result["stdout"] == "tsh output"
        
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        
        assert args[0] == "tsh"
        assert args[1] == "ssh"
        assert args[2] == "tele-node-name"
        assert "whoami" in args[3]


# --- 4. SecureShellExecutor Adapter Routing Tests ---

@pytest.mark.asyncio
async def test_executor_delegates_to_remote_session():
    mock_session = AsyncMock()
    mock_session.execute.return_value = {
        "stdout": "remote exec",
        "stderr": "",
        "returncode": 0,
        "status": "success"
    }
    
    executor = SecureShellExecutor(remote_session=mock_session)
    
    result = await executor.run_command("uptime")
    assert result["status"] == "success"
    assert result["stdout"] == "remote exec"
    
    # Assert that command was routed to the remote session
    mock_session.execute.assert_called_once_with("uptime", timeout=30.0)


@pytest.mark.asyncio
async def test_executor_still_applies_blacklist_on_remote_command():
    mock_session = AsyncMock()
    executor = SecureShellExecutor(
        blacklist=["rm -rf"],
        remote_session=mock_session,
    )
    
    # This command should be blocked locally without calling remote execute
    result = await executor.run_command("rm -rf /tmp")
    assert result["status"] == "blocked"
    assert "blocked" in result["stderr"].lower()
    
    mock_session.execute.assert_not_called()
