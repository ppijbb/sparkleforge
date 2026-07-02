import asyncio
import os
import shutil
import tempfile
import pytest
from src.core.bootstrap_graph import BootstrapGraph
from src.core.actuate.actuation_plane import ActuationPlane
from src.core.actuate.shell_executor import SecureShellExecutor
from src.core.actuate.os_control import OSControl
from src.core.actuate.semantic_fs import SemanticFS
from src.core.actuate.package_manager import UnifiedPackageManager
from src.core.actuate.audit_proxy import AuditProxy
from src.core.actuate.firmware_assistant import FirmwareAssistant


@pytest.mark.asyncio
async def test_shell_executor():
    executor = SecureShellExecutor()
    
    # Test dry run
    res = await executor.run_command("echo 'hello'", dry_run=True)
    assert res["status"] == "success"
    assert "echo 'hello'" in res["stdout"]
    assert res["returncode"] == 0

    # Test real execution
    res = await executor.run_command("echo 'real run'")
    assert res["status"] == "success"
    assert "real run" in res["stdout"].strip()
    assert res["returncode"] == 0

    # Test blocked command (blacklist)
    res = await executor.run_command("rm -rf /")
    assert res["status"] == "blocked"
    assert res["returncode"] == -1
    assert "blocked" in res["stderr"].lower()

    # Test timeout
    # On Windows/Linux 'sleep 5' is a good command to test timeout
    res = await executor.run_command("sleep 5", timeout=0.1)
    assert res["status"] == "timeout"
    assert res["returncode"] == -1


@pytest.mark.asyncio
async def test_os_control():
    ctrl = OSControl()
    
    # Test screen size
    w, h = ctrl.get_screen_size()
    assert w > 0 and h > 0

    # Test actions (should return True or successfully complete fallbacks)
    assert await ctrl.click(100, 100) is True
    assert await ctrl.type_text("test typing") is True
    assert await ctrl.drag_to(200, 200) is True


@pytest.mark.asyncio
async def test_semantic_fs():
    fs = SemanticFS()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temp files
        f1 = os.path.join(tmpdir, "test1.txt")
        f2 = os.path.join(tmpdir, "test2.json")
        with open(f1, "w") as f:
            f.write("hello")
        with open(f2, "w") as f:
            f.write("{}")

        # Test index_directory
        metadata = fs.index_directory(tmpdir)
        assert len(metadata) == 2
        paths = [item["path"] for item in metadata]
        assert f1 in paths
        assert f2 in paths
        assert any(item["extension"] == ".txt" for item in metadata)
        assert any(item["extension"] == ".json" for item in metadata)

        # Test watch_directory (using fallback polling observer)
        events = []
        async def watch_callback(event_type, path):
            events.append((event_type, path))

        watch_id = await fs.watch_directory(tmpdir, watch_callback)
        assert watch_id is not None

        # Modify files to trigger events
        await asyncio.sleep(1.2) # Allow polling loop to baseline
        
        f3 = os.path.join(tmpdir, "test3.txt")
        with open(f3, "w") as f:
            f.write("new file")
            
        await asyncio.sleep(1.5) # Allow polling detection
        
        assert len(events) >= 1
        assert any(evt[0] == "created" and evt[1] == f3 for evt in events)

        # Unwatch
        assert fs.unwatch_directory(watch_id) is True


@pytest.mark.asyncio
async def test_package_manager():
    mgr = UnifiedPackageManager()

    # Test unsupported package manager
    with pytest.raises(ValueError):
        mgr._get_executable_args("invalid_manager", "install", "pkg")

    # Mock package installation behavior for non-existent tools
    # Since we cannot easily install real packages in test mode, we verify detection
    # If the manager isn't available (e.g. brew or apt on some systems), it should return False
    res = await mgr.install("brew", "nonexistent-pkg-test")
    if not shutil.which("brew"):
        assert res is False
    else:
        # If brew is installed, it may return True/False depending on exit codes, but won't crash
        pass


@pytest.mark.asyncio
async def test_audit_proxy():
    with tempfile.TemporaryDirectory() as tmpdir:
        audit_file = os.path.join(tmpdir, "network_audit.log")
        proxy = AuditProxy(audit_log_path=audit_file)

        # Test blacklisted target
        res = await proxy.request("GET", "http://malicious-target.com/leak")
        assert res["status"] == 403
        assert "blocked" in res["error"].lower()

        # Check audit log contains entries
        assert os.path.exists(audit_file)
        with open(audit_file, "r") as f:
            logs = f.read()
        assert "GET" in logs
        assert "malicious-target.com" in logs

        # Test successful request with caching (e.g. to a public site like example.com)
        # We wrapper with a try-except in case of runner offline/no network connection
        res = await proxy.request("GET", "https://example.com")
        if res["status"] != -1:
            assert res["status"] >= 200 and res["status"] < 300
            
            # Second request should hit cache
            res_cached = await proxy.request("GET", "https://example.com")
            assert res_cached == res


@pytest.mark.asyncio
async def test_firmware_assistant():
    assistant = FirmwareAssistant()

    # Test recommendations list
    updates = await assistant.recommend_updates()
    assert len(updates) > 0
    assert "id" in updates[0]
    assert "component" in updates[0]

    # Test apply update
    valid_id = updates[0]["id"]
    assert await assistant.apply_update(valid_id) is True
    assert await assistant.apply_update("nonexistent_id") is False


@pytest.mark.asyncio
async def test_actuation_plane_bootstrap():
    graph = BootstrapGraph()
    res = await graph.run()
    assert res.ok is True
    
    stages = [s.name for s in res.stages]
    assert "actuation_plane" in stages
    
    stage_res = next(s for s in res.stages if s.name == "actuation_plane")
    assert stage_res.ok is True
    assert stage_res.payload["initialized"] is True
    assert isinstance(stage_res.payload["actuation_plane"], ActuationPlane)
