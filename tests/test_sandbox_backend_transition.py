import pytest


def test_docker_sandbox_defaults_to_runsc_runtime():
    from src.core.sandbox.docker_sandbox import DockerSandbox, SandboxConfig

    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.config = SandboxConfig()

    kwargs = sandbox._container_kwargs("python:3.11-slim", ["python", "-c", "1"], None)

    assert kwargs["runtime"] == "runsc"
    assert kwargs["network_mode"] == "none"
    assert kwargs["read_only"] is True
    assert kwargs["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in kwargs["security_opt"]


@pytest.mark.asyncio
async def test_legacy_sandbox_request_is_rejected():
    from src.core.mcp_integration import _execute_code_tool

    result = await _execute_code_tool(
        "execute_code",
        {"code": "print(1)", "language": "python", "sandbox": "legacy"},
    )

    assert result.success is False
    assert "unsupported sandbox" in str(result.error).lower()
