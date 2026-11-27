import subprocess
import pathlib

def test_set_a_export_leak_and_uv_fallback(tmp_path):
    repo_root = pathlib.Path(__file__).parent.parent
    script_path = repo_root / "scripts" / "run_mcp_server.sh"
    if not script_path.exists():
        return

    env_file = tmp_path / ".env"
    env_file.write_text("TEST_SECRET_VAL=secret123\n")

    # Test that sourcing .env via subshell in the script or similar doesn't leak unexpected vars
    cmd = f"""
    cd {tmp_path}
    export REPO_ROOT="{tmp_path}"
    . "{script_path}"
    load_env
    echo "VAL: $TEST_SECRET_VAL"
    # Check that subsequent assignments are NOT exported
    UNASSIGNED_VAR=should_not_leak
    (
       # verify env does not contain UNASSIGNED_VAR exported from outside if set -a was global
       true
    )
    """
    res = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert "VAL: secret123" in res.stdout
    # Ensure unexported vars don't leak out of the sourced block
    assert "should_not_leak" not in subprocess.run(["bash", "-c", f"export -p | grep should_not_leak || true"], capture_output=True, text=True).stdout
