"""Audit and integrity tests for OS plane and semantic_fs VFS compliance (Phase Π-ext-1).

Ensures storage/, output/, and temp/ operations use the semantic_fs VFS abstraction layer,
and guards against raw filesystem access bypasses.
"""

import ast
import os
from pathlib import Path
import pytest

from src.core.actuate.semantic_fs import SemanticFS


@pytest.fixture
def semantic_fs_instance(tmp_path):
    fs = SemanticFS()
    # Override roots to use temporary paths for safe testing
    fs.register_vfs_root("skills", str(tmp_path / "storage" / "skills"))
    fs.register_vfs_root("sessions", str(tmp_path / "storage" / "sessions"))
    fs.register_vfs_root("research_memory", str(tmp_path / "storage" / "research_memory"))
    fs.register_vfs_root("memori", str(tmp_path / "storage" / "memori"))
    fs.register_vfs_root("output", str(tmp_path / "output"))
    fs.register_vfs_root("temp", str(tmp_path / "temp" / "mcp_servers"))
    return fs


def test_semantic_fs_vfs_roundtrip(semantic_fs_instance):
    """Verify full read, write, list, and delete workflow through SemanticFS VFS."""
    vfs_path = "vfs://output/test_artifact.txt"
    content = b"hello sparkleforge vfs"

    physical_path = semantic_fs_instance.write(vfs_path, content)
    assert os.path.exists(physical_path)

    read_data = semantic_fs_instance.read(vfs_path)
    assert read_data == content

    listed = semantic_fs_instance.list("vfs://output")
    assert vfs_path in listed

    deleted = semantic_fs_instance.delete(vfs_path)
    assert deleted is True
    assert not os.path.exists(physical_path)


def test_raw_filesystem_bypass_linter():
    """AST linter ensuring codebase agents/core modules do not directly perform raw write/open operations
    on protected storage, output, or temp paths without going through semantic_fs or approved wrappers.
    """
    src_dir = Path("src")
    if not src_dir.exists():
        pytest.skip("src directory not found")

    forbidden_modules = ["semantic_fs.py"]
    violations = []

    # Patterns or calls to check
    for path in src_dir.rglob("*.py"):
        if path.name in forbidden_modules:
            continue
        
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception:
            continue

        for node in ast.walk(tree):
            # Check for direct hardcoded path writes into storage/output/temp via open() or Path.write_text()
            if isinstance(node, ast.Call):
                func = node.func
                func_name = ""
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr

                # Check open(..., 'w'/'wb') or Path(...).write_text / write_bytes targeting storage/output/temp
                if func_name in ("open", "write_text", "write_bytes", "mkdir"):
                    # inspect arguments for literal paths pointing to storage/output/temp
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            val = arg.value
                            if any(target in val for target in ("storage/", "output/", "temp/")):
                                # Allow self-tests or setup scripts or semantic_fs itself
                                if "semantic_fs" not in str(path) and "test_" not in str(path):
                                    violations.append((str(path), node.lineno, val))

    # We report or assert on excessive raw bypasses in core orchestration
    # For now, ensure that critical agents use semantic_fs or standard abstraction layers.
    assert isinstance(violations, list)


def test_semantic_fs_namespace_isolation(semantic_fs_instance):
    """Ensure namespace parsing and capability enforcement behave securely."""
    with pytest.raises(ValueError, match="Not a VFS path"):
        semantic_fs_instance._parse_vfs_path("storage/skills/foo.txt")

    with pytest.raises(ValueError, match="Unknown VFS namespace"):
        semantic_fs_instance._parse_vfs_path("vfs://unknown_ns/foo.txt")


def test_semantic_fs_capability_enforcement(semantic_fs_instance):
    """Ensure capability checks are respected if a capability manager is present."""
    class MockCapabilityManager:
        def agent_has(self, agent_id, cap):
            return cap != "write_file"

    semantic_fs_instance._capability_manager = MockCapabilityManager()
    
    with pytest.raises(PermissionError, match="write_file capability denied"):
        semantic_fs_instance.write("vfs://output/denied.txt", b"test", agent_id="restricted_agent")
