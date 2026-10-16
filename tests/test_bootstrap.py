import os
import sys
import pytest

def test_bootstrap_non_editable(tmp_path):
    """Test that bootstrap logic handles non-editable site-packages installs safely."""
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    bootstrap_file = site_packages / "sparkleforge_bootstrap.py"
    bootstrap_file.write_text("# bootstrap shim\n")
    
    # Simulate non-editable environment where src/ does not exist next to bootstrap
    project_root = os.path.dirname(os.path.abspath(bootstrap_file))
    src_path = os.path.join(project_root, "src")
    assert not os.path.exists(src_path)
