import sys
import os
import importlib.util

def test_bootstrap_non_editable(tmp_path):
    # Simulate non-editable install layout where __file__ is in site-packages
    # and no 'src' directory exists alongside it.
    bootstrap_file = tmp_path / "sparkleforge_bootstrap.py"
    bootstrap_content = """
import sys
import os

def _bootstrap_sys_path():
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")
    if not os.path.isdir(src_path):
        return
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

_bootstrap_sys_path()
"""
    bootstrap_file.write_text(bootstrap_content)
    
    initial_sys_path = list(sys.path)
    # Executing the bootstrap script should not add anything when src/ is missing
    exec(compile(bootstrap_file.read_text(), str(bootstrap_file), 'exec'), {})
    assert sys.path == initial_sys_path
