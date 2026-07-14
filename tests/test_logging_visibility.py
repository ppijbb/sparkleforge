import logging
import io
import sys
import pytest

def test_info_logs_are_visible_from_submodules():
    """
    Verify that INFO level logs from submodules are captured by the root logger.
    Currently, if only the __main__ logger is configured, submodules 
    (which use their own __name__) fail to output INFO logs to the console.
    """
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import main
    
    root = logging.getLogger()
    assert root.level <= logging.INFO
    
    # Setup a capture stream
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    
    # Attach to root logger
    root.addHandler(handler)
    
    submodule_logger = logging.getLogger("src.core.nightwelding.runner")
    submodule_logger.info("TEST_INFO_MESSAGE")
    
    assert "TEST_INFO_MESSAGE" in log_capture.getvalue(), "INFO log from submodule was swallowed"
