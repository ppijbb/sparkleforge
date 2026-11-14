import os
import tempfile
from pathlib import Path
from src.core.blast_radius import CausalBlastRadiusRadar

def test_blast_radius_radar():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        Path("src").mkdir()
        Path("src/module_a.py").write_text('import os\ndef foo():\n    return os.getenv("SHARED_CONFIG_KEY")')
        Path("src/module_b.py").write_text('import os\ndef bar():\n    return os.environ["SHARED_CONFIG_KEY"]')
        
        radar = CausalBlastRadiusRadar(db_path="test_blast.db")
        index_time = radar.index_repository(".")
        assert index_time <= 200.0
        
        score, indirect = radar.compute_blast_radius(["src/module_a.py"])
        assert score >= 1
        assert "module_b" in indirect
        
        result = radar.analyze_and_get_smoke_tests(["src/module_a.py"])
        assert "blast_radius_score" in result
        assert result["blast_radius_score"] >= 1
        
        # Clean up db
        if os.path.exists("test_blast.db"):
            os.remove("test_blast.db")

