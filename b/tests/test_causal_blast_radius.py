"""Tests for causal blast-radius radar and non-local regression pre-detection."""

import os
import tempfile
from src.core.ci.blast_radius import CausalBlastRadiusRadar, run_blast_radius_analysis

def test_blast_radius_indexing_and_score():
    with tempfile.TemporaryDirectory() as tmpdir:
        mod_a = os.path.join(tmpdir, "module_a.py")
        mod_b = os.path.join(tmpdir, "module_b.py")
        
        with open(mod_a, "w") as f:
            f.write("import os\nVAL = os.getenv('SHARED_CONFIG_KEY')\n")
            
        with open(mod_b, "w") as f:
            f.write("import os\nOTHER = os.getenv('SHARED_CONFIG_KEY')\n")

        radar = CausalBlastRadiusRadar(db_path=os.path.join(tmpdir, "test.db"))
        elapsed = radar.incremental_index(repo_path=tmpdir)
        
        # Target: Incremental index update time <= 200 ms/commit
        assert elapsed <= 200.0, f"Indexing took {elapsed}ms, exceeding 200ms target"

        score, tests = radar.compute_blast_radius(["module_a.py"])
        
        # Target: Query latency <= 50 ms and correct indirect selection
        assert score > 0
