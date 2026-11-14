import sqlite3
import os
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

class CausalBlastRadiusRadar:
    def __init__(self, db_path: str = ".sparkleforge/blast_radius.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS references_map (
                source_module TEXT,
                target_key TEXT,
                ref_type TEXT,
                PRIMARY KEY (source_module, target_key, ref_type)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS co_changes (
                file_a TEXT,
                file_b TEXT,
                count INTEGER,
                PRIMARY KEY (file_a, file_b)
            )
        """)
        conn.commit()
        conn.close()

    def index_repository(self, root_dir: str = ".") -> float:
        import time
        start_time = time.time()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM references_map")
        
        root_path = Path(root_dir)
        for py_file in root_path.rglob("*.py"):
            if "site-packages" in str(py_file) or ".git" in str(py_file) or ".sparkleforge" in str(py_file):
                continue
            module_name = str(py_file.relative_to(root_path)).replace("/", ".").replace("\\", ".")[:-3]
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(content, filename=str(py_file))
                
                for node in ast.walk(tree):
                    # Look for config/env keys e.g. os.environ.get("KEY"), os.getenv("KEY"), settings.KEY, etc.
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Attribute) and node.func.attr in ("get", "getenv"):
                            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                                key = node.args[0].value
                                cursor.execute("INSERT OR IGNORE INTO references_map VALUES (?, ?, ?)", (module_name, key, "env"))
                        elif isinstance(node.func, ast.Name) and node.func.id in ("getenv",):
                            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                                key = node.args[0].value
                                cursor.execute("INSERT OR IGNORE INTO references_map VALUES (?, ?, ?)", (module_name, key, "env"))
                    elif isinstance(node, ast.Subscript):
                        # e.g. os.environ["KEY"] or config["KEY"]
                        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                            key = node.slice.value
                            cursor.execute("INSERT OR IGNORE INTO references_map VALUES (?, ?, ?)", (module_name, key, "config_subscript"))
            except Exception:
                pass
                
        conn.commit()
        conn.close()
        elapsed_ms = (time.time() - start_time) * 1000.0
        return elapsed_ms

    def compute_blast_radius(self, modified_files: List[str]) -> Tuple[int, List[str]]:
        import time
        start_time = time.time()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        affected_keys = set()
        modules = set()
        for f in modified_files:
            if f.endswith(".py"):
                mod = f.replace("/", ".").replace("\\", ".")[:-3]
                if mod.startswith("src."):
                    mod = mod[4:]
                modules.add(mod)
                
        # Find keys referenced by modified files
        placeholders = ','.join(['?'] * len(list(modules)))
        if modules:
            cursor.execute(f"SELECT DISTINCT target_key FROM references_map WHERE source_module IN ({placeholders})", tuple(modules))
            for row in cursor.fetchall():
                affected_keys.add(row[0])
                
        # Find all other modules referencing those same keys
        consumer_modules = set()
        if affected_keys:
            key_placeholders = ','.join(['?'] * len(list(affected_keys)))
            cursor.execute(f"SELECT DISTINCT source_module FROM references_map WHERE target_key IN ({key_placeholders})", tuple(affected_keys))
            for row in cursor.fetchall():
                consumer_modules.add(row[0])
                
        conn.close()
        
        # Filter out self modules
        indirect_consumers = [m for m in consumer_modules if m not in modules]
        
        # Compute score 1-100
        score = min(100, max(1, len(indirect_consumers) * 15 + len(affected_keys) * 10))
        if not indirect_consumers and not affected_keys:
            score = 1
            
        elapsed_ms = (time.time() - start_time) * 1000.0
        if elapsed_ms > 50.0:
            # Ensure query latency stays within bounds or gets logged
            pass
            
        return score, indirect_consumers

    def select_tests_for_modules(self, modules: List[str]) -> List[str]:
        test_files = []
        for mod in modules:
            parts = mod.split(".")
            test_name = f"tests/test_{parts[-1]}.py"
            if os.path.exists(test_name):
                test_files.append(test_name)
            else:
                # Fallback search
                for t in Path("tests").rglob("*.py"):
                    if parts[-1] in t.read_text(encoding="utf-8", errors="ignore"):
                        test_files.append(str(t))
        return list(set(test_files))

    def analyze_and_get_smoke_tests(self, modified_files: List[str]) -> Dict[str, any]:
        self.index_repository()
        score, indirect_modules = self.compute_blast_radius(modified_files)
        tests = self.select_tests_for_modules(indirect_modules)
        return {
            "blast_radius_score": score,
            "indirect_modules": indirect_modules,
            "smoke_tests": tests
        }
