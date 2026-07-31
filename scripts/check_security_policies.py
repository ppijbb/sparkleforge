#!/usr/bin/env python3
"""
check_security_policies.py — Layer 2 Security & Policy Audit Gate

Validates code security policies:
1. No hardcoded secret keys or credentials in tracked codebase.
2. Dangerous patterns in core code paths (e.g. unsafe builtin eval/exec calls).
"""

import os
import re
import sys
import subprocess

def check_hardcoded_keys() -> bool:
    print("[Layer 2 Security Audit] Checking hardcoded secrets...")
    res = subprocess.run([sys.executable, "scripts/check_no_hardcoded_supabase_keys.py"])
    return res.returncode == 0

def check_unsafe_evals() -> bool:
    print("[Layer 2 Security Audit] Auditing unsafe eval/exec usage in src/core...")
    core_dir = "src/core"
    if not os.path.exists(core_dir):
        return True

    violations = []
    # Match standalone builtin eval(...) or exec(...) calls, not method calls like obj.exec(...)
    pattern = re.compile(r'(?<!\.)\b(eval|exec)\s*\(')

    for root, _, files in os.walk(core_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue
                        if pattern.search(stripped):
                            # Ignore safe comments or suppressed lines (noqa / nosec)
                            if any(k in stripped for k in ["noqa", "nosec", "safeeval"]):
                                continue
                            violations.append((path, i, stripped))

    if violations:
        print(f"❌ Found {len(violations)} unsafe eval/exec pattern(s) in src/core:")
        for path, line_no, content in violations:
            print(f"  - {path}:{line_no}: {content}")
        return False

    print("✅ No unsafe eval/exec patterns found in src/core.")
    return True

def main():
    print("=== Layer 2 Security & Policy Audit ===")
    ok_keys = check_hardcoded_keys()
    ok_evals = check_unsafe_evals()

    if not (ok_keys and ok_evals):
        print("❌ Layer 2 Security Audit FAILED.")
        sys.exit(1)

    print("✅ Layer 2 Security Audit PASSED.")
    sys.exit(0)

if __name__ == "__main__":
    main()
