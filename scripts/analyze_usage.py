#!/usr/bin/env python3
"""
실제 사용 여부 분석 스크립트

import 패턴을 분석하여 실제로 사용되는 파일과 사용되지 않는 파일을 식별합니다.
"""

import sys
import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure docs directory exists
docs_dir = project_root / "docs"
docs_dir.mkdir(exist_ok=True)


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory."""
    python_files = []
    for path in directory.rglob("*.py"):
        if "__pycache__" not in str(path) and ".pyc" not in str(path):
            python_files.append(path)
    return python_files


def get_module_path(file_path: Path) -> str:
    """Get module path from file path."""
    rel_path = file_path.relative_to(project_root)
    module_parts = str(rel_path.with_suffix('')).split('/')
    module_parts = [p for p in module_parts if p]
    return '.'.join(module_parts)


def parse_imports_ast(file_path: Path) -> Set[str]:
    """Parse imports using AST."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return imports


def analyze_import_patterns(src_dir: Path) -> Dict[str, Dict]:
    """Analyze import patterns across all files."""
    python_files = find_python_files(src_dir)
    import_stats = defaultdict(lambda: {'imported_by': [], 'imports': []})
    
    print(f"Analyzing {len(python_files)} Python files...")
    
    # Build module to file mapping
    module_to_file = {}
    for file_path in python_files:
        module_path = get_module_path(file_path)
        module_to_file[module_path] = file_path
    
    # Analyze each file
    for file_path in python_files:
        module_path = get_module_path(file_path)
        imports = parse_imports_ast(file_path)
        
        import_stats[module_path]['imports'] = list(imports)
        
        # Check which modules import this one
        for imported_module in imports:
            # Try to resolve the import
            if imported_module.startswith('src.'):
                # Direct import
                imported_module_clean = imported_module
            elif '.' in imported_module:
                # Try to resolve relative imports
                parts = module_path.split('.')
                if len(parts) > 1:
                    # Try parent module
                    parent_module = '.'.join(parts[:-1])
                    potential_module = f"{parent_module}.{imported_module}"
                    if potential_module in module_to_file:
                        imported_module_clean = potential_module
                    else:
                        imported_module_clean = imported_module
                else:
                    imported_module_clean = imported_module
            else:
                imported_module_clean = imported_module
            
            if imported_module_clean in import_stats:
                import_stats[imported_module_clean]['imported_by'].append(module_path)
    
    return dict(import_stats)


def analyze_orchestrator_usage(import_stats: Dict[str, Dict]) -> Dict:
    """Analyze usage of agent_orchestrator vs autonomous_orchestrator."""
    agent_orch = 'src.core.agent_orchestrator'
    auto_orch = 'src.core.autonomous_orchestrator'
    
    result = {
        'agent_orchestrator': {
            'imported_by': import_stats.get(agent_orch, {}).get('imported_by', []),
            'imports': import_stats.get(agent_orch, {}).get('imports', [])
        },
        'autonomous_orchestrator': {
            'imported_by': import_stats.get(auto_orch, {}).get('imported_by', []),
            'imports': import_stats.get(auto_orch, {}).get('imports', [])
        }
    }
    
    return result


def find_unused_files(import_stats: Dict[str, Dict], src_dir: Path) -> List[str]:
    """Find files that are never imported."""
    python_files = find_python_files(src_dir)
    unused = []
    
    for file_path in python_files:
        module_path = get_module_path(file_path)
        
        # Skip test files, scripts, and __init__.py
        if 'test' in module_path.lower() or 'script' in module_path.lower():
            continue
        if file_path.name == '__init__.py':
            continue
        
        # Check if module is imported
        imported_by = import_stats.get(module_path, {}).get('imported_by', [])
        
        # Also check if it's a main entry point
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if '__main__' in content or 'if __name__' in content:
                    # Might be an entry point
                    continue
        except:
            pass
        
        if not imported_by:
            unused.append(module_path)
    
    return unused


def check_main_py_usage() -> Dict:
    """Check how orchestrators are used in main.py."""
    main_py = project_root / "main.py"
    
    if not main_py.exists():
        return {'error': 'main.py not found'}
    
    with open(main_py, 'r', encoding='utf-8') as f:
        content = f.read()
    
    result = {
        'agent_orchestrator_imported': 'AgentOrchestrator' in content or 'agent_orchestrator' in content,
        'autonomous_orchestrator_imported': 'AutonomousOrchestrator' in content or 'autonomous_orchestrator' in content,
        'agent_orchestrator_used': False,
        'autonomous_orchestrator_used': False
    }
    
    # Check actual usage
    if 'NewAgentOrchestrator()' in content or 'AgentOrchestrator()' in content:
        result['agent_orchestrator_used'] = True
    
    if 'AutonomousOrchestrator()' in content:
        result['autonomous_orchestrator_used'] = True
    
    # Extract import lines
    import_lines = [line for line in content.split('\n') if 'import' in line and 'orchestrator' in line.lower()]
    result['import_lines'] = import_lines
    
    return result


def generate_usage_report(import_stats: Dict, unused_files: List[str], main_usage: Dict) -> str:
    """Generate markdown usage report."""
    report = []
    report.append("# 실제 사용 여부 분석 리포트\n")
    report.append(f"생성일: {Path(__file__).stat().st_mtime}\n")
    
    report.append("## 1. main.py에서의 사용 패턴\n")
    report.append("```")
    report.append(f"agent_orchestrator imported: {main_usage.get('agent_orchestrator_imported', False)}")
    report.append(f"autonomous_orchestrator imported: {main_usage.get('autonomous_orchestrator_imported', False)}")
    report.append(f"agent_orchestrator used: {main_usage.get('agent_orchestrator_used', False)}")
    report.append(f"autonomous_orchestrator used: {main_usage.get('autonomous_orchestrator_used', False)}")
    report.append("```\n")
    
    if main_usage.get('import_lines'):
        report.append("### Import 라인:")
        for line in main_usage['import_lines']:
            report.append(f"- `{line.strip()}`")
        report.append("")
    
    report.append("## 2. Orchestrator 사용 패턴\n")
    
    orch_usage = analyze_orchestrator_usage(import_stats)
    
    report.append("### agent_orchestrator")
    report.append(f"- Imported by: {len(orch_usage['agent_orchestrator']['imported_by'])} files")
    if orch_usage['agent_orchestrator']['imported_by']:
        for importer in orch_usage['agent_orchestrator']['imported_by']:
            report.append(f"  - `{importer}`")
    report.append("")
    
    report.append("### autonomous_orchestrator")
    report.append(f"- Imported by: {len(orch_usage['autonomous_orchestrator']['imported_by'])} files")
    if orch_usage['autonomous_orchestrator']['imported_by']:
        for importer in orch_usage['autonomous_orchestrator']['imported_by']:
            report.append(f"  - `{importer}`")
    report.append("")
    
    report.append("## 3. 사용되지 않는 파일 (후보)\n")
    report.append(f"총 {len(unused_files)}개 파일이 다른 모듈에서 import되지 않습니다.\n")
    report.append("⚠️ 주의: 이 파일들은 실제로 사용되지 않을 수 있지만, 다음 경우에는 사용될 수 있습니다:\n")
    report.append("- Entry point (main 함수 포함)\n")
    report.append("- 테스트 파일\n")
    report.append("- 동적 import\n")
    report.append("- 외부에서 직접 호출\n\n")
    
    if unused_files:
        for unused in sorted(unused_files):
            report.append(f"- `{unused}`")
        report.append("")
    
    report.append("## 4. 통계\n")
    report.append(f"- 총 분석된 모듈: {len(import_stats)}\n")
    report.append(f"- 사용되지 않는 파일 후보: {len(unused_files)}\n")
    
    return '\n'.join(report)


def main():
    """Main function."""
    print("=" * 80)
    print("Usage Analysis")
    print("=" * 80)
    
    src_dir = project_root / "src"
    
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return
    
    # Analyze import patterns
    print("\n[1] Analyzing import patterns...")
    import_stats = analyze_import_patterns(src_dir)
    print(f"✅ Analyzed {len(import_stats)} modules")
    
    # Check main.py usage
    print("\n[2] Checking main.py usage...")
    main_usage = check_main_py_usage()
    print("✅ Main.py analysis complete")
    print(f"   - agent_orchestrator used: {main_usage.get('agent_orchestrator_used', False)}")
    print(f"   - autonomous_orchestrator used: {main_usage.get('autonomous_orchestrator_used', False)}")
    
    # Find unused files
    print("\n[3] Finding potentially unused files...")
    unused_files = find_unused_files(import_stats, src_dir)
    print(f"✅ Found {len(unused_files)} potentially unused files")
    
    # Analyze orchestrator usage
    print("\n[4] Analyzing orchestrator usage patterns...")
    orch_usage = analyze_orchestrator_usage(import_stats)
    print(f"   - agent_orchestrator imported by: {len(orch_usage['agent_orchestrator']['imported_by'])} files")
    print(f"   - autonomous_orchestrator imported by: {len(orch_usage['autonomous_orchestrator']['imported_by'])} files")
    
    # Generate report
    print("\n[5] Generating usage report...")
    report = generate_usage_report(import_stats, unused_files, main_usage)
    
    report_file = docs_dir / "usage_analysis.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Usage report saved to: {report_file}")
    
    # Save JSON data
    json_data = {
        'import_stats': {k: {'imported_by': v['imported_by'], 'imports': v['imports']} 
                        for k, v in import_stats.items()},
        'unused_files': unused_files,
        'main_usage': main_usage,
        'orchestrator_usage': orch_usage
    }
    
    json_file = docs_dir / "usage_analysis.json"
    import json as json_module
    with open(json_file, 'w', encoding='utf-8') as f:
        json_module.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Usage data saved to: {json_file}")
    
    print("\n" + "=" * 80)
    print("✅ Usage analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

