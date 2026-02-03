#!/usr/bin/env python3
"""
의존성 그래프 생성 및 분석 스크립트

모듈 간 의존성을 분석하고 순환 의존성을 탐지합니다.
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque

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


def parse_imports(file_path: Path) -> Set[str]:
    """Parse imports from a Python file using AST."""
    imports = set()
    try:
        import ast
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        imports.add(module)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        imports.add(module)
        except SyntaxError:
            # If AST parsing fails, fall back to simple line-based parsing
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('import '):
                    module = line[7:].split()[0].split('.')[0]
                    imports.add(module)
                elif line.startswith('from '):
                    parts = line[5:].split(' import ')
                    if len(parts) == 2:
                        module = parts[0].strip().split('.')[0]
                        imports.add(module)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return imports


def get_module_name(file_path: Path) -> str:
    """Get module name from file path."""
    # Convert to relative path from project root
    rel_path = file_path.relative_to(project_root)
    # Remove .py extension and convert to module path
    module_parts = str(rel_path.with_suffix('')).split('/')
    # Filter out empty parts
    module_parts = [p for p in module_parts if p]
    return '.'.join(module_parts)


def build_dependency_graph(src_dir: Path) -> Dict[str, Set[str]]:
    """Build dependency graph from source files."""
    graph = defaultdict(set)
    python_files = find_python_files(src_dir)
    
    print(f"Analyzing {len(python_files)} Python files...")
    
    for file_path in python_files:
        module_name = get_module_name(file_path)
        imports = parse_imports(file_path)
        
        # Filter to only internal modules (starting with 'src')
        internal_imports = set()
        for imp in imports:
            if imp.startswith('src'):
                internal_imports.add(imp)
            # Also check for relative imports that resolve to src modules
            elif '.' in imp and any(part.startswith('src') for part in imp.split('.')):
                internal_imports.add(imp)
        
        if internal_imports:
            graph[module_name] = internal_imports
    
    return dict(graph)


def detect_circular_dependencies(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Detect circular dependencies using DFS."""
    cycles = []
    visited = set()
    rec_stack = set()
    path = []
    
    def dfs(node: str):
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return
        
        if node in visited:
            return
        
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, set()):
            dfs(neighbor)
        
        rec_stack.remove(node)
        path.pop()
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return cycles


def analyze_specific_modules(graph: Dict[str, Set[str]]):
    """Analyze dependencies for specific important modules."""
    important_modules = [
        'src.core.agent_orchestrator',
        'src.core.autonomous_orchestrator',
        'src.core.mcp_integration'
    ]
    
    print("\n" + "=" * 80)
    print("Important Modules Dependency Analysis")
    print("=" * 80)
    
    for module in important_modules:
        if module in graph:
            deps = graph[module]
            print(f"\n{module}:")
            print(f"  Dependencies: {len(deps)}")
            for dep in sorted(deps):
                print(f"    - {dep}")
        else:
            print(f"\n{module}: Not found in graph")


def generate_mermaid_graph(graph: Dict[str, Set[str]], output_file: Path):
    """Generate Mermaid diagram for dependencies."""
    # Limit to important modules to avoid huge graph
    important_modules = {
        'src.core.agent_orchestrator',
        'src.core.autonomous_orchestrator',
        'src.core.mcp_integration',
        'src.core.llm_manager',
        'src.core.researcher_config',
        'src.core.shared_memory',
        'src.core.skills_manager'
    }
    
    # Filter graph to important modules and their dependencies
    filtered_graph = {}
    for module, deps in graph.items():
        if module in important_modules:
            filtered_deps = {d for d in deps if d in important_modules or any(imp in d for imp in important_modules)}
            if filtered_deps:
                filtered_graph[module] = filtered_deps
    
    mermaid_content = ["graph TD"]
    
    # Add nodes
    all_nodes = set(filtered_graph.keys())
    for deps in filtered_graph.values():
        all_nodes.update(deps)
    
    for node in sorted(all_nodes):
        node_id = node.replace('.', '_').replace('-', '_')
        label = node.split('.')[-1]  # Use last part as label
        mermaid_content.append(f'    {node_id}["{label}"]')
    
    # Add edges
    for module, deps in filtered_graph.items():
        module_id = module.replace('.', '_').replace('-', '_')
        for dep in deps:
            dep_id = dep.replace('.', '_').replace('-', '_')
            mermaid_content.append(f"    {module_id} --> {dep_id}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(mermaid_content))
    
    print(f"\n✅ Mermaid graph saved to: {output_file}")


def main():
    """Main function."""
    print("=" * 80)
    print("Dependency Graph Analysis")
    print("=" * 80)
    
    src_dir = project_root / "src"
    
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return
    
    # Build dependency graph
    print("\n[1] Building dependency graph...")
    graph = build_dependency_graph(src_dir)
    print(f"✅ Found {len(graph)} modules with dependencies")
    
    # Detect circular dependencies
    print("\n[2] Detecting circular dependencies...")
    cycles = detect_circular_dependencies(graph)
    if cycles:
        print(f"⚠️ Found {len(cycles)} circular dependencies:")
        for i, cycle in enumerate(cycles, 1):
            print(f"  Cycle {i}: {' -> '.join(cycle)}")
    else:
        print("✅ No circular dependencies found")
    
    # Analyze specific modules
    print("\n[3] Analyzing important modules...")
    analyze_specific_modules(graph)
    
    # Save dependency report
    print("\n[4] Saving dependency report...")
    report = {
        "total_modules": len(graph),
        "circular_dependencies": cycles,
        "graph": {k: list(v) for k, v in graph.items()}
    }
    
    report_file = docs_dir / "dependency_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ Dependency report saved to: {report_file}")
    
    # Generate Mermaid graph
    print("\n[5] Generating Mermaid diagram...")
    mermaid_file = docs_dir / "dependency_graph.mmd"
    generate_mermaid_graph(graph, mermaid_file)
    
    # Try to generate SVG if pydeps is available
    print("\n[6] Attempting to generate SVG with pydeps...")
    try:
        import shutil
        pydeps_path = shutil.which('pydeps')
        if pydeps_path:
            svg_file = docs_dir / "dependency_graph.svg"
            result = subprocess.run(
                [pydeps_path, 'src/core/autonomous_orchestrator.py', '--max-bacon=2', '--output', str(svg_file)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root)
            )
            if result.returncode == 0 and svg_file.exists():
                print(f"✅ SVG graph saved to: {svg_file}")
            else:
                print("⚠️ pydeps execution failed. Output:", result.stderr[:200])
        else:
            print("⚠️ pydeps not found in PATH. Install with: pip install pydeps")
    except Exception as e:
        print(f"⚠️ pydeps not available: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Dependency analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

