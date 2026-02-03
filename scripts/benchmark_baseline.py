#!/usr/bin/env python3
"""
벤치마크 기준선 수립 스크립트

실행 시간, 메모리 사용량, 코드 복잡도, 테스트 커버리지를 측정합니다.
"""

import sys
import os
import time
import json
import subprocess
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure docs directory exists
docs_dir = project_root / "docs"
docs_dir.mkdir(exist_ok=True)


def measure_file_sizes(src_dir: Path) -> Dict[str, int]:
    """Measure file sizes in lines of code."""
    file_sizes = {}
    total_lines = 0
    
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" not in str(py_file):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                rel_path = str(py_file.relative_to(project_root))
                file_sizes[rel_path] = lines
                total_lines += lines
            except Exception as e:
                print(f"Error reading {py_file}: {e}")
    
    file_sizes['_total'] = total_lines
    return file_sizes


def measure_complexity(file_path: Path) -> Dict[str, Any]:
    """Measure cyclomatic complexity of a file."""
    try:
        import ast
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        
        complexity = 0
        function_count = 0
        class_count = 0
        
        def count_complexity(node):
            nonlocal complexity, function_count, class_count
            
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                function_count += 1
                # Base complexity is 1
                func_complexity = 1
                
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                        ast.Try, ast.With, ast.AsyncWith)):
                        func_complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        func_complexity += len(child.values) - 1
                
                complexity += func_complexity
            
            elif isinstance(node, ast.ClassDef):
                class_count += 1
        
        for node in ast.walk(tree):
            count_complexity(node)
        
        return {
            'total_complexity': complexity,
            'function_count': function_count,
            'class_count': class_count,
            'avg_complexity': complexity / function_count if function_count > 0 else 0
        }
    except Exception as e:
        return {'error': str(e)}


def measure_test_coverage() -> Dict[str, Any]:
    """Measure test coverage if pytest-cov is available."""
    try:
        result = subprocess.run(
            ['python3', '-m', 'pytest', '--cov=src', '--cov-report=json', '--quiet'],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(project_root)
        )
        
        if result.returncode == 0:
            coverage_file = project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                return {
                    'available': True,
                    'total_coverage': coverage_data.get('totals', {}).get('percent_covered', 0),
                    'files_covered': len(coverage_data.get('files', {}))
                }
    except Exception as e:
        pass
    
    return {'available': False, 'reason': 'pytest-cov not available or tests failed'}


def measure_memory_usage() -> Dict[str, Any]:
    """Measure memory usage of key modules."""
    memory_info = {}
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info['current_memory_mb'] = process.memory_info().rss / 1024 / 1024
        memory_info['available_memory_mb'] = psutil.virtual_memory().available / 1024 / 1024
        memory_info['total_memory_mb'] = psutil.virtual_memory().total / 1024 / 1024
    except ImportError:
        memory_info['available'] = False
        memory_info['reason'] = 'psutil not installed'
    
    return memory_info


def benchmark_import_time() -> Dict[str, float]:
    """Benchmark import times for key modules."""
    import_times = {}
    
    key_modules = [
        'src.core.agent_orchestrator',
        'src.core.autonomous_orchestrator',
        'src.core.mcp_integration',
        'src.core.llm_manager',
        'src.core.researcher_config'
    ]
    
    for module_name in key_modules:
        try:
            start = time.time()
            __import__(module_name)
            import_time = time.time() - start
            import_times[module_name] = import_time
        except Exception as e:
            import_times[module_name] = {'error': str(e)}
    
    return import_times


def analyze_large_files(file_sizes: Dict[str, int], threshold: int = 1000) -> List[Dict[str, Any]]:
    """Analyze large files that need modularization."""
    large_files = []
    
    for file_path, lines in file_sizes.items():
        if file_path != '_total' and lines >= threshold:
            large_files.append({
                'file': file_path,
                'lines': lines,
                'priority': 'high' if lines >= 2000 else 'medium'
            })
    
    return sorted(large_files, key=lambda x: x['lines'], reverse=True)


def generate_baseline_report(
    file_sizes: Dict[str, int],
    complexity_data: Dict[str, Dict],
    test_coverage: Dict[str, Any],
    memory_info: Dict[str, Any],
    import_times: Dict[str, float],
    large_files: List[Dict[str, Any]]
) -> str:
    """Generate baseline report."""
    report = []
    report.append("# 벤치마크 기준선 리포트\n")
    report.append(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("## 1. 코드 규모\n")
    report.append(f"- 총 라인 수: {file_sizes.get('_total', 0):,}\n")
    report.append(f"- 총 파일 수: {len([f for f in file_sizes.keys() if f != '_total'])}\n")
    
    report.append("\n## 2. 큰 파일 분석 (1000+ 라인)\n")
    if large_files:
        report.append(f"총 {len(large_files)}개 파일이 1000라인 이상입니다:\n\n")
        for file_info in large_files:
            report.append(f"- **{file_info['file']}**: {file_info['lines']:,} 라인 (우선순위: {file_info['priority']})\n")
    else:
        report.append("1000라인 이상인 파일이 없습니다.\n")
    
    report.append("\n## 3. 코드 복잡도\n")
    total_complexity = sum(c.get('total_complexity', 0) for c in complexity_data.values() if isinstance(c, dict))
    total_functions = sum(c.get('function_count', 0) for c in complexity_data.values() if isinstance(c, dict))
    avg_complexity = total_complexity / total_functions if total_functions > 0 else 0
    
    report.append(f"- 총 복잡도: {total_complexity}\n")
    report.append(f"- 총 함수 수: {total_functions}\n")
    report.append(f"- 평균 복잡도: {avg_complexity:.2f}\n")
    
    # Most complex files
    complex_files = []
    for file_path, comp_data in complexity_data.items():
        if isinstance(comp_data, dict) and 'total_complexity' in comp_data:
            complex_files.append((file_path, comp_data['total_complexity']))
    
    complex_files.sort(key=lambda x: x[1], reverse=True)
    if complex_files:
        report.append("\n가장 복잡한 파일 (상위 10개):\n")
        for file_path, complexity in complex_files[:10]:
            report.append(f"- `{file_path}`: {complexity}\n")
    
    report.append("\n## 4. 테스트 커버리지\n")
    if test_coverage.get('available'):
        report.append(f"- 총 커버리지: {test_coverage.get('total_coverage', 0):.2f}%\n")
        report.append(f"- 커버된 파일 수: {test_coverage.get('files_covered', 0)}\n")
    else:
        report.append(f"- 상태: {test_coverage.get('reason', 'Unknown')}\n")
        report.append("- pytest-cov를 설치하여 커버리지를 측정할 수 있습니다.\n")
    
    report.append("\n## 5. 메모리 사용량\n")
    if memory_info.get('available') is not False:
        report.append(f"- 현재 메모리 사용: {memory_info.get('current_memory_mb', 0):.2f} MB\n")
        report.append(f"- 사용 가능 메모리: {memory_info.get('available_memory_mb', 0):.2f} MB\n")
        report.append(f"- 총 메모리: {memory_info.get('total_memory_mb', 0):.2f} MB\n")
    else:
        report.append(f"- 상태: {memory_info.get('reason', 'Unknown')}\n")
    
    report.append("\n## 6. 모듈 Import 시간\n")
    for module, import_time in import_times.items():
        if isinstance(import_time, dict):
            report.append(f"- `{module}`: Error - {import_time.get('error', 'Unknown')}\n")
        else:
            report.append(f"- `{module}`: {import_time:.4f}초\n")
    
    return '\n'.join(report)


def main():
    """Main function."""
    print("=" * 80)
    print("Benchmark Baseline Measurement")
    print("=" * 80)
    
    src_dir = project_root / "src"
    
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return
    
    # Measure file sizes
    print("\n[1] Measuring file sizes...")
    file_sizes = measure_file_sizes(src_dir)
    print(f"✅ Total lines of code: {file_sizes.get('_total', 0):,}")
    
    # Analyze large files
    print("\n[2] Analyzing large files...")
    large_files = analyze_large_files(file_sizes, threshold=1000)
    print(f"✅ Found {len(large_files)} files with 1000+ lines")
    for file_info in large_files[:5]:
        print(f"   - {file_info['file']}: {file_info['lines']:,} lines")
    
    # Measure complexity
    print("\n[3] Measuring code complexity...")
    complexity_data = {}
    key_files = [
        project_root / "src/core/agent_orchestrator.py",
        project_root / "src/core/autonomous_orchestrator.py",
        project_root / "src/core/mcp_integration.py"
    ]
    
    for file_path in key_files:
        if file_path.exists():
            rel_path = str(file_path.relative_to(project_root))
            complexity_data[rel_path] = measure_complexity(file_path)
            comp = complexity_data[rel_path]
            if 'total_complexity' in comp:
                print(f"   - {rel_path}: complexity={comp['total_complexity']}, functions={comp['function_count']}")
    
    # Measure test coverage
    print("\n[4] Measuring test coverage...")
    test_coverage = measure_test_coverage()
    if test_coverage.get('available'):
        print(f"✅ Test coverage: {test_coverage.get('total_coverage', 0):.2f}%")
    else:
        print(f"⚠️ {test_coverage.get('reason', 'Test coverage not available')}")
    
    # Measure memory
    print("\n[5] Measuring memory usage...")
    memory_info = measure_memory_usage()
    if memory_info.get('available') is not False:
        print(f"✅ Current memory: {memory_info.get('current_memory_mb', 0):.2f} MB")
    else:
        print(f"⚠️ {memory_info.get('reason', 'Memory measurement not available')}")
    
    # Benchmark imports
    print("\n[6] Benchmarking import times...")
    import_times = benchmark_import_time()
    for module, import_time in import_times.items():
        if isinstance(import_time, dict):
            print(f"   ⚠️ {module}: Error")
        else:
            print(f"   ✅ {module}: {import_time:.4f}s")
    
    # Generate report
    print("\n[7] Generating baseline report...")
    report = generate_baseline_report(
        file_sizes, complexity_data, test_coverage,
        memory_info, import_times, large_files
    )
    
    report_file = docs_dir / "benchmark_baseline.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Baseline report saved to: {report_file}")
    
    # Save JSON data
    json_data = {
        'generated_at': datetime.now().isoformat(),
        'file_sizes': file_sizes,
        'large_files': large_files,
        'complexity': complexity_data,
        'test_coverage': test_coverage,
        'memory_info': memory_info,
        'import_times': import_times
    }
    
    json_file = docs_dir / "benchmark_baseline.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Baseline data saved to: {json_file}")
    
    print("\n" + "=" * 80)
    print("✅ Benchmark baseline measurement complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

