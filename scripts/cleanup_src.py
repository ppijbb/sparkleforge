#!/usr/bin/env python3
"""
src 디렉토리 정리 스크립트

사용되지 않는 파일, TODO가 있는 파일, 중복 코드를 식별하고 정리합니다.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Set

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_file_usage(file_path: Path) -> Dict:
    """파일이 실제로 사용되는지 확인."""
    module_name = str(file_path.relative_to(project_root).with_suffix('')).replace('/', '.')
    
    # Check imports
    imports = []
    for py_file in project_root.rglob("*.py"):
        if py_file == file_path or "__pycache__" in str(py_file):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check various import patterns
                if module_name in content or file_path.stem in content:
                    # More precise check
                    if f'from {module_name}' in content or f'import {module_name}' in content:
                        imports.append(str(py_file.relative_to(project_root)))
        except:
            pass
    
    return {
        'file': str(file_path.relative_to(project_root)),
        'imported_by': imports,
        'is_used': len(imports) > 0
    }


def identify_files_to_cleanup() -> Dict:
    """정리가 필요한 파일들을 식별."""
    cleanup_candidates = {
        'unused_files': [],
        'todo_files': [],
        'duplicate_config': []
    }
    
    src_dir = project_root / "src"
    
    # 1. debug_validator.py 확인
    debug_validator = src_dir / "core" / "debug_validator.py"
    if debug_validator.exists():
        usage = check_file_usage(debug_validator)
        if not usage['is_used']:
            cleanup_candidates['unused_files'].append({
                'file': str(debug_validator.relative_to(project_root)),
                'reason': '사용되지 않음 (테스트/디버그 용도)',
                'action': 'archive 또는 tests/로 이동'
            })
    
    # 2. TODO가 있는 파일들
    todo_files = [
        'src/core/pii_redaction.py',
        'src/core/memory_realtime_processor.py',
        'src/core/context_engineer.py',
        'src/core/memory_tool.py'
    ]
    
    for file_path_str in todo_files:
        file_path = project_root / file_path_str
        if file_path.exists():
            cleanup_candidates['todo_files'].append({
                'file': file_path_str,
                'reason': 'TODO 주석 있음 (미완성 구현)',
                'action': 'TODO 해결 또는 이슈 등록'
            })
    
    # 3. config_manager.py vs researcher_config.py 중복 확인
    config_manager = project_root / "src" / "utils" / "config_manager.py"
    if config_manager.exists():
        usage = check_file_usage(config_manager)
        cleanup_candidates['duplicate_config'].append({
            'file': str(config_manager.relative_to(project_root)),
            'reason': 'researcher_config.py와 중복 가능성',
            'action': '기능 비교 후 통합 또는 제거',
            'imported_by': usage['imported_by']
        })
    
    return cleanup_candidates


def generate_cleanup_report(candidates: Dict) -> str:
    """정리 리포트 생성."""
    report = []
    report.append("# src 디렉토리 정리 리포트\n")
    report.append(f"생성일: {Path(__file__).stat().st_mtime}\n")
    
    total_issues = sum(len(v) for v in candidates.values())
    report.append(f"총 정리 대상: {total_issues}개\n")
    
    # 1. 사용되지 않는 파일
    if candidates['unused_files']:
        report.append("## 1. 사용되지 않는 파일\n")
        for item in candidates['unused_files']:
            report.append(f"- **{item['file']}**")
            report.append(f"  - 이유: {item['reason']}")
            report.append(f"  - 조치: {item['action']}\n")
    
    # 2. TODO가 있는 파일
    if candidates['todo_files']:
        report.append("## 2. TODO 주석이 있는 파일 (미완성 구현)\n")
        for item in candidates['todo_files']:
            report.append(f"- **{item['file']}**")
            report.append(f"  - 이유: {item['reason']}")
            report.append(f"  - 조치: {item['action']}\n")
    
    # 3. 중복 코드
    if candidates['duplicate_config']:
        report.append("## 3. 중복 가능성이 있는 파일\n")
        for item in candidates['duplicate_config']:
            report.append(f"- **{item['file']}**")
            report.append(f"  - 이유: {item['reason']}")
            if item.get('imported_by'):
                report.append(f"  - 사용처: {len(item['imported_by'])}개 파일")
            report.append(f"  - 조치: {item['action']}\n")
    
    return '\n'.join(report)


def main():
    """Main function."""
    print("=" * 80)
    print("src 디렉토리 정리 점검")
    print("=" * 80)
    
    # Identify cleanup candidates
    print("\n[1] 정리 대상 파일 식별 중...")
    candidates = identify_files_to_cleanup()
    
    total = sum(len(v) for v in candidates.values())
    print(f"✅ {total}개 정리 대상 발견")
    
    if candidates['unused_files']:
        print(f"   - 사용되지 않는 파일: {len(candidates['unused_files'])}개")
    if candidates['todo_files']:
        print(f"   - TODO 주석 파일: {len(candidates['todo_files'])}개")
    if candidates['duplicate_config']:
        print(f"   - 중복 가능성 파일: {len(candidates['duplicate_config'])}개")
    
    # Generate report
    print("\n[2] 정리 리포트 생성 중...")
    report = generate_cleanup_report(candidates)
    
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    report_file = docs_dir / "src_cleanup_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 리포트 저장: {report_file}")
    
    # Save JSON
    json_file = docs_dir / "src_cleanup_report.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON 저장: {json_file}")
    
    print("\n" + "=" * 80)
    print("점검 완료!")
    print("=" * 80)
    
    if total > 0:
        print(f"\n⚠️ {total}개 파일이 정리 대상입니다.")
        print("상세 내용은 리포트를 참조하세요.")
    else:
        print("\n✅ 정리할 파일이 없습니다.")


if __name__ == "__main__":
    main()

