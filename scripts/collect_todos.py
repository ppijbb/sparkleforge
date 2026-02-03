#!/usr/bin/env python3
"""
TODO/FIXME 주석 수집 및 분류 스크립트

모든 TODO/FIXME 주석을 수집하고 우선순위별로 분류합니다.
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure docs directory exists
docs_dir = project_root / "docs"
docs_dir.mkdir(exist_ok=True)


class Priority(Enum):
    """TODO 우선순위."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class TodoItem:
    """TODO 항목."""
    file_path: str
    line_number: int
    content: str
    priority: Priority
    category: str
    issue_type: str  # TODO, FIXME, etc.


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory."""
    python_files = []
    for path in directory.rglob("*.py"):
        if "__pycache__" not in str(path) and ".pyc" not in str(path):
            python_files.append(path)
    return python_files


def determine_priority(content: str, file_path: str) -> Priority:
    """Determine priority based on content and context."""
    content_lower = content.lower()
    
    # Critical indicators
    critical_keywords = [
        'security', 'vulnerability', 'data loss', 'crash', 'hang',
        'memory leak', 'race condition', 'deadlock', 'critical bug',
        'production', 'breaking change'
    ]
    if any(keyword in content_lower for keyword in critical_keywords):
        return Priority.CRITICAL
    
    # High priority indicators
    high_keywords = [
        'performance', 'slow', 'optimization', 'scalability',
        'integration', 'api', 'interface', 'protocol',
        'error handling', 'exception', 'failure'
    ]
    if any(keyword in content_lower for keyword in high_keywords):
        return Priority.HIGH
    
    # Medium priority indicators
    medium_keywords = [
        'refactor', 'cleanup', 'improve', 'enhance',
        'documentation', 'test', 'validation'
    ]
    if any(keyword in content_lower for keyword in medium_keywords):
        return Priority.MEDIUM
    
    # Check file location
    if 'core' in file_path or 'orchestrator' in file_path:
        return Priority.HIGH
    
    return Priority.LOW


def categorize_todo(content: str) -> str:
    """Categorize TODO based on content."""
    content_lower = content.lower()
    
    if 'memory' in content_lower or 'storage' in content_lower:
        return 'Memory/Storage'
    elif 'rag' in content_lower or 'vector' in content_lower or 'embedding' in content_lower:
        return 'RAG/Vector'
    elif 'llm' in content_lower or 'model' in content_lower:
        return 'LLM/Model'
    elif 'mcp' in content_lower or 'tool' in content_lower:
        return 'MCP/Tools'
    elif 'test' in content_lower:
        return 'Testing'
    elif 'config' in content_lower or 'setting' in content_lower:
        return 'Configuration'
    elif 'error' in content_lower or 'exception' in content_lower:
        return 'Error Handling'
    elif 'performance' in content_lower or 'optimization' in content_lower:
        return 'Performance'
    elif 'refactor' in content_lower or 'cleanup' in content_lower:
        return 'Refactoring'
    else:
        return 'Other'


def extract_todos(file_path: Path) -> List[TodoItem]:
    """Extract TODO/FIXME comments from a file."""
    todos = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            # Match TODO, FIXME, HACK, XXX, NOTE patterns
            patterns = [
                (r'#\s*(TODO|FIXME|HACK|XXX|NOTE|BUG|WARNING):\s*(.+)', 'TODO'),
                (r'#\s*(TODO|FIXME|HACK|XXX|NOTE|BUG|WARNING)\s+(.+)', 'TODO'),
                (r'"""\s*(TODO|FIXME|HACK|XXX|NOTE|BUG|WARNING):\s*(.+)', 'TODO'),
            ]
            
            for pattern, issue_type in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    content = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    content = content.strip()
                    
                    # Determine priority and category
                    priority = determine_priority(content, str(file_path))
                    category = categorize_todo(content)
                    
                    todo = TodoItem(
                        file_path=str(file_path.relative_to(project_root)),
                        line_number=line_num,
                        content=content,
                        priority=priority,
                        category=category,
                        issue_type=issue_type
                    )
                    todos.append(todo)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return todos


def generate_inventory(todos: List[TodoItem]) -> str:
    """Generate TODO inventory markdown."""
    report = []
    report.append("# TODO/FIXME 인벤토리\n")
    report.append(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"총 TODO/FIXME 항목: {len(todos)}\n")
    
    # Group by priority
    by_priority = {}
    for priority in Priority:
        by_priority[priority] = [t for t in todos if t.priority == priority]
    
    report.append("## 우선순위별 분류\n")
    for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
        count = len(by_priority[priority])
        report.append(f"- **{priority.value}**: {count}개\n")
    
    # Group by category
    by_category = {}
    for todo in todos:
        if todo.category not in by_category:
            by_category[todo.category] = []
        by_category[todo.category].append(todo)
    
    report.append("\n## 카테고리별 분류\n")
    for category in sorted(by_category.keys()):
        count = len(by_category[category])
        report.append(f"- **{category}**: {count}개\n")
    
    # Detailed list by priority
    report.append("\n## 상세 목록 (우선순위순)\n")
    
    for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
        priority_todos = by_priority[priority]
        if not priority_todos:
            continue
        
        report.append(f"\n### {priority.value} Priority ({len(priority_todos)}개)\n")
        
        # Group by file
        by_file = {}
        for todo in priority_todos:
            if todo.file_path not in by_file:
                by_file[todo.file_path] = []
            by_file[todo.file_path].append(todo)
        
        for file_path in sorted(by_file.keys()):
            file_todos = by_file[file_path]
            report.append(f"\n#### `{file_path}`\n")
            
            for todo in file_todos:
                report.append(f"- **Line {todo.line_number}** ({todo.category}): {todo.content}\n")
    
    # Issue tracking format
    report.append("\n## 이슈 트래킹 형식\n")
    report.append("각 TODO를 GitHub Issues로 등록할 때 사용할 형식:\n\n")
    
    for todo in todos[:20]:  # Show first 20 as examples
        report.append(f"### {todo.file_path}:{todo.line_number}\n")
        report.append(f"- **Priority**: {todo.priority.value}\n")
        report.append(f"- **Category**: {todo.category}\n")
        report.append(f"- **Content**: {todo.content}\n")
        report.append(f"- **File**: `{todo.file_path}`\n")
        report.append(f"- **Line**: {todo.line_number}\n\n")
    
    if len(todos) > 20:
        report.append(f"\n... and {len(todos) - 20} more items\n")
    
    return '\n'.join(report)


def generate_json_inventory(todos: List[TodoItem]) -> Dict:
    """Generate JSON inventory for programmatic access."""
    return {
        'total_count': len(todos),
        'generated_at': datetime.now().isoformat(),
        'by_priority': {
            priority.value: [
                {
                    'file': t.file_path,
                    'line': t.line_number,
                    'content': t.content,
                    'category': t.category,
                    'issue_type': t.issue_type
                }
                for t in todos if t.priority == priority
            ]
            for priority in Priority
        },
        'by_category': {
            category: [
                {
                    'file': t.file_path,
                    'line': t.line_number,
                    'content': t.content,
                    'priority': t.priority.value,
                    'issue_type': t.issue_type
                }
                for t in todos if t.category == category
            ]
            for category in set(t.category for t in todos)
        },
        'all_items': [
            {
                'file': t.file_path,
                'line': t.line_number,
                'content': t.content,
                'priority': t.priority.value,
                'category': t.category,
                'issue_type': t.issue_type
            }
            for t in todos
        ]
    }


def main():
    """Main function."""
    print("=" * 80)
    print("TODO/FIXME Collection and Classification")
    print("=" * 80)
    
    src_dir = project_root / "src"
    
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return
    
    # Find all Python files
    print("\n[1] Finding Python files...")
    python_files = find_python_files(src_dir)
    print(f"✅ Found {len(python_files)} Python files")
    
    # Extract TODOs
    print("\n[2] Extracting TODO/FIXME comments...")
    all_todos = []
    for file_path in python_files:
        todos = extract_todos(file_path)
        all_todos.extend(todos)
    
    print(f"✅ Found {len(all_todos)} TODO/FIXME items")
    
    # Classify by priority
    print("\n[3] Classifying by priority...")
    by_priority = {}
    for priority in Priority:
        count = len([t for t in all_todos if t.priority == priority])
        by_priority[priority] = count
        print(f"   {priority.value}: {count}개")
    
    # Classify by category
    print("\n[4] Classifying by category...")
    by_category = {}
    for todo in all_todos:
        if todo.category not in by_category:
            by_category[todo.category] = 0
        by_category[todo.category] += 1
    
    for category in sorted(by_category.keys()):
        print(f"   {category}: {by_category[category]}개")
    
    # Generate inventory
    print("\n[5] Generating inventory...")
    inventory_md = generate_inventory(all_todos)
    inventory_json = generate_json_inventory(all_todos)
    
    # Save files
    md_file = docs_dir / "todo_inventory.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(inventory_md)
    print(f"✅ Markdown inventory saved to: {md_file}")
    
    json_file = docs_dir / "todo_inventory.json"
    import json
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(inventory_json, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON inventory saved to: {json_file}")
    
    print("\n" + "=" * 80)
    print("✅ TODO collection complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Total TODOs: {len(all_todos)}")
    print(f"  - Critical: {by_priority.get(Priority.CRITICAL, 0)}")
    print(f"  - High: {by_priority.get(Priority.HIGH, 0)}")
    print(f"  - Medium: {by_priority.get(Priority.MEDIUM, 0)}")
    print(f"  - Low: {by_priority.get(Priority.LOW, 0)}")


if __name__ == "__main__":
    main()

