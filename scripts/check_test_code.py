#!/usr/bin/env python3
"""테스트 코드 점검 스크립트

쓸데없는 테스트 코드가 남아있는지 확인합니다.
"""

import sys
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_debug_validator_usage() -> Dict:
    """debug_validator.py 사용 여부 확인."""
    debug_validator_file = project_root / "src/core/debug_validator.py"

    if not debug_validator_file.exists():
        return {"exists": False}

    # Check if it's imported anywhere except itself
    imports = []

    for py_file in project_root.rglob("*.py"):
        if py_file == debug_validator_file:
            continue
        if "__pycache__" in str(py_file):
            continue

        try:
            with open(py_file, encoding="utf-8") as f:
                content = f.read()
                if "debug_validator" in content or "DebugValidator" in content:
                    imports.append(str(py_file.relative_to(project_root)))
        except:
            pass

    return {"exists": True, "imported_by": imports, "is_used": len(imports) > 0}


def check_test_files_in_src() -> List[str]:
    """Src 디렉토리에 테스트 파일이 있는지 확인."""
    test_files = []

    for py_file in (project_root / "src").rglob("*.py"):
        name = py_file.name
        if name.startswith("test_") or name.endswith("_test.py"):
            test_files.append(str(py_file.relative_to(project_root)))

    return test_files


def check_async_tests_missing_marker() -> List[Dict]:
    """``async def test_`` 함수에 ``@pytest.mark.asyncio``가 없는지 확인.

    ``asyncio_mode = auto``가 설정되어 있지 않은 환경에서는 ``async def test_``
    함수가 데코레이터 없이 실행되면 코루틴 객체만 반환되고 본문이 실행되지
    않아 "통과"로 위장한다. 이 정적 검사로 해당 패턴을 사전에 탐지한다.
    """
    import ast

    issues = []

    for py_file in (project_root / "tests").rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            with open(py_file, encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content, filename=str(py_file))
        except Exception:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue

            if not _has_asyncio_marker(node):
                issues.append(
                    {
                        "file": str(py_file.relative_to(project_root)),
                        "function": node.name,
                        "line": node.lineno,
                        "type": "async_test_missing_asyncio_marker",
                    }
                )

    return issues


def _has_asyncio_marker(func_node) -> bool:
    """``func_node``에 ``@pytest.mark.asyncio`` 계열 데코레이터가 있는지 검사."""
    import ast

    for decorator in func_node.decorator_list:
        if isinstance(decorator, ast.Attribute):
            parts = []
            cur = decorator
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            joined = ".".join(reversed(parts))
            if joined.endswith("pytest.mark.asyncio") or joined.endswith("mark.asyncio") or joined == "asyncio":
                return True
        if isinstance(decorator, ast.Name) and decorator.id == "asyncio":
            return True
    return False


def check_test_code_in_production() -> List[Dict]:
    """프로덕션 코드에 테스트 코드가 섞여있는지 확인."""
    issues = []

    # Check for test functions in production code
    import ast

    for py_file in (project_root / "src").rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        if py_file.name.startswith("test_"):
            continue

        try:
            with open(py_file, encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content, filename=str(py_file))

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_") and node.name != "test_mcp_connection":
                        issues.append(
                            {
                                "file": str(py_file.relative_to(project_root)),
                                "function": node.name,
                                "line": node.lineno,
                                "type": "test_function_in_production",
                            }
                        )
        except:
            pass

    return issues


def main():
    """Main function."""
    print("=" * 80)
    print("테스트 코드 점검")
    print("=" * 80)

    # 1. debug_validator 사용 여부
    print("\n[1] debug_validator.py 사용 여부 확인...")
    debug_usage = check_debug_validator_usage()
    if debug_usage.get("exists"):
        if debug_usage.get("is_used"):
            print(f"✅ debug_validator.py는 {len(debug_usage['imported_by'])}개 파일에서 사용 중")
            for importer in debug_usage["imported_by"]:
                print(f"   - {importer}")
        else:
            print("⚠️ debug_validator.py는 사용되지 않음 (제거 후보)")
    else:
        print("✅ debug_validator.py가 없음")

    # 2. src 디렉토리의 테스트 파일
    print("\n[2] src 디렉토리의 테스트 파일 확인...")
    test_files = check_test_files_in_src()
    if test_files:
        print(f"⚠️ src 디렉토리에 {len(test_files)}개 테스트 파일 발견:")
        for test_file in test_files:
            print(f"   - {test_file}")
        print("   → tests/ 디렉토리로 이동 권장")
    else:
        print("✅ src 디렉토리에 테스트 파일 없음")

    # 3. 프로덕션 코드의 테스트 함수
    print("\n[3] 프로덕션 코드의 테스트 함수 확인...")
    test_in_prod = check_test_code_in_production()
    if test_in_prod:
        print(f"⚠️ {len(test_in_prod)}개 테스트 함수가 프로덕션 코드에 발견됨:")
        for issue in test_in_prod:
            print(f"   - {issue['file']}:{issue['line']} - {issue['function']}")
    else:
        print("✅ 프로덕션 코드에 테스트 함수 없음")

    # 4. async 테스트 함수의 @pytest.mark.asyncio 누락 확인
    print("\n[4] async 테스트 함수의 @pytest.mark.asyncio 누락 확인...")
    async_missing = check_async_tests_missing_marker()
    if async_missing:
        print(f"⚠️ {len(async_missing)}개 async 테스트 함수에 @pytest.mark.asyncio 누락:")
        for issue in async_missing:
            print(f"   - {issue['file']}:{issue['line']} - {issue['function']}")
    else:
        print("✅ async 테스트 함수에 @pytest.mark.asyncio 누락 없음")

    # Summary
    print("\n" + "=" * 80)
    print("점검 결과 요약")
    print("=" * 80)

    issues_count = 0
    if debug_usage.get("exists") and not debug_usage.get("is_used"):
        issues_count += 1
    if test_files:
        issues_count += len(test_files)
    if test_in_prod:
        issues_count += len(test_in_prod)
    if async_missing:
        issues_count += len(async_missing)

    if issues_count == 0:
        print("✅ 문제 없음 - 모든 테스트 코드가 적절히 분리되어 있음")
    else:
        print(f"⚠️ {issues_count}개 이슈 발견 - 위의 상세 내용 참조")

    print("=" * 80)


if __name__ == "__main__":
    main()
