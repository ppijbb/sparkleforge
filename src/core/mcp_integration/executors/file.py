"""Filesystem tool dispatch (ToolCategory.FILE): read/write/list/delete operations."""
import difflib
import itertools
import logging
import time
from typing import Any, Dict

from src.core.tools.registry import ToolResult

logger = logging.getLogger(__name__)

MIN_MATCH_LINES = 2


def _edit_mismatch_hint(content: str, old_string: str, context_lines: int = 3, max_chars: int = 800) -> str:
    """Best-effort hint for an edit_file 'old_string not found' failure.

    A bare "not found" error gives the model nothing to correct itself with,
    so it re-reads the whole file (an extra iteration, and one the momentum
    guard tends to flag as a stall -- #1338). Find the file's closest
    matching region to `old_string` -- typically the same code with
    different whitespace/indentation -- and return it with line numbers so
    the model can retry with the exact text in one shot.

    Matching is line-level (not character-level) to avoid false positives
    from common programming idioms (e.g. `}\n}\n}\n`) and to keep the
    SequenceMatcher complexity proportional to line counts, not file size.
    """
    content_lines = content.splitlines()
    old_lines = old_string.splitlines()
    # Compare stripped lines so a match isn't fragmented by leading/trailing
    # whitespace differences on an otherwise-identical line (e.g. a single
    # trailing space) -- exactly the kind of mismatch this hint exists to
    # surface. Line numbers/content displayed below still use the originals.
    matcher = difflib.SequenceMatcher(
        None,
        [line.strip() for line in content_lines],
        [line.strip() for line in old_lines],
        autojunk=False,
    )
    match = matcher.find_longest_match(0, len(content_lines), 0, len(old_lines))

    if match.size < MIN_MATCH_LINES:
        # No meaningful overlap at all -- showing the head of the file is
        # more useful than nothing.
        snippet = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(content_lines[: context_lines * 2 + 1]))
        return f"No similar text found in the file. Start of file instead:\n{snippet[:max_chars]}"

    start_line = match.a  # 0-indexed line where the match begins
    lo = max(0, start_line - context_lines)
    hi = min(len(content_lines), start_line + match.size + context_lines)
    numbered = "\n".join(f"{i + 1}: {content_lines[i]}" for i in range(lo, hi))
    return f"Closest match in file (lines {lo + 1}-{hi}):\n{numbered[:max_chars]}"


async def _execute_file_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """파일 작업 도구 실행."""
    start_time = time.time()

    try:
        from pathlib import Path

        # 안전성 검증: 작업 디렉토리 제한
        allowed_dirs = [
            Path.cwd(),  # 현재 작업 디렉토리
            Path("./outputs"),  # 출력 디렉토리
            Path("./workspace"),  # 워크스페이스
            Path("./temp"),  # 임시 디렉토리
        ]

        def _is_safe_path(file_path: str) -> bool:
            """경로 안전성 검증."""
            try:
                path = Path(file_path).resolve()
                # 상대 경로만 허용
                if path.is_absolute() and not any(
                    path.is_relative_to(allowed) for allowed in allowed_dirs
                ):
                    # 절대 경로인 경우 허용된 디렉토리 내에 있는지 확인
                    for allowed in allowed_dirs:
                        try:
                            path.relative_to(allowed.resolve())
                            return True
                        except ValueError:
                            continue
                    return False
                # 상대 경로는 허용
                return True
            except Exception:
                return False

        if tool_name == "filesystem":
            # 범용 filesystem 도구: operation/action 파라미터를 구체 도구로 매핑
            operation = str(parameters.get("operation") or parameters.get("action") or "").lower()
            op_map = {
                "create": "create_file",
                "read": "read_file",
                "write": "write_file",
                "edit": "edit_file",
                "list": "list_files",
                "delete": "delete_file",
            }
            mapped = op_map.get(operation)
            if not mapped:
                raise ValueError(f"Unknown filesystem operation: {operation or '(missing)'}")
            # read 대상이 디렉토리면 목록 조회로 처리
            target = parameters.get("path") or parameters.get("file_path") or ""
            if mapped == "read_file" and target:
                target_path = Path(target)
                if target_path.is_dir():
                    mapped = "list_files"
                elif (
                    not target_path.exists()
                    and target_path.suffix
                    and target_path.with_suffix("").is_dir()
                ):
                    # 모델이 디렉토리 이름에 확장자를 잘못 붙여 추측한 경우
                    # (예: 실제로는 디렉토리인 "wal/"을 "wal.rs" 파일로 착각) --
                    # 존재하지 않는 파일 에러 대신 실제 디렉토리 목록을 돌려준다.
                    target = str(target_path.with_suffix(""))
                    parameters = {**parameters, "path": target}
                    mapped = "list_files"
            if "file_path" not in parameters and "path" in parameters:
                parameters = {**parameters, "file_path": parameters["path"]}
            if mapped == "list_files" and "directory_path" not in parameters:
                parameters = {
                    **parameters,
                    "directory_path": parameters.get("path", parameters.get("file_path", ".")),
                }
            return await _execute_file_tool(mapped, parameters)

        if tool_name == "create_file":
            file_path = parameters.get("file_path", "")
            content = parameters.get("content", "")

            if not file_path:
                raise ValueError("file_path parameter is required")
            if not _is_safe_path(file_path):
                raise ValueError(f"Unsafe file path: {file_path}")

            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

            return ToolResult(
                success=True,
                data={"file_path": str(path), "size": len(content)},
                execution_time=time.time() - start_time,
                confidence=0.9,
            )

        elif tool_name == "read_file":
            file_path = parameters.get("file_path", "")

            if not file_path:
                raise ValueError("file_path parameter is required")
            if not _is_safe_path(file_path):
                raise ValueError(f"Unsafe file path: {file_path}")

            path = Path(file_path)
            if not path.exists():
                # 복구 유도: 경로가 디렉토리일 수 있으므로 명확한 안내 제공.
                # 모델이 다음 턴에서 list_files로 진짜 파일을 찾도록 유도한다.
                parent = path.parent
                if parent.exists() and parent.is_dir():
                    try:
                        siblings = list(
                            itertools.islice(
                                (p.name for p in parent.iterdir() if p.is_dir()), 10
                            )
                        )
                    except OSError:
                        siblings = []
                    hint = f" This path may be a directory; use list_files to inspect {parent} and read the actual files inside (subdirs: {', '.join(siblings)})."
                else:
                    hint = " Use list_files to inspect the parent directory and read the actual files inside."
                raise FileNotFoundError(f"File not found: {file_path}.{hint}")

            if path.is_dir():
                # 디렉토리를 파일로 읽으려 한 경우 자동으로 list_files로 복구한다.
                logger.info("read_file target is a directory; recovering via list_files: %s", file_path)
                return await _execute_file_tool(
                    "list_files",
                    {**parameters, "directory_path": file_path},
                )

            content = path.read_text(encoding="utf-8")

            return ToolResult(
                success=True,
                data={"file_path": str(path), "content": content, "size": len(content)},
                execution_time=time.time() - start_time,
                confidence=0.9,
            )

        elif tool_name == "write_file":
            file_path = parameters.get("file_path", "")
            content = parameters.get("content", "")

            if not file_path:
                raise ValueError("file_path parameter is required")
            if not _is_safe_path(file_path):
                raise ValueError(f"Unsafe file path: {file_path}")

            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

            return ToolResult(
                success=True,
                data={"file_path": str(path), "size": len(content)},
                execution_time=time.time() - start_time,
                confidence=0.9,
            )

        elif tool_name == "edit_file":
            file_path = parameters.get("file_path", "")
            old_string = parameters.get("old_string", "")
            new_string = parameters.get("new_string", "")

            if not file_path:
                raise ValueError("file_path parameter is required")
            if not _is_safe_path(file_path):
                raise ValueError(f"Unsafe file path: {file_path}")

            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            content = path.read_text(encoding="utf-8")
            if old_string not in content:
                hint = _edit_mismatch_hint(content, old_string)
                raise ValueError(f"Old string not found in file: {file_path}\n{hint}")

            new_content = content.replace(old_string, new_string)
            path.write_text(new_content, encoding="utf-8")

            return ToolResult(
                success=True,
                data={
                    "file_path": str(path),
                    "replacements": content.count(old_string),
                },
                execution_time=time.time() - start_time,
                confidence=0.9,
            )

        elif tool_name == "list_files":
            directory_path = parameters.get("directory_path", ".")
            recursive = parameters.get("recursive", False)

            if not _is_safe_path(directory_path):
                raise ValueError(f"Unsafe directory path: {directory_path}")

            path = Path(directory_path)
            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            if not path.is_dir():
                raise ValueError(f"Path is not a directory: {directory_path}")

            files = []
            if recursive:
                for item in path.rglob("*"):
                    files.append(
                        {
                            "name": item.name,
                            "path": str(item.relative_to(path)),
                            "is_file": item.is_file(),
                            "size": item.stat().st_size if item.is_file() else 0,
                        }
                    )
            else:
                for item in path.iterdir():
                    files.append(
                        {
                            "name": item.name,
                            "path": item.name,
                            "is_file": item.is_file(),
                            "size": item.stat().st_size if item.is_file() else 0,
                        }
                    )

            return ToolResult(
                success=True,
                data={"directory": str(path), "files": files, "count": len(files)},
                execution_time=time.time() - start_time,
                confidence=0.9,
            )

        elif tool_name == "delete_file":
            file_path = parameters.get("file_path", "")

            if not file_path:
                raise ValueError("file_path parameter is required")
            if not _is_safe_path(file_path):
                raise ValueError(f"Unsafe file path: {file_path}")

            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File or directory not found: {file_path}")

            if path.is_file():
                path.unlink()
            elif path.is_dir():
                import shutil

                shutil.rmtree(path)

            return ToolResult(
                success=True,
                data={"file_path": str(path), "deleted": True},
                execution_time=time.time() - start_time,
                confidence=0.9,
            )

        else:
            raise ValueError(f"Unknown file tool: {tool_name}")

    except Exception as e:
        logger.error(f"File tool execution failed: {tool_name} - {e}", exc_info=True)
        return ToolResult(
            success=False,
            data=None,
            error=f"File tool execution failed: {str(e)}",
            execution_time=time.time() - start_time,
            confidence=0.0,
        )
