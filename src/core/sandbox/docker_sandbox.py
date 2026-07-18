"""Docker 기반 코드 실행 샌드박스

안전한 코드 실행을 위한 Docker 컨테이너 기반 샌드박스 환경.
실제 SparkleForge 도구에서 사용되는 코드 실행을 담당.
"""

import logging
import os
from dataclasses import dataclass

try:
    import docker
    from docker.errors import APIError, ImageNotFound

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """샌드박스 설정"""

    image: str = "python:3.11-slim"
    node_image: str = "node:20-slim"
    bash_image: str = "debian:bookworm-slim"
    timeout: int = 30
    memory_limit: str = "512m"
    cpu_limit: float = 0.5
    network_disabled: bool = True
    read_only: bool = True
    tmpfs_size: str = "100m"
    runtime: str | None = "runsc"
    allow_default_runtime_fallback: bool = False
    pids_limit: int = 128
    dns_servers: tuple[str, ...] | None = None


@dataclass
class ExecutionResult:
    """실행 결과"""

    success: bool
    output: str
    error: str
    exit_code: int
    execution_time: float
    container_id: str | None = None


class DockerSandbox:
    """Docker 기반 코드 실행 샌드박스"""

    def __init__(self, config: SandboxConfig | None = None):
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker is not available. Please install docker package.")

        self.config = config or SandboxConfig()
        self.docker_client = docker.from_env()

    async def execute_code(
        self, code: str, language: str = "python", input_data: str | None = None
    ) -> ExecutionResult:
        """코드를 안전하게 실행

        Args:
            code: 실행할 코드
            language: 프로그래밍 언어 (python, javascript, bash 등)
            input_data: 표준 입력 데이터

        Returns:
            ExecutionResult: 실행 결과
        """
        import time

        start_time = time.time()
        container = None

        try:
            image, cmd = self._prepare_execution(code, language)
            kwargs = self._container_kwargs(image, cmd, input_data)

            try:
                container = self.docker_client.containers.create(**kwargs)
            except APIError as e:
                if not self._should_retry_without_runtime(e):
                    raise
                kwargs.pop("runtime", None)
                logger.warning(
                    "Docker runtime '%s' unavailable; retrying with Docker default runtime",
                    self.config.runtime,
                )
                container = self.docker_client.containers.create(**kwargs)

            container.start()
            wait_result = container.wait(timeout=self.config.timeout)
            exit_code = int(wait_result.get("StatusCode", -1))
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            execution_time = time.time() - start_time

            return ExecutionResult(
                success=exit_code == 0,
                output=stdout,
                error=stderr,
                exit_code=exit_code,
                execution_time=execution_time,
                container_id=container.id,
            )

        except ImageNotFound as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                output="",
                error=f"Docker image not found: {e}",
                exit_code=-1,
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Sandbox execution failed: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=execution_time,
            )
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    async def execute_file(
        self, file_path: str, language: str = "python", input_data: str | None = None
    ) -> ExecutionResult:
        """파일을 안전하게 실행

        Args:
            file_path: 실행할 파일 경로
            language: 프로그래밍 언어
            input_data: 표준 입력 데이터

        Returns:
            ExecutionResult: 실행 결과
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                code = f.read()
            return await self.execute_code(code, language, input_data)
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Failed to read file: {e}",
                exit_code=-1,
                execution_time=0.0,
            )

    def _prepare_execution(self, code: str, language: str) -> tuple[str, list[str]]:
        """언어별 이미지와 실행 명령 생성."""
        if language.lower() in ["python", "py"]:
            return self.config.image, ["python", "-c", code]

        elif language.lower() in ["javascript", "js", "node", "nodejs"]:
            return self.config.node_image, ["node", "-e", code]

        elif language.lower() in ["bash", "sh"]:
            return self.config.bash_image, ["bash", "-lc", code]

        else:
            raise ValueError(f"Unsupported language: {language}")

    def _container_kwargs(self, image: str, command: list[str], input_data: str | None) -> dict:
        kwargs = {
            "image": image,
            "command": command,
            "detach": True,
            "stdin_open": input_data is not None,
            "mem_limit": self.config.memory_limit,
            "cpu_quota": int(self.config.cpu_limit * 100000),
            "network_mode": "none" if self.config.network_disabled else "bridge",
            "read_only": self.config.read_only,
            "tmpfs": {"/tmp": f"rw,noexec,nosuid,size={self.config.tmpfs_size}"},
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "pids_limit": self.config.pids_limit,
        }
        if self.config.runtime:
            kwargs["runtime"] = self.config.runtime
        if self.config.dns_servers:
            kwargs["dns"] = list(self.config.dns_servers)
        return kwargs

    def _should_retry_without_runtime(self, error: Exception) -> bool:
        if not self.config.runtime or not self.config.allow_default_runtime_fallback:
            return False
        message = str(error).lower()
        return "unknown or invalid runtime" in message or self.config.runtime in message

    async def cleanup(self):
        """샌드박스 정리"""
        return None

    async def health_check(self) -> bool:
        """샌드박스 상태 확인"""
        try:
            # 간단한 코드 실행으로 테스트
            result = await self.execute_code("print('sandbox ready')", "python")
            return result.success
        except:
            return False


# 전역 샌드박스 인스턴스
_sandbox_instance = None


def get_sandbox() -> DockerSandbox:
    """전역 샌드박스 인스턴스 가져오기"""
    global _sandbox_instance
    if _sandbox_instance is None:
        runtime = os.getenv("SPARKLEFORGE_DOCKER_RUNTIME", "runsc").strip() or None
        allow_fallback = os.getenv(
            "SPARKLEFORGE_ALLOW_DOCKER_DEFAULT_RUNTIME_FALLBACK", "false"
        ).lower() in ("true", "1", "yes")
        dns_env = os.getenv("SPARKLEFORGE_SANDBOX_DNS_SERVERS", "").strip()
        dns_servers = tuple(host.strip() for host in dns_env.split(",") if host.strip()) or None
        config = SandboxConfig(
            image=os.getenv("SPARKLEFORGE_SANDBOX_PYTHON_IMAGE", "python:3.11-slim"),
            node_image=os.getenv("SPARKLEFORGE_SANDBOX_NODE_IMAGE", "node:20-slim"),
            bash_image=os.getenv("SPARKLEFORGE_SANDBOX_BASH_IMAGE", "debian:bookworm-slim"),
            runtime=runtime,
            allow_default_runtime_fallback=allow_fallback,
            memory_limit=os.getenv("SPARKLEFORGE_SANDBOX_MEMORY_LIMIT", SandboxConfig.memory_limit),
            cpu_limit=float(
                os.getenv("SPARKLEFORGE_SANDBOX_CPU_LIMIT", str(SandboxConfig.cpu_limit))
            ),
            tmpfs_size=os.getenv("SPARKLEFORGE_SANDBOX_TMPFS_SIZE", SandboxConfig.tmpfs_size),
            pids_limit=int(
                os.getenv("SPARKLEFORGE_SANDBOX_PIDS_LIMIT", str(SandboxConfig.pids_limit))
            ),
            dns_servers=dns_servers,
        )
        _sandbox_instance = DockerSandbox(config)
    return _sandbox_instance
