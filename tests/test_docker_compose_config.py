"""Issues #692/#694: docker-compose.yaml health gating and sandbox network isolation.

#692: `docker-compose up` previously returned success even when the backend
crash-looped, because dependents only waited for containers to *start*, not
to become healthy. #694: the sandbox service shared the default network with
backend/db/redis, so an internet-enabled sandbox container could also reach
internal services. These tests parse the compose file directly (no live
`docker compose up` required) to guard the structural fix.
"""

from pathlib import Path

import yaml

COMPOSE_PATH = Path(__file__).resolve().parent.parent / "docker-compose.yaml"


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def test_backend_waits_for_redis_and_db_to_be_healthy():
    compose = _load_compose()
    depends_on = compose["services"]["backend"]["depends_on"]

    assert depends_on["redis"]["condition"] == "service_healthy"
    assert depends_on["db"]["condition"] == "service_healthy"


def test_frontend_waits_for_backend_to_be_healthy_and_has_its_own_healthcheck():
    compose = _load_compose()
    frontend = compose["services"]["frontend"]

    assert frontend["depends_on"]["backend"]["condition"] == "service_healthy"
    assert "healthcheck" in frontend
    assert "8501" in frontend["healthcheck"]["test"][-1]


def test_sandbox_service_is_isolated_from_default_network():
    compose = _load_compose()
    sandbox = compose["services"]["sandbox"]

    assert sandbox["networks"] == ["sandbox_network"]
    for service_name in ("backend", "frontend", "redis", "db"):
        assert "networks" not in compose["services"][service_name]


def test_sandbox_network_is_distinct_from_default_network():
    compose = _load_compose()
    networks = compose["networks"]

    assert networks["default"]["name"] != networks["sandbox_network"]["name"]
