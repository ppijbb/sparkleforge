"""Regression tests for the read-only status/report API (src/web/status_api.py)."""

import pytest
from starlette.testclient import TestClient

from src.web import status_api

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("STATUS_API_TOKEN", "test-token")


def test_job_status_401_without_token():
    client = TestClient(status_api.app)

    response = client.get("/jobs/abc/status")

    assert response.status_code == 401


def test_job_status_401_with_wrong_token():
    client = TestClient(status_api.app)

    response = client.get("/jobs/abc/status", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401


def test_job_status_503_when_token_not_configured(monkeypatch):
    monkeypatch.delenv("STATUS_API_TOKEN", raising=False)
    client = TestClient(status_api.app)

    response = client.get("/jobs/abc/status", headers=AUTH)

    assert response.status_code == 503


def test_job_status_503_when_supabase_unconfigured(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: None)
    client = TestClient(status_api.app)

    response = client.get("/jobs/abc/status", headers=AUTH)

    assert response.status_code == 503


def test_job_status_404_when_job_missing(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: object())

    async def fake_get_job_status(job_id):
        return None

    monkeypatch.setattr(status_api, "get_job_status", fake_get_job_status)
    client = TestClient(status_api.app)

    response = client.get("/jobs/missing-id/status", headers=AUTH)

    assert response.status_code == 404


def test_job_status_503_when_query_fails(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: object())

    async def fake_get_job_status(job_id):
        raise status_api.SupabaseQueryError("network down")

    monkeypatch.setattr(status_api, "get_job_status", fake_get_job_status)
    client = TestClient(status_api.app)

    response = client.get("/jobs/abc/status", headers=AUTH)

    assert response.status_code == 503


def test_job_status_200_returns_job_row(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: object())

    async def fake_get_job_status(job_id):
        return {"id": job_id, "status": "running", "topic": "quantum ML"}

    monkeypatch.setattr(status_api, "get_job_status", fake_get_job_status)
    client = TestClient(status_api.app)

    response = client.get("/jobs/abc/status", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"id": "abc", "status": "running", "topic": "quantum ML"}


def test_report_401_without_token():
    client = TestClient(status_api.app)

    response = client.get("/reports/xyz")

    assert response.status_code == 401


def test_report_404_when_missing(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: object())

    async def fake_get_report(report_id):
        return None

    monkeypatch.setattr(status_api, "get_report", fake_get_report)
    client = TestClient(status_api.app)

    response = client.get("/reports/missing-id", headers=AUTH)

    assert response.status_code == 404


def test_report_503_when_query_fails(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: object())

    async def fake_get_report(report_id):
        raise status_api.SupabaseQueryError("network down")

    monkeypatch.setattr(status_api, "get_report", fake_get_report)
    client = TestClient(status_api.app)

    response = client.get("/reports/xyz", headers=AUTH)

    assert response.status_code == 503


def test_report_200_returns_report_row(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: object())

    async def fake_get_report(report_id):
        return {"id": report_id, "topic": "quantum ML", "full_report": "..."}

    monkeypatch.setattr(status_api, "get_report", fake_get_report)
    client = TestClient(status_api.app)

    response = client.get("/reports/xyz", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["id"] == "xyz"
