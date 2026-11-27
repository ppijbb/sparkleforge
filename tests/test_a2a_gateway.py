"""Tests for the A2A protocol gateway and agent card endpoints."""

import os
import pytest
from starlette.testclient import TestClient

from src.core.a2a.gateway import app


@pytest.fixture
def test_client(monkeypatch):
    monkeypatch.setenv("STATUS_API_TOKEN", "test-token-123")
    return TestClient(app)


def test_agent_card_endpoint(test_client):
    response = test_client.get("/.well-known/agent.json")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "SparkleForge"
    assert data["authentication"]["scheme"] == "Bearer"
    assert data["streaming"] is True


def test_message_send_unauthorized(test_client):
    response = test_client.post("/message/send", json={"message": "hello"})
    assert response.status_code == 401


def test_message_
