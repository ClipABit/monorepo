"""
Unit tests for auth endpoints.

Tests the device flow OAuth endpoints:
- POST /auth/device/code - Generate device codes
- POST /auth/device/poll - Poll authorization status
- POST /auth/device/authorize - Authorize device with Firebase token
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typing import Dict, Any, Optional

from api.server_fastapi_router import ServerFastAPIRouter


class FakeAuthConnector:
    """Fake auth connector for testing device flow."""

    def __init__(self):
        self.device_codes: Dict[str, Dict[str, Any]] = {}
        self.user_codes: Dict[str, str] = {}  # user_code -> device_code
        self._device_code_counter = 0
        self._user_code_counter = 0

    def generate_device_code(self) -> str:
        """Generate fake device code."""
        self._device_code_counter += 1
        return f"device_code_{self._device_code_counter}"

    def generate_user_code(self) -> str:
        """Generate fake user code."""
        self._user_code_counter += 1
        return f"ABC-{self._user_code_counter:03d}"

    def create_device_code_entry(
        self, device_code: str, user_code: str, expires_in: int = 600
    ) -> bool:
        """Create device code entry."""
        self.device_codes[device_code] = {
            "user_code": user_code,
            "status": "pending",
            "expires_in": expires_in,
        }
        self.user_codes[user_code] = device_code
        return True

    def get_device_code_poll_status(self, device_code: str) -> Optional[Dict[str, Any]]:
        """Get poll status for device code."""
        if device_code not in self.device_codes:
            return {"status": "expired", "error": "device_code_expired"}

        entry = self.device_codes[device_code]
        status = entry["status"]

        if status == "pending":
            return {"status": "pending"}
        elif status == "authorized":
            return {
                "status": "authorized",
                "user_id": entry.get("user_id"),
                "id_token": entry.get("id_token"),
                "refresh_token": entry.get("refresh_token", ""),
            }
        elif status == "denied":
            return {"status": "denied", "error": "user_denied_authorization"}
        else:
            return {"status": "expired", "error": "device_code_expired"}

    def get_device_code_by_user_code(self, user_code: str) -> Optional[str]:
        """Lookup device_code by user_code."""
        return self.user_codes.get(user_code)

    def set_device_code_authorized(
        self, device_code: str, user_id: str, id_token: str, refresh_token: str
    ) -> bool:
        """Mark device as authorized."""
        if device_code not in self.device_codes:
            return False

        self.device_codes[device_code].update(
            {
                "status": "authorized",
                "user_id": user_id,
                "id_token": id_token,
                "refresh_token": refresh_token,
            }
        )
        return True

    def verify_firebase_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        """Verify Firebase token (fake)."""
        # Fake verification - accept any token that starts with "valid_"
        if id_token.startswith("valid_"):
            user_id = id_token.replace("valid_", "user_")
            return {
                "user_id": user_id,
                "email": f"{user_id}@example.com",
                "email_verified": True,
            }
        return None


class FakeJobStore:
    """Minimal job store for router initialization."""

    def __init__(self):
        self._jobs = {}

    def create_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self._jobs[job_id] = data

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)


class ServerStub:
    """Minimal server stub with auth connector."""

    def __init__(self):
        self.auth_connector = FakeAuthConnector()
        self.job_store = FakeJobStore()


@pytest.fixture
def test_client() -> TestClient:
    """Create test client with auth endpoints."""
    server = ServerStub()
    app = FastAPI()
    router = ServerFastAPIRouter(
        server, is_file_change_enabled=False, environment="test"
    )
    app.include_router(router.router)
    return TestClient(app), server


def test_device_code_generation(test_client):
    """Test POST /auth/device/code generates codes correctly."""
    client, server = test_client

    response = client.post("/auth/device/code")

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "device_code" in data
    assert "user_code" in data
    assert "verification_url" in data
    assert "expires_in" in data
    assert "interval" in data

    # Check values
    assert data["device_code"] == "device_code_1"
    assert data["user_code"] == "ABC-001"
    assert data["verification_url"] == "clipabit.web.app/auth/device"
    assert data["expires_in"] == 600
    assert data["interval"] == 3

    # Verify stored in auth connector
    assert "device_code_1" in server.auth_connector.device_codes
    assert server.auth_connector.user_codes["ABC-001"] == "device_code_1"


def test_device_code_multiple_requests_generate_unique_codes(test_client):
    """Test multiple device code requests generate unique codes."""
    client, _ = test_client

    response1 = client.post("/auth/device/code")
    response2 = client.post("/auth/device/code")

    data1 = response1.json()
    data2 = response2.json()

    # Codes should be unique
    assert data1["device_code"] != data2["device_code"]
    assert data1["user_code"] != data2["user_code"]
    assert data1["user_code"] == "ABC-001"
    assert data2["user_code"] == "ABC-002"


def test_poll_device_code_pending(test_client):
    """Test polling returns pending when not yet authorized."""
    client, server = test_client

    # Create device code
    device_code = "device_code_1"
    user_code = "ABC-001"
    server.auth_connector.create_device_code_entry(device_code, user_code)

    # Poll
    response = client.post(
        "/auth/device/poll", json={"device_code": device_code}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"


def test_poll_device_code_not_found(test_client):
    """Test polling returns expired for non-existent code."""
    client, _ = test_client

    response = client.post(
        "/auth/device/poll", json={"device_code": "nonexistent"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "expired"
    assert data["error"] == "device_code_expired"


def test_poll_device_code_authorized(test_client):
    """Test polling returns tokens when authorized."""
    client, server = test_client

    # Create and authorize device code
    device_code = "device_code_1"
    user_code = "ABC-001"
    server.auth_connector.create_device_code_entry(device_code, user_code)
    server.auth_connector.set_device_code_authorized(
        device_code,
        user_id="user_123",
        id_token="token_abc",
        refresh_token="refresh_xyz",
    )

    # Poll
    response = client.post(
        "/auth/device/poll", json={"device_code": device_code}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "authorized"
    assert data["user_id"] == "user_123"
    assert data["id_token"] == "token_abc"
    assert data["refresh_token"] == "refresh_xyz"


def test_poll_device_code_missing_device_code(test_client):
    """Test polling without device_code returns 422 (validation error)."""
    client, _ = test_client

    response = client.post("/auth/device/poll", json={})

    assert response.status_code == 422  # FastAPI validation error for missing required field


def test_authorize_device_success(test_client):
    """Test authorizing device with valid Firebase token."""
    client, server = test_client

    # Create device code
    device_code = "device_code_1"
    user_code = "ABC-001"
    server.auth_connector.create_device_code_entry(device_code, user_code)

    # Authorize
    response = client.post(
        "/auth/device/authorize",
        json={
            "user_code": user_code,
            "firebase_id_token": "valid_firebase_user_123",
            "firebase_refresh_token": "refresh_token_xyz",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    # Verify device is authorized
    entry = server.auth_connector.device_codes[device_code]
    assert entry["status"] == "authorized"
    assert entry["user_id"] == "user_firebase_user_123"
    assert entry["id_token"] == "valid_firebase_user_123"


def test_authorize_device_invalid_firebase_token(test_client):
    """Test authorizing with invalid Firebase token returns 401."""
    client, server = test_client

    # Create device code
    device_code = "device_code_1"
    user_code = "ABC-001"
    server.auth_connector.create_device_code_entry(device_code, user_code)

    # Try to authorize with invalid token
    response = client.post(
        "/auth/device/authorize",
        json={
            "user_code": user_code,
            "firebase_id_token": "invalid_token",
            "firebase_refresh_token": "",
        },
    )

    assert response.status_code == 401
    assert "Invalid Firebase token" in response.json()["detail"]

    # Verify device is still pending
    entry = server.auth_connector.device_codes[device_code]
    assert entry["status"] == "pending"


def test_authorize_device_user_code_not_found(test_client):
    """Test authorizing with non-existent user_code returns 404."""
    client, _ = test_client

    response = client.post(
        "/auth/device/authorize",
        json={
            "user_code": "NONEXISTENT",
            "firebase_id_token": "valid_token_123",
            "firebase_refresh_token": "",
        },
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_authorize_device_missing_user_code(test_client):
    """Test authorizing without user_code returns 422 (validation error)."""
    client, _ = test_client

    response = client.post(
        "/auth/device/authorize",
        json={
            "firebase_id_token": "valid_token_123",
            "firebase_refresh_token": "",
        },
    )

    assert response.status_code == 422  # FastAPI validation error for missing required field


def test_authorize_device_missing_firebase_token(test_client):
    """Test authorizing without Firebase token returns 422 (validation error)."""
    client, _ = test_client

    response = client.post(
        "/auth/device/authorize",
        json={
            "user_code": "ABC-001",
            "firebase_refresh_token": "",
        },
    )

    assert response.status_code == 422  # FastAPI validation error for missing required field


def test_full_device_flow_integration(test_client):
    """Test complete device flow from generation to authorization."""
    client, server = test_client

    # Step 1: Generate device code
    response = client.post("/auth/device/code")
    assert response.status_code == 200
    codes = response.json()
    device_code = codes["device_code"]
    user_code = codes["user_code"]

    # Step 2: Poll (should be pending)
    response = client.post("/auth/device/poll", json={"device_code": device_code})
    assert response.status_code == 200
    assert response.json()["status"] == "pending"

    # Step 3: Authorize device
    response = client.post(
        "/auth/device/authorize",
        json={
            "user_code": user_code,
            "firebase_id_token": "valid_user_alice",
            "firebase_refresh_token": "refresh_alice",
        },
    )
    assert response.status_code == 200
    assert response.json()["success"] is True

    # Step 4: Poll again (should be authorized with tokens)
    response = client.post("/auth/device/poll", json={"device_code": device_code})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "authorized"
    assert data["user_id"] == "user_user_alice"
    assert data["id_token"] == "valid_user_alice"
    assert data["refresh_token"] == "refresh_alice"
