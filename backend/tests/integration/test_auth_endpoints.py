"""
Integration tests for authentication endpoints.

Tests for:
- POST /auth/device/code - Request device code
- POST /auth/device/poll - Poll device code status
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from api.server_fastapi_router import ServerFastAPIRouter


class FakeAuthConnector:
    """Fake AuthConnector for testing."""
    
    def __init__(self):
        self._device_codes: Dict[str, Dict[str, Any]] = {}
        self._user_codes: Dict[str, str] = {}
    
    def generate_device_code(self) -> str:
        """Generate a fake device code."""
        return "test-device-code-12345"
    
    def generate_user_code(self) -> str:
        """Generate a fake user code."""
        return "ABC-123"
    
    def create_device_code_entry(
        self,
        device_code: str,
        user_code: str,
        expires_in: int = 600
    ) -> bool:
        """Create a device code entry."""
        from datetime import datetime, timezone, timedelta
        
        self._device_codes[device_code] = {
            "user_code": user_code,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
        }
        self._user_codes[user_code] = device_code
        return True
    
    def get_device_code_poll_status(self, device_code: str) -> Dict[str, Any] | None:
        """Get device code poll status."""
        entry = self._device_codes.get(device_code)
        if entry is None:
            return {
                "status": "expired",
                "error": "device_code_expired"
            }
        
        status = entry.get("status", "pending")
        
        if status == "authorized":
            return {
                "status": "authorized",
                "user_id": entry.get("user_id"),
                "id_token": entry.get("id_token"),
                "refresh_token": entry.get("refresh_token")
            }
        elif status == "denied":
            return {
                "status": "denied",
                "error": "user_denied_authorization"
            }
        elif status == "pending":
            return {
                "status": "pending"
            }
        else:
            return {
                "status": "expired",
                "error": "device_code_expired"
            }
    
    def set_device_code_authorized(
        self,
        device_code: str,
        user_id: str,
        id_token: str,
        refresh_token: str
    ) -> bool:
        """Mark device code as authorized."""
        if device_code not in self._device_codes:
            return False
        self._device_codes[device_code]["status"] = "authorized"
        self._device_codes[device_code]["user_id"] = user_id
        self._device_codes[device_code]["id_token"] = id_token
        self._device_codes[device_code]["refresh_token"] = refresh_token
        return True
    
    def set_device_code_denied(self, device_code: str) -> bool:
        """Mark device code as denied."""
        if device_code not in self._device_codes:
            return False
        self._device_codes[device_code]["status"] = "denied"
        return True


class ServerStub:
    """Minimal server stub for auth endpoint tests."""
    
    def __init__(self, auth_connector=None):
        self.auth_connector = auth_connector or FakeAuthConnector()


@pytest.fixture()
def test_client_auth() -> tuple[TestClient, ServerStub, FakeAuthConnector]:
    """
    FastAPI TestClient with auth connector for testing auth endpoints.
    """
    auth_connector = FakeAuthConnector()
    server = ServerStub(auth_connector=auth_connector)
    app = FastAPI()
    router = ServerFastAPIRouter(server, is_internal_env=True, environment="dev")
    app.include_router(router.router)
    return TestClient(app), server, auth_connector


# ==============================================================================
# /auth/device/code endpoint tests
# ==============================================================================

def test_request_device_code_success(test_client_auth):
    """Test successful device code request."""
    client, server, auth_connector = test_client_auth
    
    resp = client.post("/auth/device/code")
    
    assert resp.status_code == 200
    data = resp.json()
    
    # Check response structure
    assert "device_code" in data
    assert "user_code" in data
    assert "verification_url" in data
    assert "expires_in" in data
    assert "interval" in data
    
    # Check response values
    assert data["device_code"] == "test-device-code-12345"
    assert data["user_code"] == "ABC-123"
    assert data["verification_url"] == "clipabit.web.app/auth/device"
    assert data["expires_in"] == 600
    assert data["interval"] == 3
    
    # Verify device code entry was created
    assert data["device_code"] in auth_connector._device_codes
    entry = auth_connector._device_codes[data["device_code"]]
    assert entry["user_code"] == data["user_code"]
    assert entry["status"] == "pending"


def test_request_device_code_creates_entry(test_client_auth):
    """Test that device code request creates proper entry in store."""
    client, server, auth_connector = test_client_auth
    
    resp = client.post("/auth/device/code")
    assert resp.status_code == 200
    data = resp.json()
    
    device_code = data["device_code"]
    user_code = data["user_code"]
    
    # Check device code entry exists
    assert device_code in auth_connector._device_codes
    assert auth_connector._user_codes[user_code] == device_code
    
    # Check entry structure
    entry = auth_connector._device_codes[device_code]
    assert entry["user_code"] == user_code
    assert entry["status"] == "pending"
    assert "created_at" in entry
    assert "expires_at" in entry


def test_request_device_code_handles_creation_failure(test_client_auth):
    """Test device code request when entry creation fails."""
    client, server, auth_connector = test_client_auth
    
    # Mock create_device_code_entry to return False
    auth_connector.create_device_code_entry = MagicMock(return_value=False)
    
    resp = client.post("/auth/device/code")
    
    assert resp.status_code == 500
    data = resp.json()
    assert "Failed to create device code entry" in data["detail"]


def test_request_device_code_handles_exception(test_client_auth):
    """Test device code request when exception occurs."""
    client, server, auth_connector = test_client_auth
    
    # Mock generate_device_code to raise exception
    auth_connector.generate_device_code = MagicMock(side_effect=Exception("Database error"))
    
    resp = client.post("/auth/device/code")
    
    assert resp.status_code == 500
    data = resp.json()
    assert "detail" in data


# ==============================================================================
# /auth/device/poll endpoint tests
# ==============================================================================

def test_poll_device_code_pending(test_client_auth):
    """Test polling device code when status is pending."""
    client, server, auth_connector = test_client_auth
    
    # Create a device code entry
    device_code = "test-device-code-pending"
    user_code = "XYZ-789"
    auth_connector.create_device_code_entry(device_code, user_code)
    
    # Poll for status
    resp = client.post("/auth/device/poll", json={"device_code": device_code})
    
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"


def test_poll_device_code_authorized(test_client_auth):
    """Test polling device code when user has authorized."""
    client, server, auth_connector = test_client_auth
    
    # Create and authorize a device code
    device_code = "test-device-code-authorized"
    user_code = "DEF-456"
    auth_connector.create_device_code_entry(device_code, user_code)
    auth_connector.set_device_code_authorized(
        device_code=device_code,
        user_id="user-123",
        id_token="id-token-abc",
        refresh_token="refresh-token-xyz"
    )
    
    # Poll for status
    resp = client.post("/auth/device/poll", json={"device_code": device_code})
    
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "authorized"
    assert data["user_id"] == "user-123"
    assert data["id_token"] == "id-token-abc"
    assert data["refresh_token"] == "refresh-token-xyz"


def test_poll_device_code_denied(test_client_auth):
    """Test polling device code when user has denied."""
    client, server, auth_connector = test_client_auth
    
    # Create and deny a device code
    device_code = "test-device-code-denied"
    user_code = "GHI-789"
    auth_connector.create_device_code_entry(device_code, user_code)
    auth_connector.set_device_code_denied(device_code)
    
    # Poll for status
    resp = client.post("/auth/device/poll", json={"device_code": device_code})
    
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "denied"
    assert data["error"] == "user_denied_authorization"


def test_poll_device_code_expired(test_client_auth):
    """Test polling device code when code is expired or not found."""
    client, server, auth_connector = test_client_auth
    
    # Poll for non-existent device code
    resp = client.post("/auth/device/poll", json={"device_code": "non-existent-code"})
    
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "expired"
    assert data["error"] == "device_code_expired"


def test_poll_device_code_missing_field(test_client_auth):
    """Test polling device code when device_code is missing."""
    client, server, auth_connector = test_client_auth
    
    # Try to poll without device_code
    resp = client.post("/auth/device/poll", json={})
    
    # FastAPI will return 422 for validation error when required field is missing
    assert resp.status_code == 422


def test_poll_device_code_empty_string(test_client_auth):
    """Test polling device code when device_code is empty string."""
    client, server, auth_connector = test_client_auth
    
    # Poll with empty device_code
    resp = client.post("/auth/device/poll", json={"device_code": ""})
    
    # The endpoint checks for empty string and returns 400
    assert resp.status_code == 400
    data = resp.json()
    assert "Missing required field: 'device_code'" in data["detail"]


def test_poll_device_code_handles_exception(test_client_auth):
    """Test polling device code when exception occurs."""
    client, server, auth_connector = test_client_auth
    
    # Mock get_device_code_poll_status to raise exception
    auth_connector.get_device_code_poll_status = MagicMock(side_effect=Exception("Database error"))
    
    resp = client.post("/auth/device/poll", json={"device_code": "test-code"})
    
    assert resp.status_code == 500
    data = resp.json()
    assert "detail" in data


# ==============================================================================
# Integration flow tests
# ==============================================================================

def test_device_flow_integration(test_client_auth):
    """Test complete device flow: request code -> poll pending -> authorize -> poll authorized."""
    client, server, auth_connector = test_client_auth
    
    # Step 1: Request device code
    resp1 = client.post("/auth/device/code")
    assert resp1.status_code == 200
    code_data = resp1.json()
    device_code = code_data["device_code"]
    user_code = code_data["user_code"]
    
    # Step 2: Poll while pending
    resp2 = client.post("/auth/device/poll", json={"device_code": device_code})
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "pending"
    
    # Step 3: Simulate user authorization
    auth_connector.set_device_code_authorized(
        device_code=device_code,
        user_id="user-456",
        id_token="id-token-123",
        refresh_token="refresh-token-789"
    )
    
    # Step 4: Poll after authorization
    resp3 = client.post("/auth/device/poll", json={"device_code": device_code})
    assert resp3.status_code == 200
    auth_data = resp3.json()
    assert auth_data["status"] == "authorized"
    assert auth_data["user_id"] == "user-456"
    assert auth_data["id_token"] == "id-token-123"
    assert auth_data["refresh_token"] == "refresh-token-789"


def test_device_flow_denial(test_client_auth):
    """Test device flow when user denies authorization."""
    client, server, auth_connector = test_client_auth
    
    # Step 1: Request device code
    resp1 = client.post("/auth/device/code")
    assert resp1.status_code == 200
    code_data = resp1.json()
    device_code = code_data["device_code"]
    
    # Step 2: Poll while pending
    resp2 = client.post("/auth/device/poll", json={"device_code": device_code})
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "pending"
    
    # Step 3: Simulate user denial
    auth_connector.set_device_code_denied(device_code)
    
    # Step 4: Poll after denial
    resp3 = client.post("/auth/device/poll", json={"device_code": device_code})
    assert resp3.status_code == 200
    deny_data = resp3.json()
    assert deny_data["status"] == "denied"
    assert deny_data["error"] == "user_denied_authorization"
