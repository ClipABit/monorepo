"""
Unit tests for AuthConnector.

Tests JWT verification, JWKS caching, and FastAPI dependency interface.
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException

from auth.auth_connector import AuthConnector


# Fake RSA key data for JWKS responses
FAKE_KID = "test-key-id"
FAKE_JWKS = {
    "keys": [
        {"kid": FAKE_KID, "kty": "RSA", "n": "fake-n", "e": "fake-e"}
    ]
}


@pytest.fixture
def connector():
    """AuthConnector with test domain and audience."""
    return AuthConnector(domain="test.auth0.com", audience="https://api.test.com")


@pytest.fixture
def mock_requests(mocker):
    """Mock requests.get for JWKS fetching."""
    mock_resp = mocker.MagicMock()
    mock_resp.json.return_value = FAKE_JWKS
    mock_resp.raise_for_status.return_value = None
    mock_get = mocker.patch("auth.auth_connector.requests.get", return_value=mock_resp)
    return mock_get


@pytest.fixture
def mock_jwt(mocker):
    """Mock jwt.decode and jwt.get_unverified_header."""
    mock_header = mocker.patch("auth.auth_connector.jwt.get_unverified_header")
    mock_header.return_value = {"kid": FAKE_KID, "alg": "RS256"}

    mock_from_jwk = mocker.patch("auth.auth_connector.jwt.algorithms.RSAAlgorithm.from_jwk")
    mock_from_jwk.return_value = "fake-public-key"

    mock_decode = mocker.patch("auth.auth_connector.jwt.decode")
    mock_decode.return_value = {"sub": "auth0|user123", "aud": "https://api.test.com"}

    return mock_decode, mock_header, mock_from_jwk


class TestAuthConnectorInitialization:
    """Test connector initialization."""

    def test_initializes_with_domain_and_audience(self):
        """Verify connector stores domain and audience."""
        connector = AuthConnector(domain="example.auth0.com", audience="https://api.example.com")

        assert connector.domain == "example.auth0.com"
        assert connector.audience == "https://api.example.com"

    def test_initializes_with_empty_jwks_cache(self):
        """Verify JWKS cache starts empty."""
        connector = AuthConnector(domain="example.auth0.com", audience="https://api.example.com")

        assert connector._jwks_cache is None
        assert connector._jwks_cache_time == 0


class TestGetJWKS:
    """Test JWKS fetching and caching."""

    def test_fetches_jwks_from_auth0(self, connector, mock_requests):
        """Verify JWKS is fetched from the correct URL."""
        result = connector._get_jwks()

        assert result == FAKE_JWKS
        mock_requests.assert_called_once_with(
            "https://test.auth0.com/.well-known/jwks.json", timeout=10
        )

    def test_caches_jwks_after_fetch(self, connector, mock_requests):
        """Verify JWKS is cached after first fetch."""
        connector._get_jwks()
        connector._get_jwks()

        mock_requests.assert_called_once()

    def test_refreshes_jwks_after_ttl_expires(self, connector, mock_requests, mocker):
        """Verify JWKS cache is refreshed after TTL."""
        connector._get_jwks()
        assert mock_requests.call_count == 1

        # Simulate cache expiry
        connector._jwks_cache_time = time.time() - AuthConnector.JWKS_CACHE_TTL - 1
        connector._get_jwks()

        assert mock_requests.call_count == 2

    def test_raises_on_network_error(self, connector, mocker):
        """Verify network errors propagate."""
        import requests as req
        mocker.patch(
            "auth.auth_connector.requests.get",
            side_effect=req.ConnectionError("Network error"),
        )

        with pytest.raises(req.ConnectionError):
            connector._get_jwks()


class TestGetSigningKey:
    """Test signing key lookup from JWKS."""

    def test_finds_matching_key(self, connector, mock_requests, mock_jwt):
        """Verify correct key is returned for matching kid."""
        _, _, mock_from_jwk = mock_jwt

        key = connector._get_signing_key("fake-token")

        assert key == "fake-public-key"
        mock_from_jwk.assert_called_once_with(FAKE_JWKS["keys"][0])

    def test_raises_401_for_unknown_kid(self, connector, mock_requests, mocker):
        """Verify 401 when token kid doesn't match any JWKS key."""
        mocker.patch(
            "auth.auth_connector.jwt.get_unverified_header",
            return_value={"kid": "unknown-kid", "alg": "RS256"},
        )

        with pytest.raises(HTTPException) as exc_info:
            connector._get_signing_key("fake-token")

        assert exc_info.value.status_code == 401
        assert "signing key" in exc_info.value.detail.lower()


class TestVerifyToken:
    """Test token verification logic."""

    def test_returns_user_id_for_valid_token(self, connector, mock_requests, mock_jwt):
        """Verify valid token returns user ID."""
        user_id = connector.verify_token("valid-token")

        assert user_id == "auth0|user123"

    def test_passes_correct_decode_params(self, connector, mock_requests, mock_jwt):
        """Verify jwt.decode is called with correct parameters."""
        mock_decode, _, _ = mock_jwt

        connector.verify_token("valid-token")

        mock_decode.assert_called_once_with(
            "valid-token",
            "fake-public-key",
            algorithms=["RS256"],
            audience="https://api.test.com",
            issuer="https://test.auth0.com/",
        )

    def test_raises_401_for_expired_token(self, connector, mock_requests, mock_jwt, mocker):
        """Verify expired token raises 401."""
        import jwt as pyjwt
        mock_decode, _, _ = mock_jwt
        mock_decode.side_effect = pyjwt.ExpiredSignatureError()

        with pytest.raises(HTTPException) as exc_info:
            connector.verify_token("expired-token")

        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_raises_401_for_invalid_audience(self, connector, mock_requests, mock_jwt, mocker):
        """Verify wrong audience raises 401."""
        import jwt as pyjwt
        mock_decode, _, _ = mock_jwt
        mock_decode.side_effect = pyjwt.InvalidAudienceError()

        with pytest.raises(HTTPException) as exc_info:
            connector.verify_token("bad-audience-token")

        assert exc_info.value.status_code == 401
        assert "audience" in exc_info.value.detail.lower()

    def test_raises_401_for_invalid_issuer(self, connector, mock_requests, mock_jwt, mocker):
        """Verify wrong issuer raises 401."""
        import jwt as pyjwt
        mock_decode, _, _ = mock_jwt
        mock_decode.side_effect = pyjwt.InvalidIssuerError()

        with pytest.raises(HTTPException) as exc_info:
            connector.verify_token("bad-issuer-token")

        assert exc_info.value.status_code == 401
        assert "issuer" in exc_info.value.detail.lower()

    def test_raises_401_for_generic_jwt_error(self, connector, mock_requests, mock_jwt, mocker):
        """Verify generic JWT errors raise 401."""
        import jwt as pyjwt
        mock_decode, _, _ = mock_jwt
        mock_decode.side_effect = pyjwt.PyJWTError("something went wrong")

        with pytest.raises(HTTPException) as exc_info:
            connector.verify_token("bad-token")

        assert exc_info.value.status_code == 401

    def test_raises_401_for_missing_sub_claim(self, connector, mock_requests, mock_jwt):
        """Verify token without sub claim raises 401."""
        mock_decode, _, _ = mock_jwt
        mock_decode.return_value = {"aud": "https://api.test.com"}  # no "sub"

        with pytest.raises(HTTPException) as exc_info:
            connector.verify_token("no-sub-token")

        assert exc_info.value.status_code == 401
        assert "sub" in exc_info.value.detail.lower()


class TestCallDependency:
    """Test __call__ FastAPI dependency interface."""

    @pytest.mark.asyncio
    async def test_extracts_bearer_token_and_verifies(self, connector, mock_requests, mock_jwt):
        """Verify __call__ extracts Bearer token and returns user ID."""
        request = MagicMock()
        request.headers.get.return_value = "Bearer valid-token"

        user_id = await connector(request)

        assert user_id == "auth0|user123"

    @pytest.mark.asyncio
    async def test_calls_user_store_get_or_create(self, mock_requests, mock_jwt):
        """Verify __call__ triggers JIT user creation when user_store is set."""
        mock_user_store = MagicMock()
        connector = AuthConnector(
            domain="test.auth0.com",
            audience="https://api.test.com",
            user_store=mock_user_store,
        )

        request = MagicMock()
        request.headers.get.return_value = "Bearer valid-token"

        user_id = await connector(request)

        assert user_id == "auth0|user123"
        mock_user_store.get_or_create_user.assert_called_once_with("auth0|user123")

    @pytest.mark.asyncio
    async def test_skips_user_store_when_not_set(self, connector, mock_requests, mock_jwt):
        """Verify __call__ works without user_store (no JIT creation)."""
        assert connector.user_store is None

        request = MagicMock()
        request.headers.get.return_value = "Bearer valid-token"

        user_id = await connector(request)

        assert user_id == "auth0|user123"

    @pytest.mark.asyncio
    async def test_raises_401_for_missing_auth_header(self, connector):
        """Verify 401 when no Authorization header present."""
        request = MagicMock()
        request.headers.get.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await connector(request)

        assert exc_info.value.status_code == 401
        assert "authorization" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_raises_401_for_non_bearer_auth(self, connector):
        """Verify 401 when Authorization header is not Bearer type."""
        request = MagicMock()
        request.headers.get.return_value = "Basic dXNlcjpwYXNz"

        with pytest.raises(HTTPException) as exc_info:
            await connector(request)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_raises_401_for_empty_bearer_token(self, connector, mock_requests, mocker):
        """Verify 401 when Bearer token is empty/invalid."""
        import jwt as pyjwt
        mocker.patch(
            "auth.auth_connector.jwt.get_unverified_header",
            side_effect=pyjwt.DecodeError("Not enough segments"),
        )

        request = MagicMock()
        request.headers.get.return_value = "Bearer "

        with pytest.raises((HTTPException, pyjwt.DecodeError)):
            await connector(request)
