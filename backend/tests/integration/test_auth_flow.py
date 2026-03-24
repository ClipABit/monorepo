"""
Integration and E2E tests for the Auth0 authentication flow.

Uses real RSA keys and real JWT signing/verification to test the complete
auth pipeline — no mocking of jwt.decode or jwt.get_unverified_header.

Layers tested:
  - Integration: AuthConnector with real crypto, mocked JWKS HTTP endpoint
  - E2E: Full HTTP requests through FastAPI apps with real AuthConnector + real JWTs
"""

import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa as crypto_rsa
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from auth.auth_connector import AuthConnector
from api.server_fastapi_router import ServerFastAPIRouter
from api.search_fastapi_router import SearchFastAPIRouter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMAIN = "test-integration.auth0.com"
AUDIENCE = "https://api.integration-test.com"
ISSUER = f"https://{DOMAIN}/"
KID = "integration-test-kid-001"


# ---------------------------------------------------------------------------
# Module-scoped RSA key fixtures (generated once, shared across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rsa_private_key():
    """Generate a real 2048-bit RSA private key for signing JWTs."""
    return crypto_rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def rsa_public_key(rsa_private_key):
    """Derive the public key from the private key."""
    return rsa_private_key.public_key()


@pytest.fixture(scope="module")
def jwks_response(rsa_public_key):
    """Build a JWKS JSON response from the real public key."""
    jwk_dict = pyjwt.algorithms.RSAAlgorithm.to_jwk(rsa_public_key, as_dict=True)
    jwk_dict["kid"] = KID
    jwk_dict["use"] = "sig"
    jwk_dict["alg"] = "RS256"
    return {"keys": [jwk_dict]}


@pytest.fixture(scope="module")
def second_rsa_private_key():
    """A second RSA key to test token signed with wrong key."""
    return crypto_rsa.generate_private_key(public_exponent=65537, key_size=2048)


# ---------------------------------------------------------------------------
# Helper: sign a real JWT
# ---------------------------------------------------------------------------

def _sign_token(private_key, claims, kid=KID):
    """Sign a JWT with the given RSA private key and claims."""
    return pyjwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


# ---------------------------------------------------------------------------
# Token fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_claims():
    """Standard valid JWT claims."""
    return {
        "sub": "auth0|integration-user-42",
        "aud": AUDIENCE,
        "iss": ISSUER,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }


@pytest.fixture
def valid_token(rsa_private_key, valid_claims):
    """A real JWT with valid claims signed by the test RSA key."""
    return _sign_token(rsa_private_key, valid_claims)


@pytest.fixture
def expired_token(rsa_private_key):
    """A real JWT that has already expired."""
    claims = {
        "sub": "auth0|expired-user",
        "aud": AUDIENCE,
        "iss": ISSUER,
        "iat": int(time.time()) - 7200,
        "exp": int(time.time()) - 3600,
    }
    return _sign_token(rsa_private_key, claims)


@pytest.fixture
def wrong_audience_token(rsa_private_key):
    """A real JWT with an incorrect audience claim."""
    claims = {
        "sub": "auth0|wrong-aud-user",
        "aud": "https://wrong-api.example.com",
        "iss": ISSUER,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    return _sign_token(rsa_private_key, claims)


@pytest.fixture
def wrong_issuer_token(rsa_private_key):
    """A real JWT with an incorrect issuer claim."""
    claims = {
        "sub": "auth0|wrong-iss-user",
        "aud": AUDIENCE,
        "iss": "https://evil.auth0.com/",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    return _sign_token(rsa_private_key, claims)


@pytest.fixture
def no_sub_token(rsa_private_key):
    """A real JWT missing the sub claim."""
    claims = {
        "aud": AUDIENCE,
        "iss": ISSUER,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    return _sign_token(rsa_private_key, claims)


@pytest.fixture
def wrong_key_token(second_rsa_private_key):
    """A JWT signed with a different RSA key (not in the JWKS)."""
    claims = {
        "sub": "auth0|wrong-key-user",
        "aud": AUDIENCE,
        "iss": ISSUER,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    return _sign_token(second_rsa_private_key, claims)


# ---------------------------------------------------------------------------
# Connector fixture with mocked JWKS HTTP endpoint
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_jwks(mocker, jwks_response):
    """Patch requests.get to return the real JWKS without hitting the network."""
    mock_resp = mocker.MagicMock()
    mock_resp.json.return_value = jwks_response
    mock_resp.raise_for_status.return_value = None
    return mocker.patch("auth.auth_connector.requests.get", return_value=mock_resp)


@pytest.fixture
def connector(mock_jwks):
    """AuthConnector wired to the test domain/audience with mocked JWKS."""
    return AuthConnector(domain=DOMAIN, audience=AUDIENCE)


# ===========================================================================
# INTEGRATION TESTS: AuthConnector with real JWT crypto
# ===========================================================================

class TestRealJWTVerification:
    """Integration: Real RSA signing + real jwt.decode through AuthConnector."""

    def test_verifies_valid_jwt_and_returns_user_id(self, connector, valid_token):
        """A properly signed JWT with correct claims returns the user ID."""
        user_id = connector.verify_token(valid_token)
        assert user_id == "auth0|integration-user-42"

    def test_rejects_expired_jwt(self, connector, expired_token):
        """An expired JWT raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            connector.verify_token(expired_token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_rejects_wrong_audience(self, connector, wrong_audience_token):
        """A JWT with wrong audience raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            connector.verify_token(wrong_audience_token)
        assert exc_info.value.status_code == 401
        assert "audience" in exc_info.value.detail.lower()

    def test_rejects_wrong_issuer(self, connector, wrong_issuer_token):
        """A JWT with wrong issuer raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            connector.verify_token(wrong_issuer_token)
        assert exc_info.value.status_code == 401
        assert "issuer" in exc_info.value.detail.lower()

    def test_rejects_missing_sub_claim(self, connector, no_sub_token):
        """A JWT without a sub claim raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            connector.verify_token(no_sub_token)
        assert exc_info.value.status_code == 401
        assert "sub" in exc_info.value.detail.lower()

    def test_rejects_tampered_token(self, connector, valid_token):
        """A JWT with a modified payload (broken signature) raises 401."""
        # Split the token and corrupt the payload
        parts = valid_token.split(".")
        # Flip a character in the payload
        corrupted_payload = parts[1][:-1] + ("A" if parts[1][-1] != "A" else "B")
        tampered = f"{parts[0]}.{corrupted_payload}.{parts[2]}"

        with pytest.raises(HTTPException) as exc_info:
            connector.verify_token(tampered)
        assert exc_info.value.status_code == 401

    def test_rejects_token_signed_with_different_key(self, connector, wrong_key_token):
        """A JWT signed with a key not in the JWKS raises 401.

        The token header has the same kid, but the signature won't match
        the public key in the JWKS, so jwt.decode will fail.
        """
        with pytest.raises(HTTPException) as exc_info:
            connector.verify_token(wrong_key_token)
        assert exc_info.value.status_code == 401


class TestRealJWTDependencyInterface:
    """Integration: AuthConnector.__call__ with real JWTs."""

    @pytest.mark.asyncio
    async def test_extracts_and_verifies_real_bearer_token(self, connector, valid_token):
        """__call__ extracts Bearer token from request and verifies it."""
        request = MagicMock()
        request.headers.get.return_value = f"Bearer {valid_token}"

        user_id = await connector(request)
        assert user_id == "auth0|integration-user-42"

    @pytest.mark.asyncio
    async def test_rejects_real_expired_token_via_dependency(self, connector, expired_token):
        """__call__ rejects expired Bearer tokens."""
        request = MagicMock()
        request.headers.get.return_value = f"Bearer {expired_token}"

        with pytest.raises(HTTPException) as exc_info:
            await connector(request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_jit_user_creation_with_real_token(self, mock_jwks, valid_token):
        """__call__ triggers user store JIT creation with real user ID from JWT."""
        mock_user_store = MagicMock()
        auth = AuthConnector(domain=DOMAIN, audience=AUDIENCE, user_store=mock_user_store)

        request = MagicMock()
        request.headers.get.return_value = f"Bearer {valid_token}"

        user_id = await auth(request)

        assert user_id == "auth0|integration-user-42"
        mock_user_store.get_or_create_user.assert_called_once_with("auth0|integration-user-42")

    @pytest.mark.asyncio
    async def test_failed_auth_does_not_trigger_user_creation(self, mock_jwks, expired_token):
        """__call__ does NOT call user store when auth fails."""
        mock_user_store = MagicMock()
        auth = AuthConnector(domain=DOMAIN, audience=AUDIENCE, user_store=mock_user_store)

        request = MagicMock()
        request.headers.get.return_value = f"Bearer {expired_token}"

        with pytest.raises(HTTPException):
            await auth(request)

        mock_user_store.get_or_create_user.assert_not_called()


class TestJWKSCachingWithRealKeys:
    """Integration: JWKS caching behavior with real keys."""

    def test_caches_jwks_across_multiple_verifications(self, connector, mock_jwks, rsa_private_key):
        """Multiple verify_token calls only fetch JWKS once (cache hit)."""
        for i in range(5):
            token = _sign_token(rsa_private_key, {
                "sub": f"auth0|user-{i}",
                "aud": AUDIENCE,
                "iss": ISSUER,
                "iat": int(time.time()),
                "exp": int(time.time()) + 3600,
            })
            connector.verify_token(token)

        # JWKS should be fetched only once
        assert mock_jwks.call_count == 1

    def test_refreshes_jwks_after_ttl(self, connector, mock_jwks, valid_token):
        """JWKS is re-fetched after cache TTL expires."""
        connector.verify_token(valid_token)
        assert mock_jwks.call_count == 1

        # Expire the cache
        connector._jwks_cache_time = time.time() - AuthConnector.JWKS_CACHE_TTL - 1
        connector.verify_token(valid_token)
        assert mock_jwks.call_count == 2


# ===========================================================================
# E2E TESTS: Full HTTP requests through FastAPI with real AuthConnector
# ===========================================================================

def _make_server_app(auth_connector):
    """Create a full FastAPI app with ServerFastAPIRouter and real auth."""
    server = MagicMock()
    server.auth_connector = auth_connector
    server.job_store = MagicMock()
    server.job_store.get_job.return_value = None
    server.r2_connector = MagicMock()
    server.r2_connector.list_videos_page.return_value = (
        [{"file_name": "test.mp4", "presigned_url": "https://example.com/test.mp4", "hashed_identifier": "abc"}],
        None, 1, 1,
    )

    app = FastAPI()
    router = ServerFastAPIRouter(
        server_instance=server,
        is_file_change_enabled=True,
        environment="test",
    )
    app.include_router(router.router)
    return TestClient(app, raise_server_exceptions=False), server


def _make_search_app(auth_connector):
    """Create a full FastAPI app with SearchFastAPIRouter and real auth."""
    class FakeUserStore:
        def get_or_create_user(self, user_id):
            return {
                "user_id": user_id,
                "namespace": "user_test_ns",
                "vector_count": 0,
                "vector_quota": 10_000,
            }

    class FakeSearchService:
        def __init__(self):
            self.user_store = FakeUserStore()

        def _search_internal(self, query, namespace="", top_k=10, metadata_filter=None):
            return [{"id": "result-1", "score": 0.9, "metadata": {}}]

    app = FastAPI()
    router = SearchFastAPIRouter(
        search_service_instance=FakeSearchService(),
        auth_connector=auth_connector,
    )
    app.include_router(router.router)
    return TestClient(app, raise_server_exceptions=False)


class TestServerEndpointsE2E:
    """E2E: Full HTTP requests to server endpoints with real JWT auth."""

    def test_health_accessible_without_auth(self, connector):
        """GET /health works with no auth header."""
        client, _ = _make_server_app(connector)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_protected_endpoint_succeeds_with_valid_jwt(self, connector, valid_token):
        """GET /status succeeds with a real valid JWT."""
        client, _ = _make_server_app(connector)
        resp = client.get(
            "/status",
            params={"job_id": "test-job"},
            headers={"Authorization": f"Bearer {valid_token}"},
        )
        assert resp.status_code == 200

    def test_list_videos_succeeds_with_valid_jwt(self, connector, valid_token):
        """GET /videos returns data with a real valid JWT."""
        client, _ = _make_server_app(connector)
        resp = client.get(
            "/videos",
            headers={"Authorization": f"Bearer {valid_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_videos"] == 1

    def test_protected_endpoint_rejects_expired_jwt(self, connector, expired_token):
        """GET /status returns 401 with an expired JWT."""
        client, _ = _make_server_app(connector)
        resp = client.get(
            "/status",
            params={"job_id": "test-job"},
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert resp.status_code == 401

    def test_protected_endpoint_rejects_wrong_audience(self, connector, wrong_audience_token):
        """GET /status returns 401 with wrong audience JWT."""
        client, _ = _make_server_app(connector)
        resp = client.get(
            "/status",
            params={"job_id": "test-job"},
            headers={"Authorization": f"Bearer {wrong_audience_token}"},
        )
        assert resp.status_code == 401

    def test_protected_endpoint_rejects_missing_auth(self, connector):
        """GET /status returns 401 with no Authorization header."""
        client, _ = _make_server_app(connector)
        resp = client.get("/status", params={"job_id": "test-job"})
        assert resp.status_code == 401

    def test_protected_endpoint_rejects_basic_auth(self, connector):
        """GET /status returns 401 with Basic auth instead of Bearer."""
        client, _ = _make_server_app(connector)
        resp = client.get(
            "/status",
            params={"job_id": "test-job"},
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert resp.status_code == 401

    def test_protected_endpoint_rejects_tampered_jwt(self, connector, valid_token):
        """GET /status returns 401 with a tampered JWT."""
        parts = valid_token.split(".")
        corrupted = f"{parts[0]}.{parts[1]}x.{parts[2]}"
        client, _ = _make_server_app(connector)
        resp = client.get(
            "/status",
            params={"job_id": "test-job"},
            headers={"Authorization": f"Bearer {corrupted}"},
        )
        assert resp.status_code == 401


class TestSearchEndpointE2E:
    """E2E: Full HTTP requests to search endpoint with real JWT auth."""

    def test_health_accessible_without_auth(self, connector):
        """GET /health works with no auth header."""
        client = _make_search_app(connector)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_search_succeeds_with_valid_jwt(self, connector, valid_token):
        """GET /search returns results with a real valid JWT."""
        client = _make_search_app(connector)
        resp = client.get(
            "/search",
            params={"query": "test query"},
            headers={"Authorization": f"Bearer {valid_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1

    def test_search_rejects_expired_jwt(self, connector, expired_token):
        """GET /search returns 401 with expired JWT."""
        client = _make_search_app(connector)
        resp = client.get(
            "/search",
            params={"query": "test"},
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert resp.status_code == 401

    def test_search_rejects_missing_auth(self, connector):
        """GET /search returns 401 with no auth header."""
        client = _make_search_app(connector)
        resp = client.get("/search", params={"query": "test"})
        assert resp.status_code == 401


class TestJITUserProvisioningE2E:
    """E2E: Auth triggers user store JIT provisioning through full HTTP request."""

    def test_successful_request_triggers_user_creation(self, mock_jwks, valid_token):
        """A valid authenticated request triggers get_or_create_user."""
        mock_user_store = MagicMock()
        auth = AuthConnector(domain=DOMAIN, audience=AUDIENCE, user_store=mock_user_store)
        client, _ = _make_server_app(auth)

        resp = client.get(
            "/status",
            params={"job_id": "test"},
            headers={"Authorization": f"Bearer {valid_token}"},
        )
        assert resp.status_code == 200
        mock_user_store.get_or_create_user.assert_called_once_with("auth0|integration-user-42")

    def test_failed_auth_does_not_trigger_user_creation(self, mock_jwks, expired_token):
        """A failed authenticated request does NOT trigger get_or_create_user."""
        mock_user_store = MagicMock()
        auth = AuthConnector(domain=DOMAIN, audience=AUDIENCE, user_store=mock_user_store)
        client, _ = _make_server_app(auth)

        resp = client.get(
            "/status",
            params={"job_id": "test"},
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert resp.status_code == 401
        mock_user_store.get_or_create_user.assert_not_called()

    def test_unauthenticated_request_does_not_trigger_user_creation(self, mock_jwks):
        """A request with no auth header does NOT trigger get_or_create_user."""
        mock_user_store = MagicMock()
        auth = AuthConnector(domain=DOMAIN, audience=AUDIENCE, user_store=mock_user_store)
        client, _ = _make_server_app(auth)

        resp = client.get("/status", params={"job_id": "test"})
        assert resp.status_code == 401
        mock_user_store.get_or_create_user.assert_not_called()


class TestKeyRotationE2E:
    """E2E: JWKS key rotation — new key replaces old one."""

    def test_connector_uses_rotated_key(self, mocker, rsa_private_key):
        """After key rotation, tokens signed with the new key are accepted."""
        # Generate a new keypair to simulate rotation
        new_private_key = crypto_rsa.generate_private_key(public_exponent=65537, key_size=2048)
        new_public_key = new_private_key.public_key()
        new_kid = "rotated-kid-002"

        # Build JWKS with ONLY the new key (old key retired)
        new_jwk = pyjwt.algorithms.RSAAlgorithm.to_jwk(new_public_key, as_dict=True)
        new_jwk["kid"] = new_kid
        new_jwk["use"] = "sig"
        new_jwk["alg"] = "RS256"
        new_jwks = {"keys": [new_jwk]}

        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = new_jwks
        mock_resp.raise_for_status.return_value = None
        mocker.patch("auth.auth_connector.requests.get", return_value=mock_resp)

        auth = AuthConnector(domain=DOMAIN, audience=AUDIENCE)

        # Sign a token with the new key
        token = _sign_token(new_private_key, {
            "sub": "auth0|rotated-user",
            "aud": AUDIENCE,
            "iss": ISSUER,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }, kid=new_kid)

        client, _ = _make_server_app(auth)
        resp = client.get(
            "/status",
            params={"job_id": "test"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_old_key_rejected_after_rotation(self, mocker, rsa_private_key):
        """After key rotation, tokens signed with the OLD key are rejected."""
        # Generate new key
        new_private_key = crypto_rsa.generate_private_key(public_exponent=65537, key_size=2048)
        new_public_key = new_private_key.public_key()
        new_kid = "rotated-kid-003"

        # JWKS only has the NEW key
        new_jwk = pyjwt.algorithms.RSAAlgorithm.to_jwk(new_public_key, as_dict=True)
        new_jwk["kid"] = new_kid
        new_jwk["use"] = "sig"
        new_jwk["alg"] = "RS256"
        new_jwks = {"keys": [new_jwk]}

        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = new_jwks
        mock_resp.raise_for_status.return_value = None
        mocker.patch("auth.auth_connector.requests.get", return_value=mock_resp)

        auth = AuthConnector(domain=DOMAIN, audience=AUDIENCE)

        # Sign a token with the OLD key but old kid
        token = _sign_token(rsa_private_key, {
            "sub": "auth0|old-key-user",
            "aud": AUDIENCE,
            "iss": ISSUER,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }, kid=KID)  # old kid not in new JWKS

        client, _ = _make_server_app(auth)
        resp = client.get(
            "/status",
            params={"job_id": "test"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401
