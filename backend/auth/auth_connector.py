"""
Auth connector for JWT authentication via Auth0.
"""

import logging
from typing import Optional, Dict, Any
import time
import jwt
import requests
from fastapi import Request, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuthConnector:
    """
    Auth0 JWT verification connector.

    Fetches and caches JWKS from Auth0, verifies access tokens,
    and returns user IDs. Exposes a FastAPI dependency via verify_token().
    """

    JWKS_CACHE_TTL = 3600  # 1 hour

    def __init__(self, domain: str, audience: str, user_store=None):
        self.domain = domain
        self.audience = audience
        self.user_store = user_store
        self._jwks_cache: Optional[Dict[str, Any]] = None
        self._jwks_cache_time: float = 0

    def _get_jwks(self) -> Dict[str, Any]:
        """Fetch and cache Auth0 JWKS (JSON Web Key Set)."""
        now = time.time()
        if self._jwks_cache and (now - self._jwks_cache_time) < self.JWKS_CACHE_TTL:
            return self._jwks_cache
        url = f"https://{self.domain}/.well-known/jwks.json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        self._jwks_cache = resp.json()
        self._jwks_cache_time = now
        return self._jwks_cache

    def _get_signing_key(self, token: str):
        """Find the matching public key from JWKS for the given token."""
        jwks = self._get_jwks()
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        for key in jwks.get("keys", []):
            if key["kid"] == kid:
                return jwt.algorithms.RSAAlgorithm.from_jwk(key)
        raise HTTPException(status_code=401, detail="Unable to find signing key")

    def verify_token(self, token: str) -> str:
        """
        Verify an Auth0 JWT and return the user ID (sub claim).

        Raises HTTPException 401 on any failure.
        """
        try:
            signing_key = self._get_signing_key(token)
            payload = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=f"https://{self.domain}/",
            )
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Token missing sub claim")
            return user_id
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidAudienceError:
            raise HTTPException(status_code=401, detail="Invalid audience")
        except jwt.InvalidIssuerError:
            raise HTTPException(status_code=401, detail="Invalid issuer")
        except jwt.PyJWTError as e:
            logger.error(f"JWT verification failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid token")

    async def __call__(self, request: Request) -> str:
        """
        FastAPI dependency interface.

        Usage: Depends(auth_connector) where auth_connector is an AuthConnector instance.
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = auth_header.split(" ", 1)[1]
        user_id = self.verify_token(token)
        if self.user_store:
            self.user_store.get_or_create_user(user_id)
        return user_id
