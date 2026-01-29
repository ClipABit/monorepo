"""
Auth service for device flow authentication.
"""

import json
import os
import firebase_admin
from firebase_admin import credentials, auth
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import secrets
import string
import modal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuthConnector:
    """
    Modal Dict wrapper for device flow authentication.
    
    Stores device codes with expiration (10 minutes) for OAuth device flow.
    """

    DEFAULT_DEVICE_DICT = "clipabit-auth-device-codes"
    DEFAULT_USER_DICT = "clipabit-auth-user-codes"

    TOKEN_EXPIRY_TIME = 600

    def __init__(self, device_dict_name: str = DEFAULT_DEVICE_DICT, user_dict_name: str = DEFAULT_USER_DICT):
        self.device_dict_name = device_dict_name
        self.user_dict_name = user_dict_name

        self.device_store = modal.Dict.from_name(device_dict_name, create_if_missing=True)
        self.user_store = modal.Dict.from_name(user_dict_name, create_if_missing=True)
        logger.info(f"Initialized AuthConnector with device dict: {device_dict_name} and user dict: {user_dict_name}")

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if a device code entry is expired."""
        expires_at= entry.get("expires_at")
        if expires_at is None:
            return False
        expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
        return datetime.now(timezone.utc) > expires_at
    
    def _delete_session(self, device_code: str, entry: Optional[Dict[str, Any]]) -> None:
        """
        Delete both dicts safely if the entry is expired.
            device_code -> entry
            user_code -> device_code
        """
        if entry is None:
            entry = self.device_store.get(device_code)
        if entry:
            user_code = entry.get("user_code")
            if user_code:
                if user_code in self.user_store:
                    del self.user_store[user_code]
                    logger.info(f"Deleted expired user code entry for user_code: {user_code}")
        if device_code in self.device_store:
            del self.device_store[device_code]
            logger.info(f"Deleted expired device code entry for device_code: {device_code[:8]}...")

    def generate_device_code(self) -> str:
        """Generate a secure random device code."""
        return secrets.token_urlsafe(48)

    def generate_user_code(self) -> str:
        """Generate a user-friendly code in format ABC-420."""

        letters = ''.join(secrets.choice(string.ascii_uppercase) for _ in range(3))
        digits = ''.join(secrets.choice(string.digits) for _ in range(3))
        return f"{letters}-{digits}"

    def create_device_code_entry(
        self,
        device_code: str,
        user_code: str,
        expires_in: int = TOKEN_EXPIRY_TIME
    ) -> bool:
        """Create a new device code entry with expiration."""
        try:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            entry = {
                "user_code": user_code,
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": expires_at.isoformat()
            }
            self.device_store[device_code] = entry
            self.user_store[user_code] = device_code
            logger.info(f"Created device code entry for user_code: {user_code}")
            return True
        except Exception as e:
            logger.error(f"Error creating device code entry: {e}")
            return False

    def get_device_code_entry(self, device_code: str) -> Optional[Dict[str, Any]]:
        """Retrieve device code entry, returns None if not found or expired."""
        try:
            
            entry = self.device_store.get(device_code)
            if entry is None:
                return None
            if self._is_expired(entry):
                self._delete_session(device_code, entry)
                return None
            return entry
        except Exception as e:
            logger.error(f"Error retrieving device code entry: {e}")
            return None

    def get_device_code_by_user_code(self, user_code: str) -> Optional[str]:
        """
        Lookup device_code by user_code.
        
        Returns the device_code if found and not expired, None otherwise.
        """
        try:
            device_code = self.user_store.get(user_code)
            if not device_code:
                return None

            entry =  self.get_device_code_entry(device_code)
            if entry is None:
                if user_code in self.user_store:
                    del self.user_store[user_code]
                return None
            
            return device_code
        except Exception as e:
            logger.error(f"Error looking up device_code by user_code: {e}")
            return None

    def update_device_code_status(self, device_code: str, status: str) -> bool:
        """Update the status of a device code (e.g., 'pending' -> 'authorized')."""
        try:
            entry = self.get_device_code_entry(device_code)
            if entry is None:
                return False
            
            entry["status"] = status
            self.device_store[device_code] = entry
            logger.info(f"Updated device code {device_code[:8]}... status to: {status}")
            return True
        except Exception as e:
            logger.error(f"Error updating device code status: {e}")
            return False

    def set_device_code_authorized(
        self,
        device_code: str,
        user_id: str,
        id_token: str,
        refresh_token: str
    ) -> bool:
        """Mark device code as authorized and store user tokens."""
        try:
            entry = self.get_device_code_entry(device_code)
            if entry is None:
                return False
            
            entry["status"] = "authorized"
            entry["user_id"] = user_id
            entry["id_token"] = id_token
            entry["refresh_token"] = refresh_token
            entry["authorized_at"] = datetime.now(timezone.utc).isoformat()
            self.device_store[device_code] = entry
            logger.info(f"Device code {device_code[:8]}... authorized for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error setting device code as authorized: {e}")
            return False

    def set_device_code_denied(self, device_code: str) -> bool:
        """Mark device code as denied by user."""
        try:
            entry = self.get_device_code_entry(device_code)
            if entry is None:
                return False
            
            entry["status"] = "denied"
            entry["denied_at"] = datetime.now(timezone.utc).isoformat()
            self.device_store[device_code] = entry
            logger.info(f"Device code {device_code[:8]}... denied by user")
            return True
        except Exception as e:
            logger.error(f"Error setting device code as denied: {e}")
            return False

    def get_device_code_poll_status(self, device_code: str) -> Optional[Dict[str, Any]]:
        """
        Get device code status for polling endpoint.
        
        Returns status dict with appropriate fields based on state:
        - pending: {"status": "pending"}
        - authorized: {"status": "authorized", "user_id": ..., "id_token": ..., "refresh_token": ...}
        - expired: {"status": "expired", "error": "device_code_expired"}
        - denied: {"status": "denied", "error": "user_denied_authorization"}
        - not_found: None (treat as expired)
        """
        entry = self.get_device_code_entry(device_code)
        
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

    def delete_device_code(self, device_code: str) -> bool:
        """Remove device code entry."""
        try:
            entry = self.get_device_code_entry(device_code)
            if entry is None:
                return False
            self._delete_session(device_code, entry)
            logger.info(f"Deleted device code entry for device_code: {device_code[:8]}...")
            return True
        except Exception as e:
            logger.error(f"Error deleting device code: {e}")
            return False
        
    def verify_firebase_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        """Verify Firebase ID token from website/plugin.
        
        Note: Firebase Admin SDK is initialized at server startup in http_server.py.
        """
        try:
            decoded_token = auth.verify_id_token(id_token)
            return {
                "user_id": decoded_token['uid'],
                "email": decoded_token.get('email'),
                "email_verified": decoded_token.get('email_verified', False)
            }
        except auth.InvalidIdTokenError as e:
            logger.error(f"Invalid Firebase token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error verifying Firebase token: {e}")
            return None
    