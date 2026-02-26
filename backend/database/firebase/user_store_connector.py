"""
Firestore-backed user store for JIT user creation.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from database.firebase.firebase_connector import FirebaseConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserStoreConnector(FirebaseConnector):
    """
    Firestore wrapper for user records.

    Creates user documents on first authentication (JIT provisioning).
    """

    DEFAULT_COLLECTION = "users"

    def __init__(self, firestore_client, collection: str = DEFAULT_COLLECTION):
        super().__init__(firestore_client)
        self.collection = collection
        logger.info(f"Initialized UserStoreConnector with collection: {collection}")

    def get_or_create_user(self, user_id: str) -> Dict[str, Any]:
        """
        Get existing user or create a new one with default fields.

        Returns the user document data.
        """
        doc_ref = self.db.collection(self.collection).document(user_id)
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()

        user_data = {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        doc_ref.set(user_data)
        logger.info(f"Created new user: {user_id}")
        return user_data

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID, returns None if not found."""
        doc = self.db.collection(self.collection).document(user_id).get()
        if doc.exists:
            return doc.to_dict()
        return None

    def user_exists(self, user_id: str) -> bool:
        """Check if a user exists."""
        return self.db.collection(self.collection).document(user_id).get().exists
