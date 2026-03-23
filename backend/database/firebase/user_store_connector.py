"""
Firestore-backed user store for JIT user creation, namespace management, and vector quota tracking.
"""

import hashlib
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone

from google.cloud.firestore_v1.transforms import Increment

from database.firebase.firebase_connector import FirebaseConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserStoreConnector(FirebaseConnector):
    """
    Firestore wrapper for user records.

    Creates user documents on first authentication (JIT provisioning).
    Manages per-user Pinecone namespaces and vector quota tracking.
    """

    DEFAULT_COLLECTION = "users"
    DEFAULT_VECTOR_QUOTA = 10_000

    def __init__(self, firestore_client, collection: str = DEFAULT_COLLECTION):
        super().__init__(firestore_client)
        self.collection = collection
        logger.info(f"Initialized UserStoreConnector with collection: {collection}")

    @staticmethod
    def resolve_namespace(user_id: str) -> str:
        """
        Generate a deterministic, URL-safe Pinecone namespace from a user ID.

        Uses SHA-256 hash prefix to avoid special characters in Auth0 IDs (|, @, etc.).
        """
        return "user_" + hashlib.sha256(user_id.encode()).hexdigest()[:16]

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
            "namespace": self.resolve_namespace(user_id),
            "vector_count": 0,
            "vector_quota": self.DEFAULT_VECTOR_QUOTA,
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

    def check_quota(self, user_id: str) -> Tuple[bool, int, int]:
        """
        Check if a user is under their vector quota.

        Handles pre-existing users missing quota fields by treating them as defaults.

        Returns:
            (is_under_quota, current_count, quota)
        """
        user_data = self.get_or_create_user(user_id)
        current_count = user_data.get("vector_count", 0)
        quota = user_data.get("vector_quota", self.DEFAULT_VECTOR_QUOTA)

        # Backfill missing fields for pre-existing users
        updates = {}
        if "vector_count" not in user_data:
            updates["vector_count"] = 0
        if "vector_quota" not in user_data:
            updates["vector_quota"] = self.DEFAULT_VECTOR_QUOTA
        if "namespace" not in user_data:
            updates["namespace"] = self.resolve_namespace(user_id)
        if updates:
            self.db.collection(self.collection).document(user_id).update(updates)

        return (current_count < quota, current_count, quota)

    def increment_vector_count(self, user_id: str, count: int) -> None:
        """
        Atomically increment a user's vector count.

        Uses Firestore's server-side Increment to avoid race conditions.
        """
        if count <= 0:
            return
        doc_ref = self.db.collection(self.collection).document(user_id)
        doc_ref.update({"vector_count": Increment(count)})
        logger.info(f"Incremented vector count by {count} for user {user_id}")

    def decrement_vector_count(self, user_id: str, count: int) -> None:
        """
        Atomically decrement a user's vector count, flooring at 0.

        Uses Firestore's server-side Increment with a negative value.
        Reads after update to floor at 0 if the result went negative.
        """
        if count <= 0:
            return
        doc_ref = self.db.collection(self.collection).document(user_id)
        doc_ref.update({"vector_count": Increment(-count)})

        # Floor at 0 to prevent negative counts
        doc = doc_ref.get()
        if doc.exists:
            current = doc.to_dict().get("vector_count", 0)
            if current < 0:
                doc_ref.update({"vector_count": 0})
                logger.warning(f"Floored negative vector count to 0 for user {user_id}")

        logger.info(f"Decremented vector count by {count} for user {user_id}")

    def register_video(self, user_id: str, hashed_identifier: str, chunk_count: int, filename: str) -> None:
        """
        Register a processed video in the user's videos subcollection.

        Stores chunk count so we know how many vectors to decrement on deletion.
        """
        doc_ref = (
            self.db.collection(self.collection)
            .document(user_id)
            .collection("videos")
            .document(hashed_identifier)
        )
        doc_ref.set({
            "hashed_identifier": hashed_identifier,
            "chunk_count": chunk_count,
            "filename": filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(f"Registered video {hashed_identifier} ({chunk_count} chunks) for user {user_id}")

    def get_video_chunk_count(self, user_id: str, hashed_identifier: str) -> int:
        """
        Get the chunk count for a registered video.

        Returns 0 if the video is not found.
        """
        doc = (
            self.db.collection(self.collection)
            .document(user_id)
            .collection("videos")
            .document(hashed_identifier)
            .get()
        )
        if doc.exists:
            return doc.to_dict().get("chunk_count", 0)
        return 0

    def deregister_video(self, user_id: str, hashed_identifier: str) -> None:
        """
        Remove a video from the user's videos subcollection.

        Safe to call even if the video doesn't exist.
        """
        doc_ref = (
            self.db.collection(self.collection)
            .document(user_id)
            .collection("videos")
            .document(hashed_identifier)
        )
        doc_ref.delete()
        logger.info(f"Deregistered video {hashed_identifier} for user {user_id}")
