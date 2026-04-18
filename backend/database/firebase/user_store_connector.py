"""
Firestore-backed user store for JIT user creation, namespace pool management, and vector quota tracking.
"""

import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone

from google.cloud import firestore
from google.cloud.firestore_v1.transforms import Increment

from database.firebase.firebase_connector import FirebaseConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserStoreConnector(FirebaseConnector):
    """
    Firestore wrapper for user records.

    Creates user documents on first authentication (JIT provisioning).
    Assigns users to a shared pool of Pinecone namespaces (ns_00..ns_19).
    Tracks vector counts at both user and namespace level.
    """

    DEFAULT_COLLECTION = "users"
    DEFAULT_VECTOR_QUOTA = 2_000

    NAMESPACE_POOL_SIZE = 20
    MAX_VECTORS_PER_NAMESPACE = 100_000
    MAX_USERS_PER_NAMESPACE = 10
    NAMESPACES_COLLECTION = "namespaces"

    @classmethod
    def _is_valid_pool_namespace(cls, namespace: str) -> bool:
        """
        Return True if namespace is a pool namespace id in the form ns_00..ns_{NAMESPACE_POOL_SIZE-1}.

        We intentionally require exactly two digits to match _namespace_id() formatting.
        """
        if not isinstance(namespace, str):
            return False
        if not namespace.startswith("ns_"):
            return False
        suffix = namespace[3:]
        if not suffix.isdigit():
            return False
        idx = int(suffix)
        return 0 <= idx < cls.NAMESPACE_POOL_SIZE

    def __init__(self, firestore_client, collection: str = DEFAULT_COLLECTION):
        super().__init__(firestore_client)
        self.collection = collection
        self._namespace_docs_initialized = False
        logger.info(f"Initialized UserStoreConnector with collection: {collection}")

    # ── Namespace pool ──────────────────────────────────────────────

    def _namespace_id(self, index: int) -> str:
        return f"ns_{index:02d}"

    def _ensure_namespace_docs(self) -> None:
        """Lazily create the 20 namespace docs if they don't exist yet."""
        if self._namespace_docs_initialized:
            return
        ns_col = self.db.collection(self.NAMESPACES_COLLECTION)
        for i in range(self.NAMESPACE_POOL_SIZE):
            ns_id = self._namespace_id(i)
            doc = ns_col.document(ns_id).get()
            if not doc.exists:
                ns_col.document(ns_id).set(
                    {
                        "namespace_id": ns_id,
                        "vector_count": 0,
                        "user_count": 0,
                    }
                )
        self._namespace_docs_initialized = True

    def _assign_namespace(self) -> str:
        """
        Pick the namespace with the most remaining vector capacity.

        Uses a Firestore transaction so the read + user_count increment
        is atomic — concurrent signups cannot push a namespace past its caps.

        Returns the namespace_id string (e.g. "ns_03").
        Raises RuntimeError if every namespace is full.
        """
        self._ensure_namespace_docs()
        ns_col = self.db.collection(self.NAMESPACES_COLLECTION)

        @firestore.transactional
        def _pick_and_claim(transaction):
            best_ns = None
            best_remaining = -1

            for i in range(self.NAMESPACE_POOL_SIZE):
                ns_id = self._namespace_id(i)
                doc = ns_col.document(ns_id).get(transaction=transaction)
                data = (
                    doc.to_dict()
                    if doc.exists
                    else {"vector_count": 0, "user_count": 0}
                )

                if data.get("user_count", 0) >= self.MAX_USERS_PER_NAMESPACE:
                    continue

                remaining = self.MAX_VECTORS_PER_NAMESPACE - data.get("vector_count", 0)
                if remaining <= 0:
                    continue

                if remaining > best_remaining:
                    best_remaining = remaining
                    best_ns = ns_id

            if best_ns is None:
                raise RuntimeError(
                    "All namespaces are full — no capacity for new users"
                )

            transaction.update(ns_col.document(best_ns), {"user_count": Increment(1)})
            return best_ns, best_remaining

        transaction = self.db.transaction()
        best_ns, best_remaining = _pick_and_claim(transaction)
        logger.info(
            f"Assigned namespace {best_ns} (remaining capacity ~{best_remaining} vectors)"
        )
        return best_ns

    # ── User CRUD ───────────────────────────────────────────────────

    def get_or_create_user(self, user_id: str) -> Dict[str, Any]:
        """
        Get existing user or create a new one with default fields.

        Always guarantees namespace, vector_count, and vector_quota exist.
        """
        doc_ref = self.db.collection(self.collection).document(user_id)
        doc = doc_ref.get()

        if doc.exists:
            user_data = doc.to_dict()
            updates: Dict[str, Any] = {}
            if not self._is_valid_pool_namespace(user_data.get("namespace", "")):
                updates["namespace"] = self._assign_namespace()
            if "vector_count" not in user_data:
                updates["vector_count"] = 0
            if "vector_quota" not in user_data:
                updates["vector_quota"] = self.DEFAULT_VECTOR_QUOTA
            if updates:
                doc_ref.update(updates)
                user_data.update(updates)
                logger.info(
                    f"Backfilled fields {list(updates.keys())} for user {user_id}"
                )
            return user_data

        namespace = self._assign_namespace()
        user_data = {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "namespace": namespace,
            "vector_count": 0,
            "vector_quota": self.DEFAULT_VECTOR_QUOTA,
        }
        doc_ref.set(user_data)
        logger.info(f"Created new user: {user_id} -> {namespace}")
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

    # ── Quota ───────────────────────────────────────────────────────

    def check_quota(self, user_id: str) -> Tuple[bool, int, int]:
        """
        Check if a user is under their vector quota.

        Returns:
            (is_under_quota, current_count, quota)
        """
        user_data = self.get_or_create_user(user_id)
        current_count = user_data.get("vector_count", 0)
        quota = user_data.get("vector_quota", self.DEFAULT_VECTOR_QUOTA)
        return (current_count < quota, current_count, quota)

    def reserve_quota(
        self, user_id: str, count: int, namespace: str = ""
    ) -> Tuple[bool, int, int]:
        """
        Atomically check quota and reserve capacity inside a Firestore transaction.

        Args:
            user_id: User ID to reserve quota for
            count: Number of vectors to reserve
            namespace: Optional namespace for namespace-level tracking

        Returns:
            Tuple[bool, int, int]: (reserved success, current count before, quota limit)
        """
        if count <= 0:
            user_data = self.get_or_create_user(user_id)
            current = user_data.get("vector_count", 0)
            quota = user_data.get("vector_quota", self.DEFAULT_VECTOR_QUOTA)
            return (True, current, quota)

        if namespace and not self._is_valid_pool_namespace(namespace):
            raise ValueError(
                f"Invalid namespace '{namespace}'. Expected a pool ID in the form "
                f"ns_00..ns_{self.NAMESPACE_POOL_SIZE - 1:02d}."
            )

        user_ref = self.db.collection(self.collection).document(user_id)

        @firestore.transactional
        def _reserve(transaction):
            snapshot = user_ref.get(transaction=transaction)
            # Invariant (intended flow): the caller has already ensured the user exists
            # via get_or_create_user(), which creates the user doc and assigns a valid
            # pool namespace (ns_00..ns_19) and initializes namespace docs.
            #
            # Firestore transaction.update() fails if the document doesn't exist, so
            # we fail fast with a clearer error to avoid silent misuse in other call sites.
            if not snapshot.exists:
                raise RuntimeError(
                    f"reserve_quota requires an existing user doc for '{user_id}'. "
                    "Call get_or_create_user(user_id) before reserve_quota()."
                )

            data = snapshot.to_dict() if snapshot.exists else {}
            current = data.get("vector_count", 0)
            quota = data.get("vector_quota", self.DEFAULT_VECTOR_QUOTA)

            # If we're tracking namespace usage, read it before any writes.
            # Firestore does not allow reads after writes in the same transaction.
            ns_ref = None
            if namespace:
                ns_ref = self.db.collection(self.NAMESPACES_COLLECTION).document(
                    namespace
                )
                # Same invariant as above: namespace docs are expected to exist already.
                ns_snapshot = ns_ref.get(transaction=transaction)
                if not ns_snapshot.exists:
                    raise RuntimeError(
                        f"reserve_quota expected namespace doc '{namespace}' to exist. "
                        "Ensure get_or_create_user()/_ensure_namespace_docs() has run and "
                        "that the namespace is a valid pool id (ns_00..ns_19)."
                    )

            if current + count > quota:
                return (False, current, quota)

            transaction.update(user_ref, {"vector_count": Increment(count)})
            if ns_ref is not None:
                transaction.update(ns_ref, {"vector_count": Increment(count)})
            return (True, current, quota)

        transaction = self.db.transaction()
        return _reserve(transaction)

    def increment_vector_count(
        self, user_id: str, count: int, namespace: str = ""
    ) -> None:
        """
        Atomically increment vector count at both user and namespace level.

        Args:
            user_id: User ID to increment count for
            count: Number of vectors to add
            namespace: Optional namespace for namespace-level tracking

        Returns:
            None
        """
        if count <= 0:
            return
        self.db.collection(self.collection).document(user_id).update(
            {"vector_count": Increment(count)}
        )
        if namespace:
            self.db.collection(self.NAMESPACES_COLLECTION).document(namespace).update(
                {"vector_count": Increment(count)}
            )
        logger.info(
            f"Incremented vector count by {count} for user {user_id} (namespace={namespace})"
        )

    def decrement_vector_count(
        self, user_id: str, count: int, namespace: str = ""
    ) -> None:
        """
        Atomically decrement vector count at both user and namespace level, flooring at 0.

        Args:
            user_id: User ID to decrement count for
            count: Number of vectors to remove
            namespace: Optional namespace for namespace-level tracking

        Returns:
            None
        """
        if count <= 0:
            return

        # User-level decrement
        user_ref = self.db.collection(self.collection).document(user_id)
        user_ref.update({"vector_count": Increment(-count)})
        doc = user_ref.get()
        if doc.exists and doc.to_dict().get("vector_count", 0) < 0:
            user_ref.update({"vector_count": 0})
            logger.warning(f"Floored negative vector count to 0 for user {user_id}")

        # Namespace-level decrement
        if namespace:
            ns_ref = self.db.collection(self.NAMESPACES_COLLECTION).document(namespace)
            ns_ref.update({"vector_count": Increment(-count)})
            ns_doc = ns_ref.get()
            if ns_doc.exists and ns_doc.to_dict().get("vector_count", 0) < 0:
                ns_ref.update({"vector_count": 0})
                logger.warning(
                    f"Floored negative vector count to 0 for namespace {namespace}"
                )

        logger.info(
            f"Decremented vector count by {count} for user {user_id} (namespace={namespace})"
        )

    # ── Video registration ──────────────────────────────────────────

    def register_video(
        self, user_id: str, hashed_identifier: str, chunk_count: int, filename: str
    ) -> None:
        """
        Register a processed video in the user's videos subcollection.

        Args:
            user_id: User ID who owns the video
            hashed_identifier: Client-generated hash of the video file
            chunk_count: Number of chunks created for this video
            filename: Original filename for tracking

        Returns:
            None
        """
        doc_ref = (
            self.db.collection(self.collection)
            .document(user_id)
            .collection("videos")
            .document(hashed_identifier)
        )
        doc_ref.set(
            {
                "hashed_identifier": hashed_identifier,
                "chunk_count": chunk_count,
                "filename": filename,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info(
            f"Registered video {hashed_identifier} ({chunk_count} chunks) for user {user_id}"
        )

    def get_video_chunk_count(self, user_id: str, hashed_identifier: str) -> int:
        """
        Get the chunk count for a registered video. Returns 0 if not found.

        Args:
            user_id: User ID who owns the video
            hashed_identifier: Client-generated hash of the video file

        Returns:
            int: Number of chunks, or 0 if video not found
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

        Args:
            user_id: User ID who owns the video
            hashed_identifier: Client-generated hash of the video file

        Returns:
            None
        """
        doc_ref = (
            self.db.collection(self.collection)
            .document(user_id)
            .collection("videos")
            .document(hashed_identifier)
        )
        doc_ref.delete()
        logger.info(f"Deregistered video {hashed_identifier} for user {user_id}")
