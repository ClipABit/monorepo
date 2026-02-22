"""
Base Firebase connector for Firestore-backed storage.
"""

import logging
from google.cloud.firestore import Client as FirestoreClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FirebaseConnector:
    """
    Base class for Firestore-backed connectors.

    Holds a shared Firestore client reference. Firebase Admin SDK
    must be initialized before constructing instances.
    """

    def __init__(self, firestore_client: FirestoreClient):
        self.db = firestore_client
