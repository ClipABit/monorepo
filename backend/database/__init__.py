from .pinecone_connector import PineconeConnector
from .r2_connector import R2Connector
from .cache.job_store_connector import JobStoreConnector
from .cache.url_cache_connector import UrlCacheConnector


def __getattr__(name):
    if name == "FirebaseConnector":
        from .firebase.firebase_connector import FirebaseConnector
        return FirebaseConnector
    if name == "UserStoreConnector":
        from .firebase.user_store_connector import UserStoreConnector
        return UserStoreConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'PineconeConnector',
    'R2Connector',
    'JobStoreConnector',
    'UrlCacheConnector',
    'FirebaseConnector',
    'UserStoreConnector',
]
