import logging
import time
from typing import Optional, Dict, Any, List

import modal

logger = logging.getLogger(__name__)


# Cache entries must expire slightly before the R2 presigned URLs (1 hour) to avoid
# returning dead links; keep this under the URL TTL to prevent race conditions.
VIDEO_PAGE_TTL_SECONDS = 59 * 60  # 59 minutes


class UrlCacheConnector:
    """Modal Dict-backed cache for paginated video listings."""

    DEFAULT_DICT_NAME = "clipabit-url-cache"

    def __init__(self, environment: str, dict_name: str = DEFAULT_DICT_NAME):
        self.environment = environment
        self.dict_name = dict_name
        self.cache = modal.Dict.from_name(dict_name, create_if_missing=True)
        logger.info(
            "Initialized UrlCacheConnector with Dict '%s' for environment '%s'",
            dict_name,
            environment,
        )

    def _make_page_key(self, namespace: str, page_token: Optional[str], page_size: int) -> str:
        token = page_token or ""
        return f"{self.environment}:{namespace}:{page_size}:{token}"

    def _make_meta_key(self, namespace: str) -> str:
        return f"{self.environment}:{namespace}:meta"

    def _is_expired(self, payload: Dict[str, Any]) -> bool:
        cached_at = payload.get("cached_at") if isinstance(payload, dict) else None
        if cached_at is None:
            return True
        return (time.time() - cached_at) > VIDEO_PAGE_TTL_SECONDS

    def get_page(
        self,
        namespace: str,
        page_token: Optional[str],
        page_size: int,
    ) -> Optional[Dict[str, Any]]:
        key = self._make_page_key(namespace, page_token, page_size)
        try:
            if key in self.cache:
                value = self.cache[key]
                if self._is_expired(value):
                    try:
                        del self.cache[key]
                    except KeyError:
                        pass
                    logger.debug("UrlCacheConnector entry expired for key %s", key)
                    return None
                logger.debug("UrlCacheConnector hit for key %s", key)
                return value
        except Exception as exc:
            logger.error("UrlCacheConnector read failed for key %s: %s", key, exc)
        return None

    def set_page(
        self,
        namespace: str,
        page_token: Optional[str],
        page_size: int,
        videos: List[Dict[str, Any]],
        next_token: Optional[str],
    ) -> None:
        key = self._make_page_key(namespace, page_token, page_size)
        payload = {
            "videos": videos,
            "next_token": next_token,
            "cached_at": time.time(),
        }
        try:
            self.cache[key] = payload
        except Exception as exc:
            logger.error("UrlCacheConnector write failed for key %s: %s", key, exc)

    def get_namespace_metadata(self, namespace: str) -> Optional[Dict[str, Any]]:
        key = self._make_meta_key(namespace)
        try:
            if key in self.cache:
                value = self.cache[key]
                if self._is_expired(value):
                    try:
                        del self.cache[key]
                    except KeyError:
                        pass
                    logger.debug("UrlCacheConnector metadata expired for namespace %s", namespace)
                    return None
                logger.debug("UrlCacheConnector metadata hit for namespace %s", namespace)
                return value
        except Exception as exc:
            logger.error("UrlCacheConnector metadata read failed for namespace %s: %s", namespace, exc)
        return None

    def set_namespace_metadata(self, namespace: str, metadata: Dict[str, Any]) -> None:
        key = self._make_meta_key(namespace)
        payload = {
            **metadata,
            "cached_at": time.time(),
        }
        try:
            self.cache[key] = payload
        except Exception as exc:
            logger.error("UrlCacheConnector metadata write failed for namespace %s: %s", namespace, exc)

    def clear_namespace(self, namespace: str) -> int:
        prefix = f"{self.environment}:{namespace}:"
        removed = 0
        try:
            keys_to_delete = [key for key in self.cache.keys() if key.startswith(prefix)]
            for key in keys_to_delete:
                try:
                    del self.cache[key]
                    removed += 1
                except KeyError:
                    continue
            meta_key = self._make_meta_key(namespace)
            if meta_key in self.cache:
                try:
                    del self.cache[meta_key]
                    removed += 1
                except KeyError:
                    pass
        except Exception as exc:
            logger.error("UrlCacheConnector namespace clear failed for %s: %s", namespace, exc)
        if removed:
            logger.info("UrlCacheConnector cleared %s entries for namespace %s", removed, namespace)
        return removed
