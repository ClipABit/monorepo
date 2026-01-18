import logging
import time
from typing import Optional, Dict, Any

import modal

logger = logging.getLogger(__name__)


# Cache entries must expire slightly before the R2 presigned URLs (1 hour) to avoid
# returning dead links; keep this under the URL TTL to prevent race conditions.
VIDEO_PAGE_TTL_SECONDS = 59 * 60  # 59 minutes


class VideoCache:
    """Modal Dict-backed cache for paginated video listings."""

    DEFAULT_DICT_NAME = "clipabit-video-cache"

    def __init__(self, environment: str, dict_name: str = DEFAULT_DICT_NAME):
        self.environment = environment
        self.dict_name = dict_name
        self.cache = modal.Dict.from_name(dict_name, create_if_missing=True)
        logger.info(
            "Initialized VideoCache with Dict '%s' for environment '%s'",
            dict_name,
            environment,
        )

    def _make_key(self, namespace: str, page_token: Optional[str], page_size: int) -> str:
        token = page_token or ""
        return f"{self.environment}:{namespace}:{page_size}:{token}"

    def _make_meta_key(self, namespace: str) -> str:
        return f"{self.environment}:{namespace}:meta"

    def get_page(
        self,
        namespace: str,
        page_token: Optional[str],
        page_size: int,
    ) -> Optional[Dict[str, Any]]:
        """Return cached page payload if present."""
        key = self._make_key(namespace, page_token, page_size)
        try:
            if key in self.cache:
                value = self.cache[key]
                # Expire entries that outlive the presigned URL window.
                cached_at = value.get("cached_at") if isinstance(value, dict) else None
                if cached_at and (time.time() - cached_at) > VIDEO_PAGE_TTL_SECONDS:
                    try:
                        del self.cache[key]
                    except KeyError:
                        pass
                    logger.debug("VideoCache entry expired for key %s", key)
                    return None
                logger.debug("VideoCache hit for key %s", key)
                return value
        except Exception as exc:
            logger.error("VideoCache read failed for key %s: %s", key, exc)
        return None

    def set_page(
        self,
        namespace: str,
        page_token: Optional[str],
        page_size: int,
        videos: list[Dict[str, Any]],
        next_token: Optional[str],
    ) -> None:
        """Cache page payload."""
        key = self._make_key(namespace, page_token, page_size)
        payload = {
            "videos": videos,
            "next_token": next_token,
            "cached_at": time.time(),
        }
        try:
            self.cache[key] = payload
            logger.debug("VideoCache stored entry for key %s", key)
        except Exception as exc:
            logger.error("VideoCache write failed for key %s: %s", key, exc)

    def get_namespace_metadata(self, namespace: str) -> Optional[Dict[str, Any]]:
        """Return cached namespace-level metadata (e.g., totals)."""
        key = self._make_meta_key(namespace)
        try:
            if key in self.cache:
                value = self.cache[key]
                cached_at = value.get("cached_at") if isinstance(value, dict) else None
                if cached_at and (time.time() - cached_at) > VIDEO_PAGE_TTL_SECONDS:
                    try:
                        del self.cache[key]
                    except KeyError:
                        pass
                    logger.debug("VideoCache metadata expired for key %s", key)
                    return None
                logger.debug("VideoCache metadata hit for key %s", key)
                return value
        except Exception as exc:
            logger.error("VideoCache metadata read failed for namespace %s: %s", namespace, exc)
        return None

    def set_namespace_metadata(self, namespace: str, metadata: Dict[str, Any]) -> None:
        """Cache namespace-level metadata (e.g., totals)."""
        key = self._make_meta_key(namespace)
        payload = {
            **metadata,
            "cached_at": time.time(),
        }
        try:
            self.cache[key] = payload
            logger.debug("VideoCache stored metadata for namespace %s", namespace)
        except Exception as exc:
            logger.error("VideoCache metadata write failed for namespace %s: %s", namespace, exc)

    def clear_namespace(self, namespace: str) -> int:
        """Remove all cached pages for a namespace.

        Returns number of entries removed.
        """
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
            # Remove namespace metadata separately to ensure clean slate
            meta_key = self._make_meta_key(namespace)
            if meta_key in self.cache:
                try:
                    del self.cache[meta_key]
                    removed += 1
                except KeyError:
                    pass
        except Exception as exc:
            logger.error("VideoCache namespace clear failed for %s: %s", namespace, exc)
        if removed:
            logger.info("VideoCache cleared %s entries for namespace %s", removed, namespace)
        return removed
