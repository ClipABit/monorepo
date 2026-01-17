import logging
import time
from typing import Optional, Dict, Any

import modal

logger = logging.getLogger(__name__)


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
        except Exception as exc:
            logger.error("VideoCache namespace clear failed for %s: %s", namespace, exc)
        if removed:
            logger.info("VideoCache cleared %s entries for namespace %s", removed, namespace)
        return removed
