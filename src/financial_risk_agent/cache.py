from __future__ import annotations

import hashlib
import json
from typing import Any

import redis

from financial_risk_agent.config import CACHE_TTL_SECONDS, REDIS_URL

_MEMORY_CACHE: dict[str, Any] = {}


def stable_hash(payload: Any) -> str:
    """Create a stable hash for dict/list payloads."""
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class CacheClient:
    """Redis cache with automatic in-memory fallback.

    In production, Redis is the cache backend. In interviews/local demos, this still works even
    if Redis is not running because it falls back to process memory.
    """

    def __init__(self) -> None:
        self.redis_client: redis.Redis | None = None
        try:
            client = redis.from_url(REDIS_URL, decode_responses=True)
            client.ping()
            self.redis_client = client
        except Exception:
            self.redis_client = None

    @property
    def backend(self) -> str:
        return "redis" if self.redis_client else "memory"

    def get_json(self, key: str) -> Any | None:
        if self.redis_client:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        return _MEMORY_CACHE.get(key)

    def set_json(self, key: str, value: Any, ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
        if self.redis_client:
            self.redis_client.setex(key, ttl_seconds, json.dumps(value, default=str))
        else:
            _MEMORY_CACHE[key] = value

    def status(self) -> dict[str, Any]:
        return {"backend": self.backend, "redis_url": REDIS_URL}
