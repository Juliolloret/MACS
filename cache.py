import os
import json
import hashlib
from threading import Lock
from typing import Any, Optional, Tuple

class Cache:
    """Simple JSON-backed cache for deterministic computations."""

    def __init__(self, cache_file: Optional[str] = None):
        self.cache_file = cache_file
        self._lock = Lock()
        self.data = {}
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except Exception:
                # Corrupted cache is ignored
                self.data = {}

    @staticmethod
    def make_key(*parts: Any) -> str:
        """Create a stable hash key from arbitrary components."""
        serialized = json.dumps(parts, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.data[key] = value
            if self.cache_file:
                tmp = self.cache_file + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.cache_file)

class CachingEmbeddings:
    """Wrap an embeddings client with memoization."""

    def __init__(self, embeddings_client: Any, cache: Cache):
        self._embeddings = embeddings_client
        self._cache = cache

    def embed_query(self, text: str):
        key = self._cache.make_key('embed_query', text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._embeddings.embed_query(text)
        self._cache.set(key, result)
        return result

    def embed_documents(self, texts: list[str]):
        results: list[Any] = [None] * len(texts)
        missing = []
        missing_idx = []
        for idx, text in enumerate(texts):
            key = self._cache.make_key('embed_doc', text)
            cached = self._cache.get(key)
            if cached is None:
                missing.append(text)
                missing_idx.append((idx, key))
            else:
                results[idx] = cached
        if missing:
            computed = self._embeddings.embed_documents(missing)
            for (idx, key), emb in zip(missing_idx, computed):
                self._cache.set(key, emb)
                results[idx] = emb
        return results
