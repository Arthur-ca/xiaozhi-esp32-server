from __future__ import annotations
import time

class SimpleCache:
    def __init__(self, max_size=100, ttl=3600):
        self.max_size, self.ttl = max_size, ttl
        self._d = {}

    def get(self, k):
        v = self._d.get(k)
        if not v: return None
        if time.time() - v["ts"] > self.ttl:
            self._d.pop(k, None); return None
        return v["val"]

    def set(self, k, val):
        if len(self._d) >= self.max_size:
            oldest = min(self._d.items(), key=lambda kv: kv[1]["ts"])[0]
            self._d.pop(oldest, None)
        self._d[k] = {"val": val, "ts": time.time()}
