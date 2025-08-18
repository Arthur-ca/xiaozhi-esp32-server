from __future__ import annotations
import time
from typing import Dict, Any
from contextlib import contextmanager

class StageTimer:
    def __init__(self): self._t: Dict[str, float] = {}
    def tic(self, name: str): self._t[name] = time.perf_counter()
    def toc(self, name: str) -> float:
        t0 = self._t.get(name)
        return float(time.perf_counter() - t0) if t0 else 0.0
    def summary(self, *names: str) -> str:
        parts = []
        for n in names:
            try: parts.append(f"{n}={self.toc(n):.3f}s")
            except Exception: parts.append(f"{n}=n/a")
        return " | ".join(parts)

def numeric_timings(t: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in t.items():
        if isinstance(v, (int, float)):
            try: out[k] = float(v)
            except Exception: pass
    return out

@contextmanager
def log_stage(logger, timings: Dict[str, float], name: str):
    t0 = time.perf_counter()
    logger.info(f"[RAG] {name} 进行中")
    try:
        yield
    finally:
        dt = float(time.perf_counter() - t0)
        timings[name] = dt
        logger.info(f"[RAG] done ({dt:.3f}s)")