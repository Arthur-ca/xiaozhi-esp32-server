# main/core/providers/llm/rag/rerank_batcher.py
import os
import time
import threading
import queue
from typing import List, Tuple, Sequence

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

_BATCH_MAX       = int(os.getenv("RAG_RERANK_MAX_BATCH", "32"))
_BATCH_WIN_MS    = int(os.getenv("RAG_RERANK_LATENCY_MS", "20"))
_RERANK_TIMEOUT  = int(os.getenv("RAG_RERANK_TIMEOUT_MS", "5000"))  # 新增：等待超时（毫秒）
_GPU_CONCURRENCY = int(os.getenv("RAG_GPU_CONCURRENCY", "2"))
_CPU_CONCURRENCY = int(os.getenv("RAG_CPU_CONCURRENCY", "8"))

_GPU_SEMA = threading.Semaphore(_GPU_CONCURRENCY)
_CPU_SEMA = threading.Semaphore(_CPU_CONCURRENCY)

class _BatchItem:
    def __init__(self, pairs: Sequence[Tuple[str, str]]):
        self.pairs = list(pairs)
        self._evt = threading.Event()
        self.result: List[float] | None = None

    def set_result(self, scores: List[float]):
        self.result = scores
        self._evt.set()

    def wait(self) -> List[float]:
        # 新增：超时兜底，避免主线程“永等”
        ok = self._evt.wait(_RERANK_TIMEOUT / 1000.0)
        if not ok:
            return [0.0] * len(self.pairs)
        return self.result or [0.0] * len(self.pairs)

class _RerankBatcher:
    def __init__(self):
        self._q: "queue.Queue[_BatchItem]" = queue.Queue()
        self._worker = None
        self._model = None
        self._model_name = None
        self._device = None
        self._lock = threading.Lock()
        self._ensure_worker()

    def _ensure_worker(self):
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _ensure_model(self, model_name: str, device: str | None):
        if CrossEncoder is None:
            return
        with self._lock:
            if self._model is None or self._model_name != model_name or self._device != device:
                args = {}
                if device and device != "auto":
                    args["device"] = device
                try:
                    # 新版本：用 automodel_args 传 trust_remote_code
                    self._model = CrossEncoder(model_name, **args, automodel_args={"trust_remote_code": True})
                except TypeError:
                    # 老版本：不支持 automodel_args
                    self._model = CrossEncoder(model_name, **args)
                except Exception:
                    self._model = None
                    self._model_name = None
                    self._device = None
                    return
                self._model_name = model_name
                self._device = device

    def submit(self, pairs: Sequence[Tuple[str, str]], model_name: str, device: str | None) -> List[float]:
        self._ensure_model(model_name, device)
        item = _BatchItem(pairs)
        # 若 worker 异常退出，仍可能 put 成功但 loop 不取；wait 有超时兜底
        try:
            self._q.put(item, timeout=1.0)
        except Exception:
            return [0.0] * len(pairs)
        return item.wait()

    def _loop(self):
        while True:
            try:
                first = self._q.get()
            except Exception:
                time.sleep(0.001)
                continue

            bucket = [first]
            deadline = time.time() + (_BATCH_WIN_MS / 1000.0)
            while len(bucket) < _BATCH_MAX and time.time() < deadline:
                try:
                    bucket.append(self._q.get_nowait())
                except queue.Empty:
                    time.sleep(0.001)
                    break

            try:
                flat_pairs: List[Tuple[str, str]] = [p for it in bucket for p in it.pairs]
                scores = self._predict(flat_pairs)
            except Exception:
                # 任意异常：给本批次全部返回 0 分，避免阻断上层
                scores = [0.0] * sum(len(it.pairs) for it in bucket)

            idx = 0
            for it in bucket:
                it.set_result(scores[idx: idx + len(it.pairs)])
                idx += len(it.pairs)

    def _predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if CrossEncoder is None or self._model is None:
            return [0.0] * len(pairs)
        with _GPU_SEMA:
            try:
                return self._model.predict(pairs, convert_to_numpy=True).tolist()
            except Exception:
                return [0.0] * len(pairs)

# 单例
_BATCHER = _RerankBatcher()

def rerank_with_scores_batching(
    query: str,
    docs: Sequence[object],
    top_n: int,
    model_name: str,
    device: str | None,
) -> List[Tuple[object, float]]:
    pairs = [(query, getattr(d, "page_content", str(d))) for d in docs]
    scores = _BATCHER.submit(pairs, model_name=model_name, device=device)
    items = list(zip(docs, [float(s) for s in scores]))
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_n]
