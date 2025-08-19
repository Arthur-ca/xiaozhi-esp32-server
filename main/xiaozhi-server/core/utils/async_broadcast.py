# main/core/utils/async_broadcast.py
import asyncio
import contextlib
from typing import AsyncIterator, Generic, List, Optional, TypeVar

T = TypeVar("T")
_SENTINEL = object()

class AsyncBroadcastStream(Generic[T]):
    """把“只能一个消费者”的上游 async 迭代器安全 fan-out 给多下游。"""
    def __init__(self, source: AsyncIterator[T], max_queue: int = 256):
        self._source = source
        self._max_queue = max_queue
        self._queues: List[asyncio.Queue] = []
        self._producer: Optional[asyncio.Task] = None
        self._closed = asyncio.Event()

    def subscribe(self) -> AsyncIterator[T]:
        q: asyncio.Queue = asyncio.Queue(self._max_queue)
        self._queues.append(q)

        async def _iter():
            try:
                while True:
                    item = await q.get()
                    if item is _SENTINEL:
                        return
                    yield item  # type: ignore
            finally:
                with contextlib.suppress(ValueError):
                    self._queues.remove(q)

        if self._producer is None:
            self._producer = asyncio.create_task(self._run())
        return _iter()

    async def _run(self):
        try:
            async for item in self._source:
                for q in list(self._queues):
                    try:
                        q.put_nowait(item)
                    except asyncio.QueueFull:
                        with contextlib.suppress(Exception):
                            _ = q.get_nowait()
                        with contextlib.suppress(Exception):
                            q.put_nowait(item)
            for q in self._queues:
                await q.put(_SENTINEL)
        except Exception:
            for q in self._queues:
                with contextlib.suppress(Exception):
                    await q.put(_SENTINEL)
            raise
        finally:
            self._closed.set()

    async def aclose(self):
        if self._producer:
            self._producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._producer
        await self._closed.wait()
