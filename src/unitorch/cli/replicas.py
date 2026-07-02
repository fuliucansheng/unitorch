# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import queue
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


@dataclass
class _PipelineReplica:
    index: int
    pipeline: Any


class PipelineReplicaLease:
    """
    Leased pipeline replica returned by :class:`PipelineReplicaPool.acquire`.

    The lease proxies attribute access and calls to the underlying pipeline, and
    must be released after use unless it is used as a context manager.
    """

    def __init__(
        self,
        pool: "PipelineReplicaPool",
        replica: _PipelineReplica,
        release_to_pool: bool = True,
    ):
        self._pool = pool
        self._replica = replica
        self._release_to_pool = release_to_pool
        self._released = False

    @property
    def index(self) -> int:
        return self._replica.index

    @property
    def pipeline(self) -> Any:
        return self._replica.pipeline

    def __getattr__(self, name: str) -> Any:
        return getattr(self._replica.pipeline, name)

    def __call__(self, *args, **kwargs) -> Any:
        return self._replica.pipeline(*args, **kwargs)

    def __enter__(self) -> "PipelineReplicaLease":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    async def __aenter__(self) -> "PipelineReplicaLease":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._pool.release(self)


class PipelineReplicaPool:
    """
    Schedule one or more already-initialized callable pipelines.

    The pool only manages locking and round-robin scheduling. Pipeline
    construction and lifecycle management must happen before pipelines are
    passed in.
    """

    def __init__(
        self,
        pipelines: Any,
        lock: bool = True,
    ):
        self._lock_enabled = _parse_bool(lock)
        self._closed = False
        self._cursor = 0
        self._state_lock = threading.RLock()
        self._replicas = self._init_replicas(pipelines)
        self._available = (
            queue.Queue(maxsize=len(self._replicas)) if self._lock_enabled else None
        )

        if self._available is not None:
            for replica in self._replicas:
                self._available.put(replica)

    @property
    def num_replicas(self) -> int:
        return len(self._replicas)

    @property
    def replicas(self) -> List[Any]:
        return [replica.pipeline for replica in self._replicas]

    @property
    def available_replicas(self) -> int:
        if self._closed:
            return 0
        if not self._lock_enabled:
            return len(self._replicas)
        return self._available.qsize()

    @property
    def busy_replicas(self) -> int:
        if self._closed or not self._lock_enabled:
            return 0
        return len(self._replicas) - self.available_replicas

    @property
    def closed(self) -> bool:
        return self._closed

    def status(self) -> dict:
        return {
            "num_replicas": len(self._replicas),
            "available_replicas": self.available_replicas,
            "busy_replicas": self.busy_replicas,
            "lock": self._lock_enabled,
            "closed": self._closed,
        }

    def _init_replicas(self, pipelines: Any) -> List[_PipelineReplica]:
        if pipelines is None:
            raise ValueError("pipelines must be provided")
        if self._is_pipeline_sequence(pipelines):
            pipeline_list = list(pipelines)
        else:
            pipeline_list = [pipelines]
        if not pipeline_list:
            raise ValueError("pipelines must contain at least one pipeline")
        if any(pipeline is None for pipeline in pipeline_list):
            raise ValueError("pipelines must not contain None")
        return [
            _PipelineReplica(index=index, pipeline=pipeline)
            for index, pipeline in enumerate(pipeline_list)
        ]

    def _is_pipeline_sequence(self, pipelines: Any) -> bool:
        if isinstance(pipelines, (str, bytes, bytearray)):
            return False
        return isinstance(pipelines, Sequence)

    @classmethod
    def from_config(
        cls,
        config: Any,
        pipelines: Any = None,
        section: str = "core/cli",
        lock: Optional[bool] = None,
    ) -> "PipelineReplicaPool":
        if lock is None and config is not None:
            lock = config.getdefault(section, "pipeline_replica_lock", True)
        return cls(pipelines=pipelines, lock=_parse_bool(lock))

    def acquire(
        self,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> PipelineReplicaLease:
        if self._closed:
            raise RuntimeError("PipelineReplicaPool is closed")

        if not self._lock_enabled:
            with self._state_lock:
                replica = self._replicas[self._cursor]
                self._cursor = (self._cursor + 1) % len(self._replicas)
            return PipelineReplicaLease(self, replica, release_to_pool=False)

        try:
            replica = self._available.get(block=block, timeout=timeout)
        except queue.Empty as e:
            raise TimeoutError("No available pipeline replicas") from e
        return PipelineReplicaLease(self, replica, release_to_pool=True)

    async def acquire_async(
        self,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> PipelineReplicaLease:
        return self.acquire(block=block, timeout=timeout)

    def release(self, lease: PipelineReplicaLease) -> None:
        if lease._pool is not self:
            raise ValueError("Cannot release a pipeline lease to a different pool")
        if not lease._release_to_pool or self._closed:
            return
        try:
            self._available.put_nowait(lease._replica)
        except queue.Full:
            pass

    def __call__(self, *args, **kwargs) -> Any:
        with self.acquire() as pipeline:
            return pipeline(*args, **kwargs)

    async def async_call(self, *args, **kwargs) -> Any:
        return self.__call__(*args, **kwargs)

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
            if self._available is not None:
                while True:
                    try:
                        self._available.get_nowait()
                    except queue.Empty:
                        break
            self._replicas.clear()

    def __len__(self) -> int:
        return len(self._replicas)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off", ""}:
            return False
        raise ValueError(f"Unsupported boolean value: {value!r}.")
    return bool(value)
