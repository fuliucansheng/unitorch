# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import abc
from typing import Any, Callable

from torch.multiprocessing import Process, Queue

GENERATE_FINISHED = "done"
POSTPROCESS_FINISHED = None


class GenericWriter(abc.ABC):
    """Abstract base class for streaming chunk writers."""

    @abc.abstractmethod
    def process_start(self, chunk: Any) -> None:
        """Handle the very first chunk of a new stream."""

    @abc.abstractmethod
    def process_chunk(self, chunk: Any) -> None:
        """Handle a subsequent chunk."""

    @abc.abstractmethod
    def process_end(self) -> None:
        """Finalise the stream after all chunks have been processed."""


class IOProcess(Process):
    """A worker process that writes ordered chunks from a multiprocessing queue.

    Chunks may arrive out of order.  They are buffered and flushed in index
    order so that ``GenericWriter`` always receives chunks sequentially.
    """

    def __init__(self, msg_queue: Queue, writer: GenericWriter) -> None:
        super().__init__()
        self.msg_queue = msg_queue
        self.writer = writer
        self._waiting_for: int = 0
        self._chunk_buf: dict = {}

    def process_start(self, chunk: Any) -> None:
        self.writer.process_start(chunk)

    def process_chunk(self, chunk: Any) -> None:
        self.writer.process_chunk(chunk)

    def process_end(self) -> None:
        self.writer.process_end()

    def _flush_buffer(self) -> None:
        """Deliver buffered chunks in order for as long as the next index is ready."""
        while self._waiting_for in self._chunk_buf:
            self.process_chunk(self._chunk_buf.pop(self._waiting_for))
            self._waiting_for += 1

    def run(self) -> None:
        while True:
            ind, chunk = self.msg_queue.get()
            if chunk == GENERATE_FINISHED:
                self.process_end()
                break
            if ind != self._waiting_for:
                self._chunk_buf[ind] = chunk
            else:
                if ind == 0:
                    self.process_start(chunk)
                else:
                    self.process_chunk(chunk)
                self._waiting_for += 1
                self._flush_buffer()

        self._flush_buffer()
        assert not self._chunk_buf, "IO buffer not empty after stream finished."
        self.msg_queue.close()
        self.msg_queue.join_thread()


class PostProcess(Process):
    """A worker process that applies a post-processing function to raw outputs.

    Reads ``(index, outputs)`` pairs from *data_queue*, applies
    *postprocess_fn*, and forwards ``(index, chunk)`` to *msg_queue*.
    Terminates when it receives :data:`GENERATE_FINISHED` or
    :data:`POSTPROCESS_FINISHED` as the outputs value.
    """

    def __init__(
        self,
        postprocess_fn: Callable[[Any], Any],
        data_queue: Queue,
        msg_queue: Queue,
    ) -> None:
        super().__init__()
        self.postprocess_fn = postprocess_fn
        self.data_queue = data_queue
        self.msg_queue = msg_queue

    def run(self) -> None:
        while True:
            ind, outputs = self.data_queue.get()
            if outputs in (GENERATE_FINISHED, POSTPROCESS_FINISHED):
                self.data_queue.put((-1, POSTPROCESS_FINISHED))
                break
            self.msg_queue.put((ind, self.postprocess_fn(outputs)))

        self.data_queue.close()
        self.data_queue.join_thread()
        self.msg_queue.close()
        self.msg_queue.join_thread()
