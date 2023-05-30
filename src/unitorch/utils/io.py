# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import abc
from torch.multiprocessing import Process, Queue
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

GENERATE_FINISHED = "done"
POSTPROCESS_FINISHED = None


class GenericWriter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def process_start(self, chunk):
        pass

    @abc.abstractmethod
    def process_end(self):
        pass

    @abc.abstractmethod
    def process_chunk(self, chunk):
        pass


class IOProcess(Process):
    def __init__(self, msg_queue: Queue, writer: GenericWriter):
        super().__init__()
        self.msg_queue = msg_queue
        self.writer = writer
        self.waiting_for = 0
        self.chunk_buf = {}

    def process_start(self, chunk):
        self.writer.process_start(chunk)

    def process_end(self):
        self.writer.process_end()

    def process_chunk(self, chunk):
        self.writer.process_chunk(chunk)

    def process_buffer(self):
        while self.waiting_for in self.chunk_buf:
            self.process_chunk(self.chunk_buf[self.waiting_for])
            del self.chunk_buf[self.waiting_for]
            self.waiting_for += 1

    def run(self):
        while True:
            ind, chunk = self.msg_queue.get()
            if chunk == GENERATE_FINISHED:
                self.process_end()
                break
            if ind != self.waiting_for:
                self.chunk_buf[ind] = chunk
            else:
                if ind == 0:
                    self.process_start(chunk)
                else:
                    self.process_chunk(chunk)
                self.waiting_for += 1
                self.process_buffer()
        self.process_buffer()
        assert not self.chunk_buf, "IO Buffer not empty"
        self.msg_queue.close()
        self.msg_queue.join_thread()


class PostProcess(Process):
    def __init__(self, postprocess_fn: Callable, data_queue: Queue, msg_queue: Queue):
        super().__init__()
        self.data_queue = data_queue
        self.msg_queue = msg_queue
        self.postprocess_fn = postprocess_fn

    def run(self):
        while True:
            ind, outputs = self.data_queue.get()
            if outputs == GENERATE_FINISHED or outputs == POSTPROCESS_FINISHED:
                self.data_queue.put((-1, POSTPROCESS_FINISHED))
                break
            else:
                chunk = self.postprocess_fn(outputs)
                self.msg_queue.put((ind, chunk))

        self.data_queue.close()
        self.data_queue.join_thread()
        self.msg_queue.close()
        self.msg_queue.join_thread()
