# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import asyncio

import pytest

from unitorch.cli import Config, GenericFastAPI, PipelineReplicaPool


class DummyPipeline:
    def __init__(self, name="pipeline"):
        self.name = name
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return {
            "name": self.name,
            "value": value,
        }

    def __deepcopy__(self, memo):
        raise AssertionError("PipelineReplicaPool must not initialize replicas")


class DummyFastAPI(GenericFastAPI):
    def __init__(self, config, section=None):
        self.config = config
        if section is not None:
            self._section = section

    @property
    def router(self):
        return None

    def start(self):
        return "start success"

    def stop(self):
        return "stop success"


def test_pipeline_replica_pool_acquire_release_and_scheduling():
    first_pipeline = DummyPipeline("pipeline-0")
    second_pipeline = DummyPipeline("pipeline-1")
    pool = PipelineReplicaPool(
        [first_pipeline, second_pipeline],
        lock=True,
    )

    assert pool.num_replicas == 2
    assert pool.available_replicas == 2
    assert pool.busy_replicas == 0
    assert pool.replicas == [first_pipeline, second_pipeline]

    first = pool.acquire(block=False)
    second = pool.acquire(block=False)
    assert pool.available_replicas == 0
    assert pool.busy_replicas == 2
    with pytest.raises(TimeoutError):
        pool.acquire(block=False)

    assert first("a") == {"name": "pipeline-0", "value": "a"}
    first.release()
    assert pool.available_replicas == 1

    reused = pool.acquire(block=False)
    assert reused.index == first.index

    second.release()
    reused.release()
    assert pool.status() == {
        "num_replicas": 2,
        "available_replicas": 2,
        "busy_replicas": 0,
        "lock": True,
        "closed": False,
    }


def test_pipeline_replica_pool_direct_and_async_calls_release_replicas():
    pool = PipelineReplicaPool(DummyPipeline())

    assert pool("sync")["value"] == "sync"
    assert asyncio.run(pool.async_call("async"))["value"] == "async"

    lease = pool.acquire(block=False)
    assert lease("lease")["value"] == "lease"
    lease.release()


def test_pipeline_replica_pool_from_config_and_close():
    pipelines = [DummyPipeline("pipeline-0"), DummyPipeline("pipeline-1")]
    pool = PipelineReplicaPool.from_config(
        Config(
            params=[
                ("core/cli", "pipeline_replica_lock", "True"),
            ]
        ),
        pipelines,
    )

    assert pool.num_replicas == 2
    assert pool.replicas == pipelines

    pool.close()
    assert len(pool) == 0
    assert pool.closed is True
    assert pool.available_replicas == 0
    with pytest.raises(RuntimeError):
        pool.acquire(block=False)


def test_pipeline_replica_pool_without_lock_does_not_reserve_replicas():
    pool = PipelineReplicaPool(DummyPipeline(), lock=False)

    first = pool.acquire(block=False)
    second = pool.acquire(block=False)

    assert first.index == second.index == 0
    first.release()
    second.release()


def test_generic_fastapi_pipeline_pool_uses_config():
    pipelines = [DummyPipeline("pipeline-0"), DummyPipeline("pipeline-1")]
    service = DummyFastAPI(
        Config(
            params=[
                ("core/cli", "pipeline_replica_lock", "False"),
            ]
        )
    )

    pool = service.pipeline_pool(pipelines)

    assert pool.num_replicas == 2
    assert pool.available_replicas == 2
    assert pool.busy_replicas == 0


def test_generic_fastapi_pipeline_pool_prefers_service_section():
    pipelines = [DummyPipeline("pipeline-0"), DummyPipeline("pipeline-1")]
    service = DummyFastAPI(
        Config(
            params=[
                ("core/cli", "pipeline_replica_lock", "True"),
                ("core/fastapi/dummy", "pipeline_replica_lock", "False"),
            ]
        ),
        section="core/fastapi/dummy",
    )

    pool = service.pipeline_pool(pipelines)

    assert pool.status()["lock"] is False


def test_pipeline_replica_pool_rejects_empty_pipelines():
    with pytest.raises(ValueError):
        PipelineReplicaPool([])
    with pytest.raises(ValueError):
        PipelineReplicaPool([None])
