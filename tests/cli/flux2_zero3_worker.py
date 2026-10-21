"""Isolated real ZeRO-3/Gloo regression; invoked by pytest using Fire."""

import json
import os
from pathlib import Path

import fire
import torch

from flux2_test_utils import batch_inputs, model_class, model_kwargs, write_cli_config


def main(assets, output, variant="klein-4b", mode="e2e", entry="package",
         task_name="text2image", workflow="initialization", epochs=1,
         force_cuda_available=False):
    import deepspeed

    torch.set_num_threads(2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assets = Path(assets)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    os.environ["LOCAL_RANK"] = "-1"
    # Exercise native Gloo collectives on CPU, without the optional SHM JIT op.
    deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    deepspeed.init_distributed(
        dist_backend="gloo", init_method=("env://" if world_size > 1 else f"file://{output}/rendezvous"),
        rank=rank, world_size=world_size, auto_mpi_discovery=False,
    )
    ds_config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": False},
            "offload_optimizer": {"device": "cpu", "pin_memory": False},
            "stage3_param_persistence_threshold": 0,
        },
        "optimizer": {"type": "AdamW", "params": {"lr": 0.003, "torch_adam": True}},
        "zero_force_ds_cpu_optimizer": False,
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": False},
    }
    ds_path = output / "zero3.json"
    if rank == 0:
        ds_path.write_text(json.dumps(ds_config))
    deepspeed.comm.barrier()
    kwargs = model_kwargs(assets)
    kwargs["torch_dtype"] = "float32"
    # Finish optional-backend imports before forcing CUDA availability on CPU CI.
    import unitorch.cli

    if force_cuda_available:
        assert deepspeed.accelerator.get_accelerator().device_name() == "cpu"
        torch.cuda.is_available = lambda: True
        # CPU Adam's optional CUDA graph health probe needs no CUDA runtime.
        torch.cuda.is_current_stream_capturing = lambda: False

        def unexpected_cuda(*args, **kwargs):
            raise AssertionError("CPU DeepSpeed execution must not move tensors to CUDA")

        torch.Tensor.cuda = unexpected_cuda
    if entry == "cli":
        from unitorch.cli import Config
        from unitorch.cli.tasks.deepspeed import DeepspeedTask

        config_path = output / "train.ini"
        if rank == 0:
            write_cli_config(output, assets, variant, mode, task_name)
            text = config_path.read_text().replace("core/task/supervised", "core/task/deepspeed/supervised")
            text += f"\nconfig_path = {ds_path}\nzero_init = True\ncpu_offload = True\n"
            if workflow == "inference":
                text = text.replace("zero_init = True", "zero_init = False")
                text = text.replace("cpu_offload = True", "cpu_offload = False")
            config_path.write_text(text)
        deepspeed.comm.barrier()
        task = DeepspeedTask.from_config(Config(str(config_path)))
        model = task.model
        package_model = model.model
    else:
        with deepspeed.zero.Init(config_dict_or_path=ds_config, remote_device="cpu"):
            model = model_class(task_name, mode)(**kwargs)
        package_model = model
        package_model.from_pretrained(weight_path=str(assets / "initial.bin"))

    if workflow == "inference":
        from unitorch.cli.models import TensorInputs
        from unitorch.cli.models.classification_utils import ClassificationTargets
        from unitorch.cli.tasks.supervised import infer

        assert all(p.device.type == "cpu" for p in model.parameters())
        # Also exercise non-loss output/target Gloo aggregation with real generation.
        model.eval()
        inputs = {k: v for k, v in batch_inputs(task_name).items() if k != "pixel_values"}
        targets = ClassificationTargets(targets=torch.arange(2), sample_weight=torch.ones(2))
        result = infer(model, [(TensorInputs(**inputs), targets)],
                       device=task.device)
        assert torch.isfinite(result.outputs.outputs).all()
        torch.testing.assert_close(result.targets.targets, torch.arange(2))
        for metadata in (True, False):
            if not metadata:
                dataset = task.datasets.get("test")
                task.datasets = {"test": [dataset[index] for index in range(len(dataset))]}
            predictions = output / f"predictions-{metadata}.jsonl"
            task.infer(output_path=str(predictions))
            rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            assert len(rows) == 2
            for row in rows:
                from PIL import Image
                assert Image.open(output / "images" / row["diffusion"]).size == (16, 16)
        (output / "result.json").write_text(json.dumps({"status": "passed"}))
        deepspeed.comm.destroy_process_group()
        return

    assert all(hasattr(p, "ds_id") for p in model.parameters()), "ZeRO-3 must partition at construction, before GPU placement."
    assert all(p.ds_tensor.numel() == (p.ds_numel + world_size - 1) // world_size for p in model.parameters())
    expected = torch.load(assets / "initial.bin", weights_only=True)
    (output / f"constructed-parameters-rank{rank}.json").write_text(json.dumps({
        name: {"dtype": str(parameter.dtype), "shape": list(getattr(parameter, "ds_shape", parameter.shape))}
        for name, parameter in package_model.named_parameters()
    }, indent=2))
    for name, parameter in package_model.named_parameters():
        with deepspeed.zero.GatheredParameters([parameter]):
            if ".lora_B." in name:
                assert torch.count_nonzero(parameter) == 0
            elif ".lora_A." in name:
                assert torch.count_nonzero(parameter) > 0
            else:
                torch.testing.assert_close(parameter, expected[name.replace(".base_layer.", ".")], rtol=0, atol=0, msg=name)

    if workflow == "checkpoint":
        task.train(epochs=epochs)
        task.eval(monitor_fns=[], from_ckpt_dir=str(output / "no-weights"),
                  pin_memory=False, num_workers=0)
        engine = task.model
        checkpoint = output / "checkpoint"
        info = json.loads((checkpoint / "info.json").read_text())
        assert info["global_epoch"] == epochs
        assert engine.global_steps == epochs
        steps = [int(state["step"]) for state in engine.optimizer.optimizer.state.values()]
        assert steps and all(step == epochs for step in steps)
        errors = []
        for filename in ("pytorch_model.bin", "pytorch_model_latest.bin"):
            if not (checkpoint / filename).is_file():
                errors.append(f"Missing consolidated checkpoint: {filename}")
                continue
            state = torch.load(checkpoint / filename, weights_only=True)
            assert state and all(torch.isfinite(value).all() for value in state.values())
            if mode == "lora":
                if not all(".lora_" in key for key in state):
                    errors.append(f"Frozen base buffers included in adapter export: {filename}")
            else:
                if not all(key.startswith("transformer.") for key in state):
                    errors.append(f"Frozen conditioners included in transformer export: {filename}")
        for path in checkpoint.glob("*/*/*model_states.pt"):
            saved = torch.load(path, weights_only=False)
            if saved.get("frozen_param_fragments"):
                errors.append(f"Native checkpoint duplicates the immutable base: {path}")
        latest_tags = [p for p in (checkpoint / "pytorch_model_latest").iterdir() if p.is_dir()]
        if len(latest_tags) != 1:
            errors.append("Latest checkpoints accumulate optimizer shards.")
        trained = torch.load(checkpoint / "pytorch_model_latest.bin", weights_only=True)
        for name, parameter in package_model.named_parameters():
            with deepspeed.zero.GatheredParameters([parameter]):
                if parameter.requires_grad:
                    torch.testing.assert_close(parameter, trained[name], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(parameter, expected[name.replace(".base_layer.", ".")],
                                               rtol=0, atol=0, msg=name)
        (output / f"result-{epochs}.json").write_text(json.dumps({
            "status": "failed" if errors else "passed", "engine_global_steps": engine.global_steps,
            "optimizer_steps": steps, "global_epoch": info["global_epoch"],
            "execution_device": str(task.device), "cuda_available": torch.cuda.is_available(),
            "forced_cuda_available": force_cuda_available,
        }, indent=2))
        assert not errors, "\n".join(errors)
        deepspeed.comm.destroy_process_group()
        return

    engine, optim, _, _ = deepspeed.initialize(
        model=model, model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=ds_config, dist_init_required=False,
    )
    inputs = batch_inputs(task_name)
    output_loss = engine(**inputs)
    loss = output_loss.loss if hasattr(output_loss, "loss") else output_loss
    assert torch.isfinite(loss)
    engine.backward(loss)
    engine.step()
    engine.save_checkpoint(str(output / "checkpoint"))
    if rank != 0:
        deepspeed.comm.barrier()
        deepspeed.comm.destroy_process_group()
        return
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    from unitorch.cli.tasks.deepspeed import _strip_model_prefix

    state = _strip_model_prefix(get_fp32_state_dict_from_zero_checkpoint(
        str(output / "checkpoint"), exclude_frozen_parameters=True,
    ))
    state = package_model.checkpoint_state_dict(state)
    assert state and all(torch.isfinite(p).all() for p in state.values())
    if mode == "lora":
        assert all(".lora_" in key for key in state)
        assert any(torch.count_nonzero(value) for key, value in state.items() if ".lora_B." in key)
    else:
        assert any(not torch.equal(value, expected[key]) for key, value in state.items())
    (output / "result.json").write_text(json.dumps({
        "variant": variant, "mode": mode, "entry": entry,
        "loss": loss.item(), "exported_tensors": len(state),
        "status": "passed", "backend": "real DeepSpeed ZeRO-3 + CPU Gloo", "world_size": world_size,
    }, indent=2))
    deepspeed.comm.barrier()
    deepspeed.comm.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
