"""Real CPU/Gloo ZeRO checkpoint initialization, resume and export checks."""

import json
import os
from pathlib import Path
import subprocess
import sys

import fire
import torch

from flux2_test_utils import model_class, model_kwargs, write_cli_config


def main(assets, output, task="text2image", mode="lora", workflow="initialize", variant="klein-4b", epochs=1):
    import deepspeed
    from unitorch.cli import Config
    from unitorch.cli.tasks.deepspeed import DeepspeedTask

    torch.set_num_threads(2)
    assets, output = Path(assets), Path(output)
    rank = int(os.environ["RANK"])
    os.environ["LOCAL_RANK"] = "-1"
    deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    deepspeed.init_distributed(dist_backend="gloo", auto_mpi_discovery=False)
    extra = {"lora_r": 2, "lora_alpha": 2, "target_modules": ["to_q"]} if mode == "lora" else {}
    kwargs = dict(**model_kwargs(assets), torch_dtype="float32", **extra)
    source_dir = output / "source"
    if rank == 0:
        source_dir.mkdir(exist_ok=True)
        source = model_class(task, mode)(**kwargs)
        with torch.no_grad():
            for parameter in source.parameters():
                if parameter.requires_grad:
                    parameter.add_(0.125)
        source.save_checkpoint(str(source_dir))
        del source
        config_path = write_cli_config(output, assets, variant, mode, task)
        config = config_path.read_text().replace("core/task/supervised", "core/task/deepspeed/supervised")
        config = config.replace(f"{output}/no-initial-checkpoint", str(source_dir))
        config = config.replace("lora_r = 2", "target_modules = ['to_q']\nlora_r = 2")
        config += f"\nconfig_path = {output}/zero.json\nzero_init = True\ncpu_offload = True\nmerge_zero3_checkpoint = False\n"
        config_path.write_text(config)
        (output / "zero.json").write_text(json.dumps({
            "train_micro_batch_size_per_gpu": 2, "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 3, "offload_param": {"device": "cpu"},
                                  "offload_optimizer": {"device": "cpu"}, "stage3_param_persistence_threshold": 0},
            "optimizer": {"type": "AdamW", "params": {"lr": 0.003, "torch_adam": True}},
            "zero_force_ds_cpu_optimizer": False, "zero_allow_untested_optimizer": True,
        }))
    deepspeed.comm.barrier()
    runner = DeepspeedTask.from_config(Config(str(output / "train.ini")))
    model = runner.model.model
    expected = torch.load(source_dir / "pytorch_model.bin", weights_only=True)
    original_load = runner.model.from_checkpoint
    checked = []

    def checked_load(*args, **kwargs):
        original_load(*args, **kwargs)
        for name, parameter in model.named_parameters():
            assert hasattr(parameter, "ds_shape")
            with deepspeed.zero.GatheredParameters([parameter]):
                if name in expected:
                    torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0, msg=name)
        checked.append(True)

    runner.model.from_checkpoint = checked_load
    # Verify native parameter restoration immediately after the real engine
    # loader returns, before the second optimizer update can hide a bad reload.
    native_load = deepspeed.DeepSpeedEngine.load_checkpoint
    resumed = []

    def checked_native_load(engine, *args, **kwargs):
        result = native_load(engine, *args, **kwargs)
        previous = torch.load(output / f"expected-{epochs - 1}.bin", weights_only=True)
        for name, parameter in model.named_parameters():
            if name in previous:
                with deepspeed.zero.GatheredParameters([parameter]):
                    torch.testing.assert_close(parameter, previous[name], rtol=0, atol=0, msg=name)
        resumed.append(True)
        return result

    if workflow == "export" and epochs > 1:
        deepspeed.DeepSpeedEngine.load_checkpoint = checked_native_load
    try:
        runner.train(epochs=epochs)
    finally:
        deepspeed.DeepSpeedEngine.load_checkpoint = native_load
    if workflow == "export" and epochs > 1:
        assert resumed

    assert checked, "DeepspeedTask must initialize from the populated from_ckpt_dir"
    if workflow == "export":
        assert runner.model.global_steps == epochs
        steps = [int(state["step"]) for state in runner.model.optimizer.optimizer.state.values()]
        assert steps and all(step == epochs for step in steps)
        trained = {}
        for name, parameter in model.named_parameters():
            with deepspeed.zero.GatheredParameters([parameter]):
                if parameter.requires_grad and rank == 0:
                    trained[name] = parameter.detach().cpu().clone()
        if rank == 0:
            torch.save(trained, output / f"expected-{epochs}.bin")
            from unitorch.models.diffusers.checkpoint_flux2 import export_zero_checkpoint

            native = output / "checkpoint/pytorch_model_latest"
            for dtype in ("float32", "bfloat16"):
                destination = output / f"export-{epochs}-{dtype}"
                result = export_zero_checkpoint(str(native), str(destination), mode=mode, dtype=dtype)
                exported = torch.load(result["path"], weights_only=True)
                assert exported.keys() == trained.keys()
                for name, value in trained.items():
                    torch.testing.assert_close(exported[name], value.to(getattr(torch, dtype)), rtol=0, atol=0)
                restored = model_class(task, mode)(**kwargs)
                restored.from_checkpoint(str(destination))
                for name, value in restored.checkpoint_state_dict().items():
                    torch.testing.assert_close(value, exported[name].float(), rtol=0, atol=0)
            if epochs == 1:
                cli_output = output / "cli-export"
                subprocess.run([
                    sys.executable, "-m", "unitorch.cli.scripts.flux2",
                    f"--checkpoint={native}", f"--output={cli_output}",
                    f"--mode={mode}", "--dtype=float32",
                ], check=True, timeout=60)
                cli_state = torch.load(cli_output / "pytorch_model.bin", weights_only=True)
                assert cli_state.keys() == trained.keys()
                for key, value in trained.items():
                    torch.testing.assert_close(cli_state[key], value, rtol=0, atol=0)
            try:
                export_zero_checkpoint(str(native), str(destination), mode=mode)
            except FileExistsError:
                pass
            else:
                raise AssertionError("export replaced an existing checkpoint")
            # Exercise resource guards against actual native tensors, before writing.
            try:
                export_zero_checkpoint(str(native), str(output / "denied"), mode=mode, max_memory_bytes=1)
            except RuntimeError as exc:
                assert "memory" in str(exc).lower()
            else:
                raise AssertionError("export ignored memory limit")
            assert not (output / "denied/pytorch_model.bin").exists()
    deepspeed.comm.barrier()
    deepspeed.comm.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
