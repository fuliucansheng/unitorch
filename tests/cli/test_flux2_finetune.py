"""Behavioral regressions for the two exact FLUX.2 finetuning variants."""

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from diffusers import Flux2Pipeline, Flux2KleinPipeline
from PIL import Image

from flux2_test_utils import (
    MODES, TASKS, VARIANTS, batch_inputs, make_assets, model_class,
    model_kwargs, processor_kwargs,
    write_cli_config,
)


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("flux2-real-components")
    return {variant: make_assets(root / variant, variant) for variant in VARIANTS}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("task", TASKS)
def test_training_matrix(assets, variant, mode, task, tmp_path):
    """Every case must really optimize, reload, infer, then resume optimizing."""
    cls = model_class(task, mode)
    kwargs = model_kwargs(assets[variant])
    if mode == "lora":
        kwargs.update(lora_r=2, lora_alpha=2, lora_dropout=0.0)
    torch.manual_seed(811)
    model = cls(**kwargs).float().train()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable
    if mode == "lora":
        assert all(n.startswith("transformer.") and ".lora_" in n for n in trainable)
        assert any("single_transformer_blocks" in n for n in trainable)
        assert any("transformer_blocks.0.attn.to_q." in n for n in trainable)
    else:
        assert all(p.requires_grad for p in model.transformer.parameters())
        assert all(n.startswith("transformer.") for n in trainable)
    assert not model.text.training
    assert not model.vae.training
    frozen = {n: t.clone() for n, t in model.state_dict().items() if n not in trainable}
    before = {n: p.detach().clone() for n, p in trainable.items()}
    optim = torch.optim.AdamW(trainable.values(), lr=0.003)
    inputs = batch_inputs(task)
    loss = model(**inputs)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in trainable.values())
    assert all(p.grad is None for n, p in model.named_parameters() if n not in trainable)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable.values())
    optim.step()
    assert any(not torch.equal(before[n], p) for n, p in trainable.items())
    for name, value in frozen.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    model.save_checkpoint(str(tmp_path))
    torch.save(optim.state_dict(), tmp_path / "optimizer.bin")
    saved = torch.load(tmp_path / "pytorch_model.bin", weights_only=True)
    if mode == "lora":
        assert saved and all(".lora_" in name for name in saved)
    else:
        assert "transformer.x_embedder.weight" in saved

    torch.manual_seed(811)  # The same immutable base is required for adapter checkpoints.
    restored = cls(**kwargs).float()
    restored.from_checkpoint(str(tmp_path))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    restored_optim = torch.optim.AdamW(
        [p for p in restored.parameters() if p.requires_grad], lr=0.003
    )
    restored_optim.load_state_dict(torch.load(tmp_path / "optimizer.bin", weights_only=True))
    assert all(int(s["step"]) == 1 for s in restored_optim.state.values())

    generate_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected = model.generate(**generate_inputs, height=16, width=16).images
        actual = restored.generate(**generate_inputs, height=16, width=16).images
    assert actual.shape == (2, 16, 16, 3) and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored.train()
    restored_optim.zero_grad()
    loss = restored(**inputs)  # Inference must not corrupt the training schedule.
    assert torch.isfinite(loss)
    loss.backward()
    restored_optim.step()
    assert all(int(s["step"]) == 2 for s in restored_optim.state.values())


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("task", TASKS)
def test_bfloat16_training_matrix(assets, variant, mode, task):
    """Exercise the example configs' storage dtype and fp32 adapter updates."""
    model = model_class(task, mode)(**model_kwargs(assets[variant]), torch_dtype="bfloat16")
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    expected_dtype = torch.float32 if mode == "lora" else torch.bfloat16
    assert all(p.dtype == expected_dtype for p in trainable.values())
    before = {name: p.detach().clone() for name, p in trainable.items()}
    loss = model(**batch_inputs(task))
    assert loss.ndim == 0 and loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    for block in ("transformer_blocks.0.", "single_transformer_blocks.0."):
        assert any(block in name and p.grad is not None and torch.count_nonzero(p.grad)
                   for name, p in trainable.items())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable.values())
    torch.optim.SGD(trainable.values(), lr=0.03).step()
    assert any(not torch.equal(before[name], p) for name, p in trainable.items())
    assert all(p.grad is None for p in model.text.parameters())
    assert all(p.grad is None for p in model.vae.parameters())


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("task", TASKS)
def test_generation_matches_current_official_pipeline(assets, variant, task):
    """Compare complete denoising/decoding against the concrete official pipelines."""
    torch.manual_seed(811)
    model = model_class(task, "e2e")(**model_kwargs(assets[variant]), torch_dtype="float32").eval()
    inputs = batch_inputs("text2image", batch_size=1)
    inputs.pop("pixel_values")
    images = None
    if task == "editing":
        # The official image API requires references to be at least 64 pixels,
        # even when the real test VAE has a smaller downsampling factor.
        images = [Image.open(assets[variant] / f"image{index}.png").convert("RGB").resize((64, 64))
                  for index in range(2)]
        inputs["refer_pixel_values"] = torch.stack([
            model.pipeline.image_processor.preprocess(image, height=64, width=64)[0]
            for image in images
        ]).unsqueeze(0)
    actual = model.generate(**inputs, height=16, width=16).images
    with torch.no_grad():
        prompt, _ = model._encode_prompt(inputs["input_ids"], inputs["attention_mask"])
        expected = model.pipeline(
            prompt_embeds=prompt, image=images, height=16, width=16,
            num_inference_steps=2, guidance_scale=model.guidance_scale,
            generator=torch.Generator().manual_seed(model.seed), output_type="np",
        ).images
    # Official PIL preprocessing is channels-last; collated references are
    # contiguous. CPU convolution reduction order differs even for identical
    # pixels. 1e-5 in [0, 1] is below 0.003 of one 8-bit output level.
    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("variant,layers", [("dev", (10, 20, 30)), ("klein-4b", (9, 18, 27))])
def test_prompt_and_pipeline_variant(assets, variant, layers):
    model = model_class("text2image", "e2e")(**model_kwargs(assets[variant])).float().eval()
    assert model.text_encoder_out_layers == layers
    pipeline_cls = Flux2Pipeline if variant == "dev" else Flux2KleinPipeline
    assert isinstance(model.pipeline, pipeline_cls)
    inputs = batch_inputs("text2image")
    with torch.no_grad():
        output = model.text(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                            output_hidden_states=True, use_cache=False)
        expected = torch.cat([output.hidden_states[i] for i in layers], dim=-1)
        actual, ids = model._encode_prompt(inputs["input_ids"], inputs["attention_mask"])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(ids, pipeline_cls._prepare_text_ids(expected))


@pytest.mark.parametrize("variant", VARIANTS)
def test_guidance_matches_transformer_config(assets, variant):
    model = model_class("text2image", "e2e")(**model_kwargs(assets[variant])).float()
    seen = []
    handle = model.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["guidance"]), with_kwargs=True
    )
    model(**batch_inputs("text2image"))
    handle.remove()
    if variant == "dev":
        torch.testing.assert_close(seen[0], torch.full((2,), 4.0))
    else:
        assert seen == [None]


@pytest.mark.parametrize("variant", VARIANTS)
def test_prompt_tokenization_matches_official_pipeline(assets, variant):
    from unitorch.models.diffusers.processing_flux2 import Flux2Processor

    processor = Flux2Processor(**processor_kwargs(assets[variant], variant))
    prompt = "edit red [IMG] square"
    tokenizer = processor.tokenizer
    if variant == "dev":
        from diffusers.pipelines.flux2.pipeline_flux2 import SYSTEM_MESSAGE

        expected = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_MESSAGE},
             {"role": "user", "content": prompt.replace("[IMG]", "")}],
            add_generation_prompt=False, tokenize=True, return_dict=True,
            return_tensors="pt", padding="max_length", truncation=True, max_length=128,
        )
    else:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        expected = tokenizer(rendered, return_tensors="pt", padding="max_length",
                             truncation=True, max_length=128)
    actual = processor.text2image_inputs(prompt)
    torch.testing.assert_close(actual.input_ids, expected["input_ids"][0])
    torch.testing.assert_close(actual.attention_mask, expected["attention_mask"][0])


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
def test_multiple_references_preserve_samples(assets, variant, mode):
    model = model_class("editing", mode)(**model_kwargs(assets[variant])).float().eval()
    refs = batch_inputs("editing", multiple=True)["refer_pixel_values"]
    actual, ids = model._prepare_condition_latents(refs, batch_size=2)
    for sample in range(2):
        with torch.no_grad():
            latents = [model.pipeline._encode_vae_image(refs[sample:sample+1, index], None)
                       for index in range(2)]
        expected = torch.cat([Flux2Pipeline._pack_latents(x) for x in latents], dim=1)
        torch.testing.assert_close(actual[sample:sample+1], expected)
        torch.testing.assert_close(ids[sample:sample+1], Flux2Pipeline._prepare_image_ids(latents))
    assert not torch.equal(actual[0], actual[1])
    assert set(ids[0, :, 0].tolist()) == {10, 20}
    model.train()
    loss = model(**batch_inputs("editing", multiple=True))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)
               for p in model.transformer.parameters() if p.requires_grad)


def test_reference_batch_mismatch_is_rejected(assets):
    model = model_class("editing", "e2e")(**model_kwargs(assets["dev"])).float()
    with pytest.raises(ValueError, match="batch"):
        model._prepare_condition_latents(torch.zeros(1, 3, 16, 16), batch_size=2)


@pytest.mark.parametrize("variant", VARIANTS)
def test_processing_and_real_collation(assets, variant):
    from unitorch.cli.models.diffusers.processing_flux2 import Flux2Processor
    from unitorch.cli.models import TensorTargets
    from unitorch.cli.tasks.supervised import collate_fn

    processor = Flux2Processor(**processor_kwargs(assets[variant], variant))
    images = [str(assets[variant] / f"image{i}.png") for i in range(2)]
    rows = [processor._editing("edit square", images if i == 0 else images[::-1], images[i])
            for i in range(2)]
    batch, _ = collate_fn([(row, TensorTargets()) for row in rows])
    assert batch.input_ids.shape == (2, 128)
    assert batch.pixel_values.shape == (2, 3, 16, 16)
    assert batch.refer_pixel_values.shape == (2, 2, 3, 16, 16)
    assert not torch.equal(batch.refer_pixel_values[0, 0], batch.refer_pixel_values[1, 0])
    assert -1 <= batch.pixel_values.min() <= batch.pixel_values.max() <= 1


def test_training_vae_mode_and_bn_match_upstream(assets, monkeypatch):
    model = model_class("text2image", "e2e")(**model_kwargs(assets["dev"])).float()
    inputs = batch_inputs("text2image")
    seen = []
    original = model._sample_timesteps_and_sigmas

    def record_latents(latents):
        seen.append(latents.detach().clone())
        return original(latents)

    monkeypatch.setattr(model, "_sample_timesteps_and_sigmas", record_latents)
    model(**inputs)
    with torch.no_grad():
        expected = model.pipeline._encode_vae_image(inputs["pixel_values"], None)
    torch.testing.assert_close(seen[0], expected)
    raw = model.vae.encode(inputs["pixel_values"]).latent_dist.mode()
    recovered = Flux2Pipeline._unpatchify_latents(model._denormalize_patched_latents(expected))
    torch.testing.assert_close(recovered, raw)


def test_train_scheduler_survives_generation(assets):
    model = model_class("text2image", "e2e")(**model_kwargs(assets["dev"])).float().eval()
    inputs = batch_inputs("text2image")
    torch.manual_seed(91)
    before = model._sample_timesteps_and_sigmas(inputs["pixel_values"])
    model.generate(input_ids=inputs["input_ids"], height=16, width=16)
    torch.manual_seed(91)
    after = model._sample_timesteps_and_sigmas(inputs["pixel_values"])
    for actual, expected in zip(after, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("component", ["text", "vae"])
def test_optional_encoder_unfreeze_is_effective(assets, component):
    kwargs = model_kwargs(assets["klein-4b"])
    kwargs["freeze_text_encoder" if component == "text" else "freeze_vae_encoder"] = False
    model = model_class("text2image", "e2e")(**kwargs).float().train()
    module = model.text if component == "text" else model.vae.encoder
    before = {n: p.detach().clone() for n, p in module.named_parameters()}
    loss = model(**batch_inputs("text2image"))
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters())
    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    optim.step()
    assert any(not torch.equal(before[n], p) for n, p in module.named_parameters())
    # Decoding and BN statistics are not part of the diffusion training objective.
    assert not any(p.requires_grad for p in model.vae.decoder.parameters())


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
def test_local_pretrained_snapshot_is_loaded_exactly(assets, variant, mode):
    kwargs = model_kwargs(assets[variant])
    kwargs.update(pretrained_model_path=str(assets[variant]), torch_dtype="float32")
    model = model_class("text2image", mode)(**kwargs)
    if variant == "klein-4b":
        assert model.text.lm_head.weight is model.text.model.embed_tokens.weight
    expected = torch.load(assets[variant] / "initial.bin", weights_only=True)
    for name, value in model.state_dict().items():
        if ".lora_" in name:
            continue
        original = name.replace(".base_layer.", ".")
        torch.testing.assert_close(value, expected[original], rtol=0, atol=0)


def test_custom_lora_targets_load_base_weights(assets):
    model = model_class("text2image", "lora")(
        **model_kwargs(assets["dev"]), enable_text_adapter=False,
        target_modules=["x_embedder", "to_qkv_mlp_proj"],
    ).float()
    expected = torch.load(assets["dev"] / "initial.bin", weights_only=True)
    model.from_pretrained(state_dict=expected)
    torch.testing.assert_close(model.transformer.x_embedder.base_layer.weight,
                               expected["transformer.x_embedder.weight"], rtol=0, atol=0)


def test_cli_pretrained_shards_are_consumed_one_at_a_time(assets, monkeypatch):
    from unitorch.cli.models.diffusers import modeling_flux2

    root = assets["dev"]
    infos = {
        "transformer": {"weight": str(root / "transformer/diffusion_pytorch_model.safetensors")},
        "text": {"weight": str(root / "text_encoder/model.safetensors")},
        "vae": {"weight": str(root / "vae/diffusion_pytorch_model.safetensors")},
    }
    calls = []
    original = modeling_flux2.load_weight

    def load_one(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(modeling_flux2, "load_weight", load_one)
    chunks = modeling_flux2._flux2_state_dict(infos, False)
    assert calls == [], "Do not hold all official transformer/text shards in RAM."
    first = next(chunks)
    assert len(calls) == 1 and all(k.startswith("transformer.") for k in first)
    del first
    rest = list(chunks)
    assert len(calls) == 3 and len(rest) == 2


@pytest.mark.parametrize("component", [None, "text", "vae"])
def test_e2e_checkpoint_reuses_frozen_base(assets, component, tmp_path):
    kwargs = model_kwargs(assets["klein-4b"])
    kwargs.update(pretrained_model_path=str(assets["klein-4b"]), torch_dtype="float32")
    if component is not None:
        kwargs["freeze_text_encoder" if component == "text" else "freeze_vae_encoder"] = False
    model = model_class("text2image", "e2e")(**kwargs)
    model(**batch_inputs("text2image")).backward()
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1).step()
    model.save_checkpoint(str(tmp_path))
    state = torch.load(tmp_path / "pytorch_model.bin", weights_only=True)
    assert not any(key.startswith("vae.decoder.") for key in state)
    assert any(key.startswith("text.") for key in state) == (component == "text")
    assert any(key.startswith("vae.encoder.") for key in state) == (component == "vae")
    restored = model_class("text2image", "e2e")(**kwargs)
    restored.from_checkpoint(str(tmp_path))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    # Old full-model checkpoints remain valid.
    torch.save(model.state_dict(), tmp_path / "legacy.bin")
    restored.from_checkpoint(str(tmp_path), weight_name="legacy.bin")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("was_training", [True, False])
@pytest.mark.parametrize("placement", ["host", "forced_cuda"])
def test_loss_validation_restores_training_mode(assets, was_training, placement, monkeypatch):
    from unitorch.cli.models.diffusers.modeling_flux2 import Flux2ForText2ImageGeneration
    from unitorch.cli.models.diffusion_utils import diffusion_model_decorator
    from unitorch.cli.models import TensorInputs, TensorTargets
    from unitorch.cli.tasks.supervised import SupervisedTask, infer

    model = diffusion_model_decorator(Flux2ForText2ImageGeneration)(
        **model_kwargs(assets["klein-4b"]), torch_dtype="float32"
    )
    inputs = TensorInputs(**batch_inputs("text2image"))
    model(**inputs.dict())  # Enter the diffusion loss/evaluation flow.
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    placements = []
    if placement == "forced_cuda":
        # Exercise CUDA dispatch on CPU CI, retaining the real tiny model/forward.
        # An input transfer must follow the caller's model placement.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def model_cuda(module, *args, **kwargs):
            placements.append(module)
            return module

        def tensor_cuda(tensor, *args, **kwargs):
            assert placements == [model], "caller must place model before CUDA inputs"
            return tensor

        monkeypatch.setattr(torch.nn.Module, "cuda", model_cuda)
        monkeypatch.setattr(torch.Tensor, "cuda", tensor_cuda)
    # The helper's normal train/eval callers own model placement.
    model = SupervisedTask(configure=None, model=model, datasets={}).model
    if placement == "host":
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert all(parameter.device.type == expected_device for parameter in model.parameters())
    model.train(was_training)
    result = infer(model, [(inputs, TensorTargets())])
    assert torch.isfinite(result.outputs.loss).all()
    assert model.training is was_training
    assert model.model.transformer.training is was_training
    assert not model.model.text.training and not model.model.vae.training


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("task", TASKS)
def test_sequential_offload_inference_matches_resident_model(assets, variant, mode, task):
    model = model_class(task, mode)(**model_kwargs(assets[variant]), torch_dtype="float32").eval()
    inputs = {key: value for key, value in batch_inputs(task).items() if key != "pixel_values"}
    expected = model.generate(**inputs, height=16, width=16).images
    actual = model.generate(**inputs, height=16, width=16,
                            enable_cpu_offload=True, offload_device="cpu").images
    assert any(parameter.device.type == "meta" for parameter in model.parameters())
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
def test_serialized_component_shards_match_snapshot(assets, variant, mode):
    """Mistral3 save_pretrained uses legacy keys, unlike its in-memory state."""
    from unitorch.cli.models.diffusers.modeling_flux2 import _flux2_state_dict

    root = assets[variant]
    model = model_class("text2image", mode)(**model_kwargs(root), torch_dtype="float32")
    components = {
        "transformer": {"weight": str(root / "transformer/diffusion_pytorch_model.safetensors")},
        "text": {"weight": str(root / "text_encoder/model.safetensors")},
        "vae": {"weight": str(root / "vae/diffusion_pytorch_model.safetensors")},
    }
    model.from_pretrained(state_dict=_flux2_state_dict(components, False))
    expected = torch.load(root / "initial.bin", weights_only=True)
    for name, value in model.state_dict().items():
        if ".lora_" not in name:
            torch.testing.assert_close(value, expected[name.replace(".base_layer.", ".")],
                                       rtol=0, atol=0, msg=name)


def test_bad_image_size_has_actionable_error(assets):
    from unitorch.models.diffusers.processing_flux2 import Flux2Processor

    kwargs = processor_kwargs(assets["dev"], "dev")
    kwargs["image_size"] = (15, 16)
    with pytest.raises(ValueError, match="divisible"):
        Flux2Processor(**kwargs)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("task", TASKS)
def test_cli_train_resume_infer_matrix(assets, variant, mode, task, tmp_path, monkeypatch):
    """Run installed console scripts, including AST editing-column processing."""
    config = write_cli_config(tmp_path, assets[variant], variant, mode, task)
    commands = []
    load = torch.load

    def load_storage(path, *args, **kwargs):
        # Storage comparisons must work even when training saved CUDA tensors.
        assert kwargs.get("map_location") == "cpu"
        return load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", load_storage)

    def run(command, log_name):
        commands.append(command)
        (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2))
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        with (tmp_path / log_name).open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, (tmp_path / log_name).read_text()[-8000:]

    train = [str(Path(sys.executable).parent / "unitorch-train"), str(config)]
    run(train, "train.log")
    checkpoint = tmp_path / "checkpoint"
    info = json.loads((checkpoint / "info.json").read_text())
    assert info["global_epoch"] == 1
    losses = re.findall(r"train/loss=([0-9.e+-]+)", (tmp_path / "train.log").read_text())
    assert losses and all(math.isfinite(float(loss)) and float(loss) > 0 for loss in losses)
    saved = torch.load(checkpoint / "pytorch_model_latest.bin", weights_only=True, map_location="cpu")
    initial = torch.load(assets[variant] / "initial.bin", weights_only=True, map_location="cpu")
    if mode == "lora":
        assert all(k.startswith("transformer.") and ".lora_" in k for k in saved)
        assert any("single_transformer_blocks" in k for k in saved)
        assert any(torch.count_nonzero(v) for k, v in saved.items() if ".lora_B." in k)
    else:
        assert saved["transformer.x_embedder.weight"].dtype == torch.float32
        assert any(not torch.equal(v, initial[k]) for k, v in saved.items() if k.startswith("transformer."))
        for key, value in saved.items():
            if key.startswith(("vae.", "text.")):
                torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    opt = torch.load(checkpoint / "pytorch_optim_latest.bin", weights_only=True, map_location="cpu")
    assert max(int(s["step"]) for s in opt["state"].values()) == 1
    run(train + ["--core/task/supervised@epochs=2"], "resume.log")
    info = json.loads((checkpoint / "info.json").read_text())
    assert info["global_epoch"] == 2
    opt = torch.load(checkpoint / "pytorch_optim_latest.bin", weights_only=True, map_location="cpu")
    assert max(int(s["step"]) for s in opt["state"].values()) == 2
    resumed = torch.load(checkpoint / "pytorch_model_latest.bin", weights_only=True, map_location="cpu")
    assert any(not torch.equal(saved[k], v) for k, v in resumed.items())
    infer = [str(Path(sys.executable).parent / "unitorch-infer"), str(config),
             f"--from_ckpt_dir={checkpoint}"]
    run(infer, "infer.log")
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["text"]
        image = Image.open(tmp_path / "images" / row["diffusion"])
        assert image.size == (16, 16)
        if task == "editing":
            assert row["refer_image"].endswith(".png")
    assert str(checkpoint / "pytorch_model.bin") in (tmp_path / "infer.log").read_text()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("entry", ["package", "cli"])
def test_zero3_initialization_and_weight_loading(assets, variant, mode, entry, tmp_path):
    pytest.importorskip("deepspeed")
    command = [sys.executable, str(Path(__file__).with_name("flux2_zero3_worker.py")),
               f"--assets={assets[variant]}", f"--output={tmp_path}",
               f"--variant={variant}", f"--mode={mode}", f"--entry={entry}"]
    (tmp_path / "command.json").write_text(json.dumps(command))
    env = dict(os.environ, DS_ACCELERATOR="cpu", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with (tmp_path / "worker.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, (tmp_path / "worker.log").read_text()[-7000:]
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "passed"


def test_deepspeed_cpu_inference_with_cuda_available(assets, tmp_path):
    pytest.importorskip("deepspeed")
    command = [sys.executable, str(Path(__file__).with_name("flux2_zero3_worker.py")),
               f"--assets={assets['klein-4b']}", f"--output={tmp_path}",
               "--entry=cli", "--workflow=inference", "--force_cuda_available=True"]
    env = dict(os.environ, DS_ACCELERATOR="cpu", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    (tmp_path / "command.json").write_text(json.dumps(command))
    with (tmp_path / "worker.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, (tmp_path / "worker.log").read_text()[-8000:]
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "passed"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
def test_zero3_two_rank_partition_and_update(assets, variant, mode, tmp_path):
    pytest.importorskip("deepspeed")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               str(Path(__file__).with_name("flux2_zero3_worker.py")),
               f"--assets={assets[variant]}", f"--output={tmp_path}",
               f"--variant={variant}", f"--mode={mode}", "--entry=cli"]
    (tmp_path / "command.json").write_text(json.dumps(command))
    env = dict(os.environ, DS_ACCELERATOR="cpu", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with (tmp_path / "worker.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, (tmp_path / "worker.log").read_text()[-9000:]
    outcome = json.loads((tmp_path / "result.json").read_text())
    assert outcome["status"] == "passed" and outcome["world_size"] == 2


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("cuda_visibility", ["host", "forced_cuda"])
def test_zero3_task_checkpoint_resume(assets, variant, mode, task, tmp_path, cuda_visibility):
    """Real task training, validation, native optimizer resume and filtered export."""
    pytest.importorskip("deepspeed")
    command = [sys.executable, str(Path(__file__).with_name("flux2_zero3_worker.py")),
               f"--assets={assets[variant]}", f"--output={tmp_path}",
               f"--variant={variant}", f"--mode={mode}", "--entry=cli",
               "--workflow=checkpoint", f"--task_name={task}"]
    if cuda_visibility == "forced_cuda":
        command.append("--force_cuda_available=True")
    env = dict(os.environ, DS_ACCELERATOR="cpu", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    states = []
    failures = []
    for epoch in (1, 2):
        invocation = command + [f"--epochs={epoch}"]
        (tmp_path / f"command-{epoch}.json").write_text(json.dumps(invocation))
        with (tmp_path / f"worker-{epoch}.log").open("w") as log:
            result = subprocess.run(invocation, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode != 0:
            failures.append((tmp_path / f"worker-{epoch}.log").read_text()[-8000:])
            continue
        assert json.loads((tmp_path / f"result-{epoch}.json").read_text())["engine_global_steps"] == epoch
        states.append(torch.load(tmp_path / "checkpoint/pytorch_model_latest.bin", weights_only=True))
    assert not failures, "\n".join(failures)
    assert any(not torch.equal(states[0][key], value) for key, value in states[1].items())
    restored = model_class(task, mode)(**model_kwargs(assets[variant]),
                                      pretrained_model_path=str(assets[variant]),
                                      torch_dtype="float32", **({"lora_r": 2, "lora_alpha": 2} if mode == "lora" else {}))
    restored.from_checkpoint(str(tmp_path / "checkpoint"), weight_name="pytorch_model_latest.bin")
    restored.eval()
    inputs = {key: value for key, value in batch_inputs(task).items() if key != "pixel_values"}
    images = restored.generate(**inputs, height=16, width=16).images
    assert images.shape == (2, 16, 16, 3) and torch.isfinite(images).all()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("task", TASKS)
def test_explicit_finetune_configs(variant, mode, task):
    from unitorch.cli import Config
    from unitorch.cli.models.diffusers import pretrained_stable_infos

    path = f"examples/configs/diffusion/{task}/flux2.{variant}.{mode}.ini"
    config = Config(path)
    task_section = config.getdefault("core/cli", "task_name", None)
    assert config.getdefault(task_section, "optim", None) == "core/optim/adamw", "train() requires an optim argument."
    section = (f"core/model/diffusers/peft/lora/{task}/flux2" if mode == "lora"
               else f"core/model/diffusers/{task}/flux2")
    name = config.getdefault(section, "pretrained_name", None)
    expected_repo = "black-forest-labs/FLUX.2-dev" if variant == "dev" else "black-forest-labs/FLUX.2-klein-4B"
    assert pretrained_stable_infos[name]["repo_id"] == expected_repo
    assert config.getdefault(section, "text_encoder_out_layers", None) == (
        (10, 20, 30) if variant == "dev" else (9, 18, 27)
    )
    assert config.getdefault(section, "guidance_scale", None) == (4.0 if variant == "dev" else 1.0)
    if mode == "lora":
        assert config.getdefault(section, "enable_text_adapter", None) is False
    else:
        assert config.getdefault(section, "freeze_text_encoder", None) is True
        assert config.getdefault(section, "freeze_vae_encoder", None) is True
