"""Real small FLUX.3 components; these are not official-weight GPU tests."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

pytest.importorskip("diffusers", minversion="0.41.0.dev0")
pytest.importorskip("peft")

from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel
from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from unitorch.models.diffusers.modeling_flux3 import (
    Flux3ForImageEditingGeneration, Flux3ForText2ImageGeneration,
    image_ids, text_ids, flow_schedule,
)
from unitorch.models.peft.diffusers.modeling_flux3 import (
    Flux3LoraForImageEditingGeneration, Flux3LoraForText2ImageGeneration,
)


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    torch.set_num_threads(2)
    torch.manual_seed(190)
    root = tmp_path_factory.mktemp("flux3-components")
    transformer = Flux3Transformer2DModel(
        in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=128,
        timestep_channels=16, mlp_ratio=2, axes_dims_rope=(4, 4, 4, 4),
    )
    transformer.save_pretrained(root / "transformer")
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=384, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8,
                         rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=48,
                           num_heads=4, patch_size=4, temporal_patch_size=2,
                           spatial_merge_size=2, out_hidden_size=32,
                           deepstack_visual_indexes=[]),
        image_token_id=125, video_token_id=126, vision_start_token_id=124,
    )
    Qwen3VLForConditionalGeneration(config).save_pretrained(root / "text_encoder")
    from tokenizers.pre_tokenizers import ByteLevel
    vocab = {c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}
    vocab.update({"<|endoftext|>": 256, "<|im_start|>": 257, "<|im_end|>": 258})
    (root / "text_encoder/vocab.json").write_text(json.dumps(vocab))
    (root / "text_encoder/merges.txt").write_text("#version: 0.2\n")
    tokenizer = Qwen2Tokenizer(vocab=str(root / "text_encoder/vocab.json"), merges=str(root / "text_encoder/merges.txt"),
                              pad_token="<|endoftext|>", eos_token="<|im_end|>")
    tokenizer.chat_template = "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    tokenizer.save_pretrained(root / "text_encoder")
    vae = AutoencoderKLFlux2(
        block_out_channels=(8, 16), latent_channels=2, layers_per_block=1,
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        norm_num_groups=4, sample_size=16, mid_block_add_attention=False,
    )
    vae.bn.running_mean.copy_(torch.linspace(-0.2, 0.3, 8))
    vae.bn.running_var.copy_(torch.linspace(0.5, 1.5, 8))
    vae.save_pretrained(root / "vae")
    return root


def kwargs(assets):
    return dict(pretrained_model_path=str(assets), torch_dtype="float32",
                text_encoder_out_layers=(1, 2, 3, 4), num_infer_timesteps=2)


def inputs(editing=True):
    g = torch.Generator().manual_seed(981)
    result = dict(input_ids=torch.tensor([[1, 4, 5, 2, 0, 0, 0, 0]]),
                  attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]]),
                  negative_input_ids=torch.tensor([[1, 2, 0, 0, 0, 0, 0, 0]]),
                  negative_attention_mask=torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]]),
                  pixel_values=torch.randn(1, 3, 16, 16, generator=g))
    if editing:
        result["refer_pixel_values"] = torch.randn(1, 3, 16, 16, generator=g)
    return result


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_actual_train_resume_infer(assets, tmp_path, editing, lora):
    cls = {(False, False): Flux3ForText2ImageGeneration,
           (True, False): Flux3ForImageEditingGeneration,
           (False, True): Flux3LoraForText2ImageGeneration,
           (True, True): Flux3LoraForImageEditingGeneration}[editing, lora]
    options = kwargs(assets)
    if lora:
        options.update(lora_r=2, lora_alpha=4, lora_dropout=0.0)
    model = cls(**options).train()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("transformer.") for n in trainable)
    if lora:
        assert all(".lora_" in n and p.dtype == torch.float32 for n, p in trainable.items())
        assert any("image_cond" in n for n in trainable)
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    assert not model.text.training and not model.vae.training
    optimizer = torch.optim.AdamW(trainable.values(), lr=0.003)
    batch = inputs(editing)
    loss = model(**batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in trainable.values())
    assert all(p.grad is None for n, p in model.named_parameters() if n in frozen)
    optimizer.step()
    for n, p in model.named_parameters():
        if n in frozen:
            torch.testing.assert_close(p, frozen[n], atol=0, rtol=0)
    model.save_checkpoint(str(tmp_path))
    torch.save(optimizer.state_dict(), tmp_path / "optim.bin")
    restored = cls(**options)
    restored.from_checkpoint(str(tmp_path))
    restored_optim = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad])
    restored_optim.load_state_dict(torch.load(tmp_path / "optim.bin", weights_only=True))
    assert all(int(s["step"]) == 1 for s in restored_optim.state.values())
    for n, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[n], value, atol=0, rtol=0)
    generate = {k: v for k, v in batch.items() if k != "pixel_values"}
    a = model.eval().generate(**generate, height=16, width=16).images
    b = restored.eval().generate(**generate, height=16, width=16).images
    assert a.shape == (1, 16, 16, 3) and torch.isfinite(a).all()
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    restored.train()
    restored_optim.zero_grad()
    restored(**batch).backward()
    restored_optim.step()
    if lora:
        model.export_adapter(str(tmp_path / "adapter"))
        exported = cls(**options)
        exported.load_adapter(str(tmp_path / "adapter"))
        c = exported.eval().generate(**generate, height=16, width=16).images
        torch.testing.assert_close(a, c, atol=0, rtol=0)


def test_reference_and_caption_negative_controls(assets):
    model = Flux3LoraForImageEditingGeneration(**kwargs(assets), lora_r=2).eval()
    batch = inputs()
    del batch["pixel_values"]
    a = model.generate(**batch, height=16, width=16).images
    batch["refer_pixel_values"] = -batch["refer_pixel_values"]
    b = model.generate(**batch, height=16, width=16).images
    assert not torch.allclose(a, b, atol=1e-6, rtol=0)
    batch["input_ids"][0, 1] = 9
    c = model.generate(**batch, height=16, width=16).images
    assert not torch.allclose(b, c, atol=1e-6, rtol=0)


def test_flow_and_position_contract():
    ids = image_ids(2, 3, "cpu", reference_index=1)
    torch.testing.assert_close(ids[:, 3], torch.full((6,), -2.0))
    torch.testing.assert_close(ids[:, 1:3], torch.tensor([[0, 0], [0, .5], [0, 1], [.5, 0], [.5, .5], [.5, 1]]))
    assert text_ids(80, "cpu")[-1].tolist() == [0, 0, 0, 79]
    schedule = flow_schedule(2, 6.93)
    torch.testing.assert_close(schedule, torch.tensor([1, 6.93 / 7.93, 0]))


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "nan"])
def test_checkpoint_rejects_before_mutation(assets, tmp_path, damage):
    model = Flux3LoraForImageEditingGeneration(**kwargs(assets), lora_r=2)
    model.save_checkpoint(str(tmp_path))
    path = tmp_path / "pytorch_model.bin"
    state = torch.load(path, weights_only=True)
    key = next(iter(state))
    if damage == "missing":
        state.pop(key)
    elif damage == "extra":
        state["transformer.unexpected"] = torch.ones(1)
    elif damage == "shape":
        state[key] = torch.ones(1)
    else:
        state[key].fill_(float("nan"))
    torch.save(state, path)
    before = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
    with pytest.raises((ValueError, RuntimeError), match="checkpoint|tensor|coverage|finite|shape"):
        model.from_checkpoint(str(tmp_path))
    for n, p in model.named_parameters():
        if n in before:
            torch.testing.assert_close(p, before[n], atol=0, rtol=0)


def test_cli_train_resume_and_infer(assets, tmp_path):
    from PIL import Image
    from unitorch.cli import cached_path

    source, target = tmp_path / "source.png", tmp_path / "target.png"
    Image.new("RGB", (16, 16), "red").save(source)
    Image.new("RGB", (16, 16), "blue").save(target)
    (tmp_path / "train.tsv").write_text(f"A blue garment on white.\t{source}\t{target}\n")
    (tmp_path / "infer.tsv").write_text(f"A blue garment on white.\t{source}\n")
    template = Path(cached_path("cli/configs/diffusion/editing/flux3.klein-base.lora.ini")).read_text()
    config = tmp_path / "config.ini"
    config.write_text(template)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
    common = [str(config), f"--pretrained_model_path={assets}", f"--cache_dir={tmp_path / 'run'}",
              f"--train_file={tmp_path / 'train.tsv'}", f"--dev_file={tmp_path / 'train.tsv'}",
              f"--test_file={tmp_path / 'infer.tsv'}",
              "--core/model/diffusers/editing/peft/lora/flux3@text_encoder_out_layers=(1,2,3,4)",
              "--core/model/diffusers/editing/peft/lora/flux3@torch_dtype=float32",
              "--core/model/diffusers/editing/peft/lora/flux3@lora_r=2",
              "--core/model/diffusers/editing/peft/lora/flux3@num_infer_timesteps=2",
              "--core/process/diffusion/flux3@image_size=(16,16)",
              "--core/process/diffusion/flux3@image_size_multiple=4",
              "--core/process/diffusion/flux3@max_seq_length=160",
              "--core/model/diffusers/flux3/generate@height=16", "--core/model/diffusers/flux3/generate@width=16",
              "--core/task/supervised@grad_acc_step=1"]
    for epochs in (1, 2):
        result = subprocess.run([str(Path(sys.executable).with_name("unitorch-train")), *common,
                                 f"--core/task/supervised@epochs={epochs}"], env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(Path(sys.executable).with_name("unitorch-infer")), *common,
                             f"--from_ckpt_dir={tmp_path / 'run'}"], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in (tmp_path / "run/predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["diffusion"]


def test_two_rank_gloo_training(assets, tmp_path):
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(Path(__file__).with_name("flux3_gloo_worker.py")), f"--assets={assets}", f"--output={tmp_path}",
    ], env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "GLOO_PASSED").exists()


def test_generic_partial_loading_is_refused(assets):
    model = Flux3LoraForImageEditingGeneration(**kwargs(assets), lora_r=2)
    with pytest.raises(ValueError, match="native|metadata|checkpoint"):
        model.from_pretrained(state_dict={"transformer.fake_lora.weight": torch.ones(1)})
