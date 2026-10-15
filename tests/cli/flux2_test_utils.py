"""Small, real FLUX.2 components; no substituted model implementations."""

import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers import Flux2Transformer2DModel
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    LlamaTokenizerFast,
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)


VARIANTS = ("dev", "klein-4b")
MODES = ("lora", "e2e")
TASKS = ("text2image", "editing")


def make_assets(root, variant):
    """Keep the official text layer counts/branches, shrinking tensor widths."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(481)
    transformer = Flux2Transformer2DModel(
        in_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=48,
        timestep_guidance_channels=16,
        mlp_ratio=2.0,
        axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=variant == "dev",
    )
    transformer.save_pretrained(root / "transformer")
    text_kwargs = dict(
        vocab_size=384,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=40 if variant == "dev" else 36,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
    )
    if variant == "dev":
        config = Mistral3Config(
            text_config={"model_type": "mistral", **text_kwargs},
            vision_config={
                "model_type": "pixtral",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "head_dim": 8,
                "image_size": 16,
                "patch_size": 4,
            },
            image_token_index=383,
            spatial_merge_size=2,
        )
        text = Mistral3ForConditionalGeneration(config)
    else:
        # The exact official klein text checkpoint omits lm_head.weight because
        # it is tied to embed_tokens; the shard loader must preserve that alias.
        text = Qwen3ForCausalLM(Qwen3Config(**text_kwargs, tie_word_embeddings=True))
    text.save_pretrained(root / "text_encoder")
    vae = AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16),
        layers_per_block=1,
        latent_channels=2,
        norm_num_groups=4,
        sample_size=16,
        mid_block_add_attention=False,
    )
    # Nontrivial statistics catch missing/wrong normalization and patch order.
    vae.bn.running_mean.copy_(torch.linspace(-0.3, 0.4, 8))
    vae.bn.running_var.copy_(torch.linspace(0.5, 1.5, 8))
    vae.save_pretrained(root / "vae")
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True
    )
    scheduler.save_pretrained(root / "scheduler")
    tokenizer_dir = root / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    if variant == "dev":
        vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
                 "red": 4, "blue": 5, "square": 6, "edit": 7}
        backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = LlamaTokenizerFast(
            tokenizer_object=backend, pad_token="<pad>", bos_token="<s>",
            eos_token="</s>", unk_token="<unk>",
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] + ' ' + "
            "message['content'] + ' ' }}{% endfor %}"
            "{% if add_generation_prompt %}assistant{% endif %}"
        )
    else:
        # A complete byte alphabet makes this a real, self-contained Qwen tokenizer.
        from tokenizers.pre_tokenizers import ByteLevel

        vocab = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2,
                 "<think>": 3, "</think>": 4}
        vocab.update({v: i + 5 for i, v in enumerate(sorted(ByteLevel.alphabet()))})
        (tokenizer_dir / "vocab.json").write_text(json.dumps(vocab))
        (tokenizer_dir / "merges.txt").write_text("#version: 0.2\n")
        tokenizer = Qwen2Tokenizer(
            vocab=str(tokenizer_dir / "vocab.json"),
            merges=str(tokenizer_dir / "merges.txt"),
            pad_token="<|endoftext|>", eos_token="<|im_end|>",
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ '<|im_start|>' + message['role'] + "
            "'\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
            "{% if enable_thinking is defined and enable_thinking is false %}"
            "{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}{% endif %}"
        )
    tokenizer.save_pretrained(tokenizer_dir)
    for index, color in enumerate(((220, 20, 30), (20, 30, 220))):
        Image.new("RGB", (16, 16), color).save(root / f"image{index}.png")
    state = {}
    for prefix, module in (("transformer", transformer), ("text", text), ("vae", vae)):
        state.update({f"{prefix}.{k}": v for k, v in module.state_dict().items()})
    torch.save(state, root / "initial.bin")
    return root


def model_kwargs(root):
    return dict(
        config_path=str(root / "transformer/config.json"),
        text_config_path=str(root / "text_encoder/config.json"),
        vae_config_path=str(root / "vae/config.json"),
        scheduler_config_path=str(root / "scheduler/scheduler_config.json"),
        num_infer_timesteps=2,
        gradient_checkpointing=True,
    )


def processor_kwargs(root, variant):
    args = dict(
        tokenizer_config=str(root / "tokenizer/tokenizer_config.json"),
        chat_template=str(root / "tokenizer/chat_template.jinja"),
        vae_config_path=str(root / "vae/config.json"),
        max_seq_length=128,
        image_size=(16, 16),
    )
    if variant == "dev":
        args.update(tokenizer_class="LlamaTokenizerFast",
                    tokenizer_path=str(root / "tokenizer/tokenizer.json"))
    else:
        args.update(tokenizer_class="Qwen2Tokenizer",
                    vocab_path=str(root / "tokenizer/vocab.json"),
                    merge_path=str(root / "tokenizer/merges.txt"))
    return args


def model_class(task, mode):
    from unitorch.models.diffusers.modeling_flux2 import (
        Flux2ForText2ImageGeneration, Flux2ForImageEditingGeneration,
    )
    from unitorch.models.peft.diffusers.modeling_flux2 import (
        Flux2LoraForText2ImageGeneration, Flux2LoraForImageEditingGeneration,
    )
    return {
        ("text2image", "e2e"): Flux2ForText2ImageGeneration,
        ("editing", "e2e"): Flux2ForImageEditingGeneration,
        ("text2image", "lora"): Flux2LoraForText2ImageGeneration,
        ("editing", "lora"): Flux2LoraForImageEditingGeneration,
    }[(task, mode)]


def batch_inputs(task, batch_size=2, multiple=False):
    generator = torch.Generator().manual_seed(922)
    batch = dict(
        input_ids=torch.randint(5, 30, (batch_size, 8), generator=generator),
        attention_mask=torch.ones(batch_size, 8, dtype=torch.long),
        pixel_values=torch.randn(batch_size, 3, 16, 16, generator=generator),
    )
    if task == "editing":
        shape = (batch_size, 2, 3, 16, 16) if multiple else (batch_size, 3, 16, 16)
        batch["refer_pixel_values"] = torch.randn(shape, generator=generator)
    return batch


def write_cli_config(root, assets, variant, mode, task):
    """An actual AST TSV dataset, train/dev/test splits and checkpoint IO."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    section = (f"core/model/diffusers/peft/lora/{task}/flux2" if mode == "lora"
               else f"core/model/diffusers/{task}/flux2")
    names = ["text", "refer_image", "image"] if task == "editing" else ["text", "image"]
    train = []
    test = []
    for index in range(2):
        image = str(assets / f"image{index}.png")
        ref = str(assets / f"image{1-index}.png")
        train.append(f"edit square\t{ref}\t{image}" if task == "editing"
                     else f"red square\t{image}")
        test.append(f"edit square\t{ref}" if task == "editing" else "red square")
    (root / "train.tsv").write_text("\n".join(train) + "\n")
    (root / "test.tsv").write_text("\n".join(test) + "\n")
    args = "text, refer_image, image" if task == "editing" else "text, image"
    infer_args = "text, refer_image" if task == "editing" else "text"
    processor = processor_kwargs(assets, variant)
    processor_options = "\n".join(f"{key} = {value}" for key, value in processor.items())
    model_options = "\n".join(f"{key} = {value}" for key, value in model_kwargs(assets).items())
    text = f"""[core/cli]
task_name = core/task/supervised
cache_dir = {root}/checkpoint
from_ckpt_dir = {root}/no-initial-checkpoint

[{section}]
pretrained_name = tiny-flux2
pretrained_model_path = {assets}
torch_dtype = float32
{model_options}
height = 16
width = 16
lora_r = 2
lora_alpha = 2
lora_dropout = 0.0

[core/process/diffusion/flux2]
pretrained_name = tiny-flux2
{processor_options}

[core/dataset/ast]
names = {names!r}

[core/dataset/ast/train]
data_files = {root}/train.tsv
preprocess_functions = ['core/process/diffusion/flux2/{task}({args})']

[core/dataset/ast/dev]
data_files = {root}/train.tsv
preprocess_functions = ['core/process/diffusion/flux2/{task}({args})']

[core/dataset/ast/test]
names = {names[:-1]!r}
data_files = {root}/test.tsv
preprocess_functions = ['core/process/diffusion/flux2/{task}/inputs({infer_args})']

[core/optim/adamw]
learning_rate = 0.003
weight_decay = 0.0

[core/process/diffusion]
output_folder = {root}/images

[core/task/supervised]
model = {section}
dataset = core/dataset/ast
optim = core/optim/adamw
loss_fn = None
score_fn = core/score/loss
monitor_fns = ['core/score/loss']
from_ckpt_dir = ${{core/cli:from_ckpt_dir}}
to_ckpt_dir = ${{core/cli:cache_dir}}
epochs = 1
train_batch_size = 2
dev_batch_size = 2
test_batch_size = 2
num_workers = 0
pin_memory = False
use_amp = False
log_freq = 1
ckpt_freq = 1
save_optimizer = True
save_checkpoint = default
postprocess_fn = core/postprocess/diffusion/image
writer = core/writer/jsonl
output_path = {root}/predictions.jsonl
output_header = {names[:-1]!r}
postprocess_workers = 1
"""
    path = root / "train.ini"
    path.write_text(text)
    return path
