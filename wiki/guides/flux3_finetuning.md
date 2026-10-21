# FLUX.3 image klein base

UniTorch supports the exact `black-forest-labs/flux-3-image-klein-base`
checkpoint for text-to-image, reference-image editing, image-DiT LoRA and
full image-DiT finetuning. This is a separate architecture from FLUX.2.
FLUX.2 adapters and distilled FLUX.3 flow-map checkpoints are rejected.

## Install and obtain weights

Use an isolated environment. Install the local package before running its CLI:

```bash
python -m pip install -e '.[flux3]' 'diffusers @ https://github.com/huggingface/diffusers/archive/b800f828f3951c4d206e154411bcbfa693cd322e.tar.gz'
```

The command explicitly pins Diffusers to the tested revision
`b800f828f3951c4d206e154411bcbfa693cd322e` (`0.41.0.dev0`), which provides
`Flux3Transformer2DModel` and the FLUX.3 checkpoint converter. The `flux3` extra
declares `diffusers>=0.38.0`, Transformers 5.8.1, PEFT 0.18.1 and
`accelerate>=1.13.0`; it does not pin the Git revision. The Diffusers lower bound
keeps package metadata compatible with the tested revision, but does not
guarantee FLUX.3 APIs in every allowed release. A PyPI Diffusers release has not
been verified as a substitute; use the explicit Git pin above.
The existing FLUX.2 extra and APIs retain their defaults. Baseline installations
without the new optional backend do not import FLUX.3-specific symbols.
No global CUDA, driver or Python upgrade is required; the isolated validation
environment uses Python 3.12 and PyTorch 2.6.0+cu124.

Use an already authorized download of revision
`3f7fa5ee22ca1d5a5cc095ac7cc96e6a173b4c86`. Do not copy credentials between
machines. The native package is used directly:

```text
weights/flux-3-image-klein-base/
  config.json
  manifest.json
  model.safetensors
  ae.safetensors
  text_encoder/...
```

The loader reads the native config and verifies the DiT SHA256 before the
public Diffusers tensor conversion. It checks complete tensor coverage,
shapes and finite values before assigning parameters. Explicit Qwen3-VL and
FLUX.2 VAE classes load the remaining components. Converted Diffusers component
folders are also supported, including small real models used in CPU tests.
There is no invented `Flux3Pipeline` API. Upstream's pipeline entry point is
`ModularPipeline.from_pretrained(...)`, followed by `load_components(dtype=...)`.
The tensor-level UniTorch model uses the same concrete transformer and recipe.

## Scene captions and data

Conditioning should describe the complete desired scene: subjects, garments,
materials, colors, composition, lighting and placement. For paired-assets
finetuning, a fixed, complete full-scene task instruction together with the
source pixels is valid; per-image VLM captions are optional. The currently
authorized 100-epoch run uses this fixed-instruction mode. It does not use
generated source-specific captions; visual details come from source-pixel
conditioning. A short editing request alone does not describe the full task.

For example, a complete task instruction can say:

> Arrange every visible garment and accessory from the source image as complete,
> separate items on a clean white background. Preserve their colors, materials,
> patterns, lace, shape and relative proportions, including trousers, dresses,
> footwear, scarves and bags when present. Show the full items without cropping,
> with clear spacing, consistent lighting and natural detail. Remove the wearer
> and original background, and do not invent additional items.

Keep the run's reviewed instruction and dataset hashes fixed. The existing
`unitorch.datasets.flux3_assets.prepare_captions` API can persist this mode using
a callback that returns the fixed instruction and a `captioner_id` identifying
that instruction/version. Record it as a fixed task instruction, not as a VLM
description of each source. Neither mode may use target/GT pixels or
target-derived text to construct conditioning.

For optional per-image captions, the separate official 9B PUP or another VLM
can use the published system prompts. `Flux3SourceCaptioner` uses the package's
Qwen3-VL-4B model in a separate preprocessing process. It sees only the source
image and instruction. It never sees the target image or target-derived text.
It describes all visible apparel/accessories arranged as complete items on
white, preserving trousers, dresses, lace, footwear, scarves and bags when
visible. Captions still need visual review for omissions and hallucinations.
This alternative is identified in each record; it is not labeled official PUP.
It can hit `max_new_tokens` and raises rather than accepting an incomplete
caption. Such a failure does not produce a completed caption corpus.

For WearIt-format JSONL, preserve `pair_id`, `group_id`, `split`, `source_local`,
`result_local`, `source_asset_sha256`, `result_asset_sha256`, algorithm metadata
and the existing `prompt` instruction. To opt into per-image VLM captions, run
the separate stage:

```bash
python -m unitorch.cli.scripts.flux3 captions \
  --dataset_path=./wearit/dataset/train.jsonl \
  --output_root=./wearit-flux3/captions \
  --pretrained_model_path=./weights/flux-3-image-klein-base
```

Repeat for `val.jsonl` using the same output directory. The SQLite cache commits
every completed caption and keys it by the source SHA256, instruction and
captioner identity. Restarting reuses verified entries. Output JSONL carries
source-only provenance; output TSV is headerless `caption<TAB>source<TAB>target`.
`--max_records=8` makes an explicitly partial pilot. Completion/count/SHA records
distinguish a pilot from the full corpus. `--stop_file=./stop-captions` permits a
graceful stop after the current caption. Do not run the optional 9B PUP alongside
the core image model on a memory-constrained GPU; run it as a separate stage.

Inference TSV is headerless `caption<TAB>source` for editing or `caption` for
text-to-image. The processor produces only tensors. It stores a fixed maximum
token window for collation; the model trims to the actual window rounded up to
80 tokens, capped at 8192. Long captions raise instead of silently truncating.
Image dimensions are validated, never silently resized or downgraded. A batch
of multiple references must have equal reference counts and spatial dimensions;
up to 10 references are supported. Different resolutions can use separate calls.

## Train and infer

One-command editing inference with full scene captions:

```bash
unitorch-infer examples/configs/diffusion/editing/flux3.ini \
  --pretrained_model_path=./weights/flux-3-image-klein-base \
  --test_file=./data/editing-captions.tsv --cache_dir=./runs/flux3-base
```

One-command LoRA pilot through the ordinary UniTorch training CLI:

```bash
unitorch-train examples/configs/diffusion/editing/flux3.klein-base.lora.ini \
  --pretrained_model_path=./weights/flux-3-image-klein-base \
  --train_file=./data/train.tsv --dev_file=./data/val.tsv \
  --cache_dir=./runs/flux3-pilot
```

The examples run one epoch. Use a genuinely new output directory for a fresh
run. A present latest checkpoint resumes; incomplete metadata or incompatible
weights fail explicitly. For checkpoint inference, use the same LoRA config
and `--from_ckpt_dir=./runs/flux3-pilot` with `unitorch-infer`.
Text-to-image has equivalent configs under `diffusion/text2image/`.
`flux3.klein-base.e2e.ini` trains the whole DiT while leaving Qwen3-VL and VAE
frozen. Full DiT optimizer memory is much larger; official full-DiT GPU training
is not claimed by the LoRA validation.

Generation size overrides live in `[core/model/diffusers/flux3/generate]`.
Defaults are 1024×1536 in the examples, 50 Euler steps, CFG 3.5 and shift 6.93.
The negative caption has its own Qwen window. CFG keeps the same clean source
reference in both branches. No distilled guidance embedding is substituted.

The training objective is mean squared velocity error with
`x_t = (1-t) * target_latent + t * noise`, target `noise - target_latent`.
Timesteps use the fixed shifted rectified-flow schedule. Source VAE latents are
clean, independently normalized conditions, positioned at `(0, h/2, w/2, -1)`;
additional references use -2, -3, etc. Targets use the final coordinate 0.
Qwen3-VL stacks hidden layers 4/8/12/16/20/24/28/32 and preserves padding position
1. References enter the VAE stream; an optional caption VLM sees only source images.
The target never enters text conditioning. Caption dropout trains the empty
caption branch needed for CFG.

Frozen DiT/VL weights are BF16, trainable LoRA weights are FP32, and the VAE
defaults to the official FP32 recipe. `vae_dtype=bfloat16` is explicit and is
not the default. Gradient checkpointing is enabled. `offload_conditioners=True`
moves frozen Qwen/VAE to CPU between stages without offloading live adapters.
It rejects attempts to offload trainable conditioners. No ZeRO-3 integration
is claimed for this new backend; ordinary DDP is supported. The tested two-GPU
fully resident recipe explicitly sets `--offload_conditioners=False` on the
WearIt controller. Its generic CLI default remains `True`.

## Bounded WearIt controller

First verify the copied corpus, including every PNG and the original manifest:

```bash
python -m unitorch.cli.scripts.flux3 validate_data \
  --dataset_root=./wearit --output_root=./wearit-flux3/validation
```

The WearIt verifier expects 58,026 RGB PNGs at 1024×1536, the pinned original
input snapshot, and the unchanged 31,640/200/0 split. The 200 validation rows
represent 130 connected groups. It writes a new image SHA manifest, verifies
all paired file hashes, and rejects train/validation sharing. Below-1K originals
retain their provenance; resampling does not create native 1K detail.

Use the restartable controller for bounded smoke pilots and formal 100-epoch
training:

```bash
python -m unitorch.cli.scripts.flux3 train \
  --pretrained_model_path=./weights/flux-3-image-klein-base \
  --train_captions=./wearit-flux3/pilot/train.jsonl \
  --val_captions=./wearit-flux3/pilot/val.jsonl \
  --data_validation=./wearit-flux3/validation/DATA_VALIDATED.json \
  --output_root=./runs/flux3-assets-pilot --pilot_steps=20
```

For two GPUs with fully resident conditioners, use
`torchrun --nproc_per_node=2 -m unitorch.cli.scripts.flux3 train --offload_conditioners=False ...`
with the remaining arguments above. Batch size is one per GPU, accumulation
four, rank 16, alpha 32, and
the initial learning rate is 1e-5. The controller checks disk/RAM/GPU memory and
aborts on nonfinite loss or gradients. It keeps only latest and best-loss
snapshots with optimizer, cursor, RNG and exact input/base identity. Explicit
`--resume=True` restores those states. STOP and save decisions are synchronized
across ranks at each optimizer boundary; `--stop_file=./stop-training` stops all
ranks after a complete snapshot. Publishing `latest` is crash-recoverable under
the run lock: recovery validates the published checkpoint or the committed
`previous-latest` fallback after an interrupted rename. Incomplete staging
directories are never resume candidates. Best loss is only a numerical monitor;
it does not prove sharpness,
item preservation or useful editing quality.

Without `--pilot_steps`, the controller defaults to 100 epochs but refuses to
start without all 31,640/200 conditioning records, an official train/save/resume/export/infer
proof, and a review approval JSON tied to the exact caption/data/proof hashes.
Formal validation evaluates exactly all 200 inherited rows at each snapshot.
A smaller pilot is smoke evidence only and cannot replace formal validation;
both keep 1024×1536 source/target geometry without a resolution downgrade.
Keep `--epochs=100`, the full split and exact conditioning/data/proof hashes for
the formal run; a pilot has a separate run identity and output directory.
The parent reviewer creates that approval after examining same-seed base/LoRA
images and swapped-reference controls. This development workflow does not start
the formal 100-epoch run automatically.

## Checkpoints and validation limits

Native UniTorch checkpoints contain only trained tensors plus a JSON sidecar
with base fingerprint, architecture options and weight SHA256. Loading validates
the entire file before changing a live parameter. LoRA `export_adapter(path)`
writes PEFT `adapter_model.safetensors`, `adapter_config.json` and base-identity
metadata; `load_adapter(path)` rejects missing, extra, corrupt or wrong-base
adapters. Export directories are never overwritten. Controller snapshots also
include the optimizer and per-rank RNG state; adapter export alone is not a full
training resume checkpoint.

Small-model CPU tests use real Qwen3-VL, Flux3Transformer2DModel and
AutoencoderKLFlux2 modules. They cover train/save/resume/infer, gradient isolation,
reference and caption controls, strict loading, CLI composition and two-rank
Gloo updates. These tests are not official-weight GPU or quality evidence.
Machine-specific official results belong in the maintainer report, with explicit
node, resolution, sampling settings, hashes and incomplete stages.

## Upstream defaults and license

Primary references are the [BFL reference implementation](https://github.com/black-forest-labs/flux-3-image-klein/tree/bfc8889a4497f846416f253a3972a1941ac698b9),
the [pinned Diffusers implementation](https://github.com/huggingface/diffusers/tree/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/modular_pipelines/flux3),
and the [exact model revision](https://huggingface.co/black-forest-labs/flux-3-image-klein-base/tree/3f7fa5ee22ca1d5a5cc095ac7cc96e6a173b4c86).
The native config supplies the authoritative 50/3.5/6.93 recipe; the reference
repository also contains older 4.0/16.0 function defaults, which are not silently
substituted for the pinned package settings.

At the pinned BFL source revision, optional prompt/reference/output safety checks
are **off by default** (`generate(..., safety=False)` and opt-in `--safety`). The
pinned Diffusers modular pipeline does not contain that checker. This tensor
adapter does not disable an upstream checker or claim that outputs were safety
screened. Applications needing those checks should retain the BFL content-filter
stage around inference and comply with the model license's content-filter terms.

Weights use FLUX Kommunity License v1.0, not UniTorch's MIT code license. It permits
specified noncommercial uses and a qualified-user commercial-output exception;
that exception is not blanket permission to fine-tune models commercially.
Review the exact checkpoint's LICENSE.md and NOTICE.md for your intended use.
No access/license acceptance or credential transfer is performed by this code.
