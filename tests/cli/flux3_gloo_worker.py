"""Two-rank CPU training against actual tiny FLUX.3 components."""
import os
from pathlib import Path

import fire
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from unitorch.models.peft.diffusers.modeling_flux3 import Flux3LoraForImageEditingGeneration


def main(assets, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(1123 + rank)
    model = Flux3LoraForImageEditingGeneration(
        pretrained_model_path=assets, torch_dtype="float32", lora_r=2,
        text_encoder_out_layers=(1, 2, 3, 4), caption_dropout=0,
    )
    ddp = DistributedDataParallel(model)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    for _ in range(2):
        optim.zero_grad()
        loss = ddp(input_ids=torch.tensor([[1, 4, 5, 2]]), pixel_values=torch.randn(1, 3, 16, 16),
                   refer_pixel_values=torch.randn(1, 3, 16, 16))
        loss.backward()
        optim.step()
    for p in model.parameters():
        if p.requires_grad:
            gathered = [torch.zeros_like(p) for _ in range(2)]
            dist.all_gather(gathered, p)
            torch.testing.assert_close(gathered[0], gathered[1], atol=0, rtol=0)
    if rank == 0:
        model.save_checkpoint(output)
        Path(output, "GLOO_PASSED").write_text("2 ranks, 2 actual optimizer updates\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
