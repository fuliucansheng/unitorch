# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.
"""Portable FLUX.3 preparation commands; model logic remains in the package."""

from pathlib import Path

import fire


class Flux3Commands:
    def train(self, pretrained_model_path: str, train_captions: str, val_captions: str,
              data_validation: str, output_root: str, epochs: int = 100,
              grad_acc_step: int = 4, learning_rate: float = 1e-5,
              lora_r: int = 16, lora_alpha: int = 32, save_every: int = 100,
              pilot_steps=None, official_proof=None, approval_file=None,
              resume: bool = False, seed: int = 1123, offload_conditioners: bool = True,
              stop_file=None):
        from unitorch.tasks.flux3 import train_assets

        return train_assets(pretrained_model_path, train_captions, val_captions, data_validation,
                            output_root, epochs, grad_acc_step, learning_rate, lora_r, lora_alpha,
                            save_every, pilot_steps, official_proof, approval_file, resume, seed,
                            offload_conditioners, stop_file)

    def validate_data(self, dataset_root: str, output_root: str, workers: int = 8):
        from unitorch.datasets.flux3_assets import validate_assets

        return validate_assets(dataset_root, output_root, workers=workers)

    def captions(self, dataset_path: str, output_root: str, pretrained_model_path: str,
                 device: str = "cuda", max_records=None, max_new_tokens: int = 768,
                 system_prompt_path=None, stop_file=None):
        from unitorch.datasets.flux3_assets import prepare_captions
        from unitorch.models.diffusers.caption_flux3 import Flux3SourceCaptioner

        captioner = Flux3SourceCaptioner(str(Path(pretrained_model_path) / "text_encoder"),
                                         device=device, max_new_tokens=max_new_tokens,
                                         system_prompt_path=system_prompt_path)
        return prepare_captions(dataset_path, output_root, captioner, captioner.identity,
                                max_records=max_records, stop_file=stop_file)


if __name__ == "__main__":
    fire.Fire(Flux3Commands)
