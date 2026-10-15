# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

"""FLUX.2 checkpoint export: python -m unitorch.cli.scripts.flux2."""

import fire

from unitorch.models.diffusers.checkpoint_flux2 import export_zero_checkpoint


if __name__ == "__main__":
    fire.Fire(export_zero_checkpoint)
