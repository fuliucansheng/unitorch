# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from pathlib import Path

from PIL import Image


def test_ast_function_uses_row_values_instead_of_column_names(tmp_path):
    import unitorch.cli.datasets.hf as hf_datasets
    import unitorch.cli.models  # noqa: F401
    from unitorch.cli import Config, init_registered_process

    image_path = Path(tmp_path) / "sample.png"
    Image.new("RGB", (9, 6), (10, 20, 30)).save(image_path)

    setattr(
        hf_datasets,
        "core_process_image_read",
        init_registered_process("core/process/image/read", Config(params=[])),
    )

    ast_fn = hf_datasets.ASTFunction("core/process/image/read(image)")
    image = ast_fn.process({"image": str(image_path)})

    assert image.size == (9, 6)
    assert image.getpixel((0, 0)) == (10, 20, 30)
