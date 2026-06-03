# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import imageio
import numpy as np
import pytest
from PIL import Image


def test_image_processor_prefers_local_path_when_http_url_is_configured(tmp_path, monkeypatch):
    from unitorch.cli.models.image_utils import ImageProcessor

    image_path = tmp_path / "local.png"
    Image.new("RGB", (7, 5), (12, 34, 56)).save(image_path)

    monkeypatch.setattr(
        ImageProcessor,
        "_request_url",
        lambda self, url: pytest.fail(f"unexpected http fetch for {url}"),
    )

    processor = ImageProcessor(http_url="http://0.0.0.0:11230/core/fastapi/servers/zip_files/?file={0}")
    image = processor._read(str(image_path))

    assert image.size == (7, 5)
    assert image.getpixel((0, 0)) == (12, 34, 56)


def test_video_processor_prefers_local_path_when_http_url_is_configured(tmp_path, monkeypatch):
    pytest.importorskip("cv2")

    from unitorch.cli.models.video_utils import VideoProcessor

    video_path = tmp_path / "local.mp4"
    writer = imageio.get_writer(video_path, fps=4)
    for value in (0, 64, 128):
        writer.append_data(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.close()

    monkeypatch.setattr(
        VideoProcessor,
        "_request_url",
        lambda self, url: pytest.fail(f"unexpected http fetch for {url}"),
    )

    processor = VideoProcessor(http_url="http://0.0.0.0:11230/core/fastapi/servers/zip_files/?file={0}")
    capture = processor._read(str(video_path))

    try:
        assert capture.isOpened()
        ret, frame = capture.read()
        assert ret
        assert frame.shape[:2] == (16, 16)
    finally:
        capture.release()
