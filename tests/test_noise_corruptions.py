import numpy as np

from cvio_asl_ldam.data.corruptions import TOP5_CORRUPTIONS, apply_corruption


def test_top5_corruptions_preserve_shape():
    image = np.full((16, 20, 3), 127, dtype=np.uint8)
    for corruption in TOP5_CORRUPTIONS:
        output = apply_corruption(image, corruption, severity=2, seed=42)
        assert output.shape == image.shape
        assert np.isfinite(output).all()
