import unittest

import numpy as np
from parameterized import parameterized

from .image import _render_image_data, get_supported_extensions


class TestGetSupportedExtensions(unittest.TestCase):
    def test_supported_ext(self):
        supported = get_supported_extensions()
        self.assertTrue(".tif" in supported)
        self.assertTrue(".tiff" in supported)
        self.assertTrue(len(supported) > 2)


class TestRenderImageData(unittest.TestCase):
    @parameterized.expand(
        [("rgb", 480), ("bgr", 512), ("sentinel", 528), ("bw", 492.96)]
    )
    def test_rendering(self, render_type, answer):
        test_array = np.arange(16 * 16 * 4).reshape((16, 16, 4))
        rendered = _render_image_data(test_array, render_type)
        result = rendered[0, :, 0].sum()

        self.assertAlmostEqual(result, answer)


if __name__ == "__main__":
    unittest.main()
