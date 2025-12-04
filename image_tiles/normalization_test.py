import unittest

import numpy as np

from .normalization import (
    invert_mean_std_normalization,
    scaling_normalization,
    sentinel_truecolor_image,
    sigmoid_normalization,
    standard_normalization,
)


class TestScalingNormalization(unittest.TestCase):
    def test_scaling_image(self):
        test_array = np.arange(16 * 16 * 3).reshape((16, 16, 3))

        normed = scaling_normalization(test_array)
        self.assertAlmostEqual(normed.sum(), 97537)
        self.assertTrue(normed.max(), 255)
        self.assertEqual(normed.dtype, np.uint8)


class TestStandardNormalization(unittest.TestCase):
    def test_good_image(self):
        test_array = np.ones(16 * 16 * 3).reshape((16, 16, 3))
        test_array = test_array.astype(float)

        normed = standard_normalization(test_array)
        self.assertAlmostEqual(normed.sum(), 16 * 16 * 3 * 255)
        self.assertTrue(normed.max(), 255)
        self.assertEqual(normed.dtype, np.uint8)

    def test_bad_image(self):
        test_array = np.arange(16 * 16 * 3).reshape((16, 16, 3))

        normed = standard_normalization(test_array)
        self.assertAlmostEqual(normed.sum(), 97537)
        self.assertTrue(normed.max(), 255)
        self.assertEqual(normed.dtype, np.uint8)


class TestSentinelTrueColor(unittest.TestCase):
    def test_sentinel_norm(self):
        test_array = np.arange(16 * 16 * 3).reshape((16, 16, 3))
        normed = sentinel_truecolor_image(test_array)
        self.assertAlmostEqual(normed.sum(), 37170)
        self.assertTrue(normed.max(), 255)
        self.assertEqual(normed.dtype, np.uint8)


class TestSigmoidNormalization(unittest.TestCase):
    def test_sig_norm(self):
        test_array = np.arange(16 * 16 * 3).reshape((16, 16, 3))
        normed = sigmoid_normalization(test_array, 10, 0.125)
        self.assertAlmostEqual(normed.sum(), 165926)
        self.assertTrue(normed.max(), 255)
        self.assertEqual(normed.dtype, np.uint8)

        normed = sigmoid_normalization(test_array, 5, 0.25)
        self.assertAlmostEqual(normed.sum(), 137511)
        self.assertTrue(normed.max(), 255)
        self.assertEqual(normed.dtype, np.uint8)

    def test_sig_norm_batch(self):
        batch_test_array = np.arange(8 * 16 * 16 * 3).reshape((8, 16, 16, 3))
        normed = sigmoid_normalization(batch_test_array, 10, 0.125)
        self.assertAlmostEqual(normed.sum(), 1327408)
        self.assertTrue(normed.max(), 255)
        self.assertEqual(normed.dtype, np.uint8)

    def test_sig_norm_channels_first(self):
        batch_test_array = np.arange(16 * 16 * 3).reshape((3, 16, 16))
        normed = sigmoid_normalization(batch_test_array, 10, 0.125, channels="first")
        self.assertAlmostEqual(normed.sum(), 165926)
        self.assertTrue(normed.max(), 255)
        self.assertEqual(normed.dtype, np.uint8)

    def test_sig_norm_channels_first_batch(self):
        batch_test_array = np.arange(8 * 16 * 16 * 3).reshape((8, 3, 16, 16))
        normed = sigmoid_normalization(batch_test_array, 10, 0.125, channels="first")
        self.assertAlmostEqual(normed.sum(), 1327408)
        self.assertTrue(normed.max(), 255)
        self.assertEqual(normed.dtype, np.uint8)


class TestMeanStdNormalization(unittest.TestCase):
    def test_mean_std_norm(self):
        test_array = np.arange(16 * 16 * 3).reshape((16, 16, 3))
        mean = np.array((0.485, 0.456, 0.406))
        std = np.array((0.229, 0.224, 0.225))
        normed = invert_mean_std_normalization(test_array, mean=mean, std=std)
        self.assertAlmostEqual(normed.sum(), 66907.136)

    def test_mean_std_norm_batch(self):
        test_array = np.arange(8 * 16 * 16 * 3).reshape((8, 16, 16, 3))
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normed = invert_mean_std_normalization(test_array, mean=mean, std=std)
        self.assertAlmostEqual(normed.sum(), 4267663.36)

    def test_mean_std_norm_channels_first(self):
        test_array = np.arange(16 * 16 * 3).reshape((3, 16, 16))
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normed = invert_mean_std_normalization(
            test_array, mean=mean, std=std, channels="first"
        )
        self.assertAlmostEqual(normed.sum(), 66646.016)

    def test_mean_std_norm_channels_first_batch(self):
        test_array = np.arange(8 * 16 * 16 * 3).reshape((8, 3, 16, 16))
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normed = invert_mean_std_normalization(
            test_array, mean=mean, std=std, channels="first"
        )
        self.assertAlmostEqual(normed.sum(), 4265574.4)


if __name__ == "__main__":
    unittest.main()
