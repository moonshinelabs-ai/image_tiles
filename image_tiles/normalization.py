"""Provides normalization functions for images in np.array format."""

from functools import partial
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike


def _format_array(image: np.ndarray, batch_operation: str, channel_operation: str):
    """Formats an array to work with the internals of the normalization
    functions. This means putting the array into channel_operation-last format
    and including a batch dimension (even if that dimension is only 1).

    Args:
        array: The input array to reformat

    Returns:
        reformatted: A reformatted array
    """
    if batch_operation == "expand":
        image = np.expand_dims(image, 0)

    if channel_operation == "to_first":
        image = np.einsum("bwhc->bcwh", image)
    elif channel_operation == "to_last":
        image = np.einsum("bcwh->bwhc", image)
    else:
        pass

    if batch_operation == "squeeze":
        image = np.squeeze(image)
    return image


def _get_format_fx(
    image: np.ndarray, image_channel_format: Literal["last", "first"]
) -> tuple[Callable, Callable]:
    add_batch = len(image.shape) == 3
    convert_ordering = image_channel_format == "first"

    apply_batch = "expand" if add_batch else None
    apply_ordering = "to_last" if convert_ordering else None
    apply_fx = partial(
        _format_array, batch_operation=apply_batch, channel_operation=apply_ordering
    )

    revert_batch = "squeeze" if add_batch else None
    revert_ordering = "to_first" if convert_ordering else None
    revert_fx = partial(
        _format_array, batch_operation=revert_batch, channel_operation=revert_ordering
    )

    return apply_fx, revert_fx


def standard_normalization(image: np.ndarray) -> np.ndarray:
    """If the image is a standard format, leave it alone. Otherwise apply
    scaling norm.

    Args:
        image: An input image to normalize, any scale and format

    Returns:
        scaled: An image in uint8 format.
    """
    _, _, c = image.shape
    if c in [1, 3, 4]:
        if image.dtype in [np.float32, np.uint8]:
            return image

    return scaling_normalization(image)


def scaling_normalization(image: np.ndarray) -> np.ndarray:
    """Normalize an image by scaling to uint8.

    This function will clip numbers under 0 and scale postive numbers.

    Args:
        image: An input image to normalize, any scale and format

    Returns:
        scaled: An image in uint8 format.
    """
    # Convert to float for scaling
    image = image.astype(float)
    # Clip negative numbers, but leave small positive numbers alone.
    image[image <= 0] = 0
    # Scale from 0 to 1
    image /= np.max(image)
    # Scale from 0 to 255
    image *= 255

    return image.astype(np.uint8)


def sigmoid_normalization(
    image: np.ndarray,
    c: float = 10.0,
    th: float = 0.125,
    channels: Literal["last", "first"] = "last",
) -> np.ndarray:
    """Normalize an image by a sigmoid function like in xarray-spatial. The
    function accepts both HWC and BHWC arrays.

    Normalize by the function
    ``normalized_pixel = 1 / (1 + np.exp(c * (th - normalized_pixel)))``

    Args:
        image: An input image to normalize.
        c: Contrast controlling parameter.
        th: Brightness controlling parameter.
    """
    apply_fx, revert_fx = _get_format_fx(image, channels)
    image = apply_fx(image)

    max_pixel_val = 255

    min_val = np.nanmin(image, axis=(1, 2, 3))
    max_val = np.nanmax(image, axis=(1, 2, 3))
    range_val = np.maximum(max_val - min_val, 1)

    norm = (image - min_val.reshape((-1, 1, 1, 1))) / range_val.reshape((-1, 1, 1, 1))
    # sigmoid contrast enhancement
    norm = 1 / (1 + np.exp(c * (th - norm)))
    normalized = norm * max_pixel_val

    return revert_fx(normalized).astype(np.uint8)


def sentinel_truecolor_image(image: np.ndarray, normalizer: int = 2000) -> np.ndarray:
    """Convert a sentinel l2a or l1a product to a true color image, using
    published values of maximum reflectance.

    Args:
        image: An unnormalized sentinel-2 image.
        normalizer: The value to be used for 255, matches published TCI algo.

    Returns:
        rgb: A true-color image of the input.
    """
    assert image.shape[2] == 3, "Expect a 3 channel RGB image"

    normalized_image = np.array(image)
    saturated_idx_flat = (normalized_image > normalizer).max(axis=2)
    saturated_idx = np.repeat(np.expand_dims(saturated_idx_flat, 2), 3, axis=2)

    normalized_image = image / float(normalizer)
    rgb = normalized_image * 255
    rgb[saturated_idx] = 255
    return (rgb).astype(np.uint8)


def invert_mean_std_normalization(
    image: np.ndarray,
    mean: ArrayLike,
    std: ArrayLike,
    channels: Literal["last", "first"] = "last",
) -> np.ndarray:
    """Invert an image normalized by a set mean and standard deviation. The
    function accepts both HWC and BHWC arrays.

    Args:
        image: An input image to normalize.
        c: Contrast controlling parameter.
        th: Brightness controlling parameter.
    """
    mean = np.expand_dims(mean, (0, 1, 2))
    std = np.expand_dims(std, (0, 1, 2))

    apply_fx, revert_fx = _get_format_fx(image, channels)
    image = apply_fx(image)
    normalized = (image * std) + mean

    return revert_fx(normalized)
