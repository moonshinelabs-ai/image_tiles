"""Provides image utility functions."""

import io
from typing import Callable, Optional

import imageio
import numpy as np
import tifffile
from PIL import Image
from smart_open import open

from .normalization import (
    scaling_normalization,
    sentinel_truecolor_image,
    sigmoid_normalization,
    standard_normalization,
)


def _render_image_data(data: np.ndarray, render_method: str) -> np.ndarray:
    """Convert a numpy array into an image we can actually render."""
    if len(data.shape) == 2:
        # If it's one channel expand the dimensions and make it a grayscale image
        data = np.expand_dims(data, -1)
    elif data.shape[2] >= 3:
        if render_method == "rgb":
            # For RGB data, take the first 3 channels (ignore alpha)
            data = data[:, :, 0:3]
        elif render_method == "bgr":
            # For BGR data, we'll do the same as above but also invert the axes
            data = data[:, :, 0:3]
            data = data[:, :, ::-1]
        elif render_method == "sentinel":
            # For satellite data from Sentinel, we'll take 3 spectral bands
            data = data[:, :, 1:4]
            data = data[:, :, ::-1]
        elif render_method == "bw":
            # Apply greyscale to this RGB image
            data = 0.30 * data[:, :, 0] + 0.59 * data[:, :, 1] + 0.11 * data[:, :, 2]
            data = np.expand_dims(data, -1)
        else:
            raise ValueError(
                f"Not a valid type of image rendering, got {render_method}"
            )
    else:
        # If the dimensions are <2 and >3 then we have some weird tensor.
        raise ValueError(f"Image dimensionality for render is wrong ({data.shape})")

    return data


def get_supported_extensions() -> set:
    """Get a list of supported extensions for this module, this is all PIL
    extensions plus our own.

    Args: None

    Returns:
        supported_extensions: A set of possible extensions.
    """
    exts = Image.registered_extensions()
    supported_extensions = {ex for ex, f in exts.items() if f in Image.OPEN}
    additional_extensions = {".tif", ".tiff"}
    return supported_extensions.union(additional_extensions)


def read_image(
    path: str, normalize: Optional[str] = "sigmoid", render_method: str = "rgb"
) -> np.ndarray:
    """Read an image file, possibly containing many channels.

    Args:
        path: A filesystem path to the file
        normalize: How to normalize the image, if at all, one of
            {None, sigmoid, sentinel}
        render_method: Which multichannel image format to use, one
            of {bw, rgb, bgr, sentinel}

    Returns:
        raw_bytes: A bytestream containing the jpeg image
    """
    # Normalize functions supported.
    normalize_fx: dict[str, Callable] = {
        "standard": standard_normalization,
        "scaling": scaling_normalization,
        "sigmoid": sigmoid_normalization,
        "sentinel": sentinel_truecolor_image,
    }

    # Reader functions supported, but we'll fallback to PIL if we don't
    # have a specialized function here.
    format = path.split(".")[-1]
    reader_fx = {
        "tiff": tifffile.imread,
        "tif": tifffile.imread,
    }

    # Select an image reader, or fallback to PIL for anything else.
    reader = imageio.imread
    if format in reader_fx.keys():
        reader = reader_fx[format]

    # Read our image and render it into RGB for the webpage
    with open(path, "rb") as f:
        file_bytes = io.BytesIO(f.read())
        data = np.asarray(reader(file_bytes))
        rendered_data = _render_image_data(data, render_method=render_method)

        if normalize is not None:
            rendered_data = normalize_fx[normalize](rendered_data)

    return rendered_data
