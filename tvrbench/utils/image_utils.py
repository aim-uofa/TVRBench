"""
Image utilities for the ActiveSpatial evaluation framework.
"""

import base64
import io

import numpy as np
from PIL import Image


def encode_image_base64(image_array: np.ndarray) -> str:
    """Encode a numpy RGB image array to a base64 PNG string."""
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
