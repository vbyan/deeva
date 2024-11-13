import base64
import colorsys
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image


class ViewPort:
    """
    Creates a callable viewport unit scaler.

    Args:
        size (int): The viewport dimension in pixels

    Usage:
        >>> vh = ViewPort(800)  # viewport size of 800px
        >>> vh(50)  # 400px (50% of viewport)
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, percent: float) -> float:
        """Scales viewport by percentage (0-100)"""
        if not 0 <= percent <= 100:
            raise ValueError(f"Percentage {percent} must be between 0 and 100")
        return (percent / 100) * self.size

    def __repr__(self):
        return f"ViewPort(size={self.size})"

    @property
    def full(self):
        return self.size

def add_caching_vars(page_name: str, params: dict):
    """
    :param page_name: The base name of the page for which caching variables need to be generated.
    :param params: A dictionary containing existing parameters that need to be merged with the caching variables.
    :return: A dictionary containing the generated caching variables merged with the provided parameters.
    """
    caching_vars = {
        f'{page_name}_input_page': True,
        f'{page_name}_sample_size': None,
        f'{page_name}_use_cached_backup': False,
        f'{page_name}_cache_backup': False,
        f'{page_name}_forget_cached_backup': False,
    }

    return {**caching_vars, **params}

@st.cache_data(show_spinner=False)
def get_image_uri(image: str | np.ndarray, gray: bool=False):
    """
    Return base64 data URI of an image

    :param image: Input image, either as a file path string or a numpy array
    :param gray: Boolean value to specify whether to convert the image to grayscale
    :return: Base64 encoded data URI of the image
    """
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = Image.fromarray(image)

    if gray:
        img = img.convert('LA')

    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')

    data_uri = base64.b64encode(img_bytes.getvalue()).decode()
    return data_uri


class ColorGenerator:
    def __init__(self, saturation: float = 0.65, lightness: float = 0.60):
        """
        Initialize the color generator with specific saturation and lightness values.

        Args:
            saturation (float): Color saturation value between 0 and 1 (default: 0.65)
            lightness (float): Color lightness value between 0 and 1 (default: 0.60)
        """
        self.saturation = saturation
        self.lightness = lightness

    def generate_colors(self, n_colors: int):
        """
        Generate n_colors distinct RGB colors.

        Args:
            n_colors (int): Number of colors to generate

        Returns:
            List of RGB tuples, where each value is between 0 and 255
        """
        colors = []
        hue = 0

        for _ in range(n_colors):
            hue += 0.8 / n_colors

            # Convert HSL to RGB
            rgb = colorsys.hls_to_rgb(hue, self.lightness, self.saturation)

            # Convert to 8-bit RGB values
            rgb_int = tuple(int(255 * x) for x in rgb)
            colors.append(rgb_int)

        return colors

    def generate_hex_colors(self, n_colors: int):
        """
        Generate n_colors distinct colors in hex format.

        Args:
            n_colors (int): Number of colors to generate

        Returns:
            List of hex color strings (e.g., '#FF0000' for red)
        """
        rgb_colors = self.generate_colors(n_colors)

        return [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in rgb_colors]



