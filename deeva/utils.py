import os
import random
from collections import Counter
from copy import copy

import streamlit as st
import numpy as np

from configs import configs


def lucky(p):
    """
    Returns True with probability `p`, where `p` is between 0 and 1.

    Args:
        p (float): The probability of returning True, between 0 and 1.

    Returns:
        bool: True approximately `p` percent of the time, otherwise False.
    """
    return random.random() < p

def has_ext(filename: str) -> bool:
    """
    Whether filename has an extension
    """
    return filename.split('.')[-1].lower() in configs.COMMON_EXTENSIONS


def get_name(filename: str) -> str:
    """
    Return filename without its extension
    """
    if not has_ext(filename):
        return filename
    return filename[:filename.rfind('.')]


def get_ext(filename: str) -> str:
    """
    Return extension of the filename
    """
    return filename.split('.')[-1]


def is_image(filename: str) -> bool:
    """
    Whether filename is an image
    """
    return get_ext(filename) in configs.IMAGE_EXTENSIONS


def is_label(filename: str) -> bool:
    """
    Whether filename is a label
    """
    return get_ext(filename) in configs.LABEL_EXTENSIONS


def is_path(x: str) -> bool:
    """
    Whether x is a valid path
    """
    return os.path.exists(os.path.dirname(x))


def complete_with_ext(original_path: str, filenames: iter) -> list:
    """
    Complete filenames with extensions

    :param original_path: Path to the directory containing the original(completed) files
    :param filenames: List of filenames without extensions
    :return: List of full filenames with extensions that match the given filenames
    """
    all_files = os.listdir(original_path)
    match_dict = {get_name(file): file for file in all_files}
    return [match_dict[filename] for filename in filenames]


def reset_state(session_state, defaults: dict, but: tuple=()) -> None:
    """
    Reset st.session_state params to their default values

    :param session_state: Dictionary representing the current state that will be reset
    :param defaults: Dictionary containing default values to reset the state with
    :param but: Iterable of keys that should be ignored and not reset
    :return: None
    """
    for key, value in defaults.items():
        if key in session_state:
            if key in but:
                continue
            session_state.pop(key)
        set_state_for(key, value, session_state)


def set_state_for(key: str, value, session_state) -> None:
    """
    Add var to st.session_state

    :param key: The key for which the state is being set
    :param value: The value to set in the state
    :param session_state: The dictionary representing the state
    :return: None
    """
    if key not in session_state:
        session_state[key] = value


def check_int(x: str) -> bool:
    """
    Whether x can be converted to int
    """
    try:
        int(x)
        return True
    except (ValueError, TypeError):
        return False


def check_float(x: str) -> bool:
    """
    Whether x can be converted to float
    """
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def cycle_fill(lst: list, expected_length: int) -> list:
    """
    Cycle through the original elements of the list until the expected length is reached

    :param lst: The list to be extended
    :param expected_length: The desired length for the list
    :return: The extended list
    """
    while len(lst) < expected_length:
        lst += lst[:min(len(lst), expected_length - len(lst))]
    return lst


def double_callback(f1: callable, f2: callable) -> None:
    """
    Use 2 functions with one call

    :param f1: The first function to be called
    :param f2: The second function to be called
    :return: None
    """
    f1()
    f2()


def closest_odd(x: float) -> int:
    """
    Closest odd value to x
    """
    x = int(round(x))
    return x if x % 2 != 0 else x + 1


def most_common(lst: list):
    """
    Get most common element in a list

    :param lst: List of elements from which the most common element is to be identified
    :return: The most common element in the list
    """
    data = Counter(lst)
    return data.most_common(1)[0][0]


def sample_proportionally(lst: list, n: int) -> list:
    """
    Sample n items proportionally from a given list

    :param lst: A list to sample items from
    :param n: integer defining number of items to sample
    :return: list of sampled items
    """
    # Get n evenly spaced indices from the list
    indices = np.linspace(0, len(lst) - 1, n, dtype=int)
    return [lst[i] for i in indices]


@st.cache_data(show_spinner=False)
def get_colormap(keys: list, colors: list) -> dict:
    """
    Get a colormap from lists of keys and colors

    :param keys: A list of names for which the colormap is to be generated
    :param colors: A list of color values to be mapped to the classes
    :return: A dictionary mapping each key to a corresponding color value
    """
    n_keys = len(keys)
    n_colors = len(colors)

    colors = copy(colors)
    colors = cycle_fill(colors, n_keys)

    if n_keys != n_colors:
        colors = sample_proportionally(colors, n_keys)

    colormap = {k: v for k, v in zip(keys, colors)}
    return colormap


class PageScroll:
    """
    A class to manage scrolling through pages within a specified range
    """

    def __init__(self, start, end):
        self._start = start
        self._end = end
        self._current = start

    def next(self):
        if self._end == self.current:
            raise ValueError('Page number out of limits')
        self._current += 1
        return self.current

    def previous(self):
        if self._start == self.current:
            raise ValueError('Page number out of limits')
        self._current -= 1
        return self.current

    @property
    def current(self):
        return self._current


@st.cache_data(show_spinner=False)
def render_html(html_filepath: str, **kwargs) -> None:
    """
    Render HTML file

    :param html_filepath: The path to the HTML file to be rendered
    :param kwargs: Optional keyword arguments for formatting the HTML content
    :return: None
    """
    # Read the HTML file
    with open(html_filepath, 'r') as file:
        html_code = file.read()

    if kwargs:
        html_code = html_code.format(**kwargs)

    st.markdown(html_code, unsafe_allow_html=True)
