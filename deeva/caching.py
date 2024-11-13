import os
import pandas as pd
import streamlit as st

def remember_path(data_path: str) -> None:
    """
    Cache provided data_path if valid

    :param data_path: The path to be remembered and stored in the cache file.
    :return: None
    """
    if not os.path.exists('.cache'):
        os.mkdir('.cache')

    if not os.path.exists('.cache/paths.txt'):
        with open('.cache/paths.txt', 'a') as f:
            f.write(f'{data_path}\n')
        return

    with open('.cache/paths.txt', 'r') as f:
        saved = f.read().splitlines()

    if data_path not in saved:
        with open('.cache/paths.txt', 'a') as f:
            f.write(f'{data_path}\n')


def forget_path(data_path: str) -> None:
    """
    Delete cached data_path

    :param data_path: The path to be removed from the cached paths file.
    :return: None
    """
    with open('.cache/paths.txt', "r") as f:
        lines = f.readlines()
    with open('.cache/paths.txt', "w") as f:
        for line in lines:
            if line.strip("\n") != data_path:
                f.write(line)


def get_saved_paths() -> list:
    """
    Retrieve and return a list of saved file paths from a cache file.

    :return: A list of file paths if cache file exists, otherwise an empty list.
    """
    if os.path.exists('.cache/paths.txt'):
        with open('.cache/paths.txt', 'r') as f:
            return f.read().splitlines()


def cache_stats(stats: pd.DataFrame, path: str, key: str) -> None:
    """
    Cache pd.DataFrame containing statistics

    :param stats: The pandas DataFrame or Series that contains the stats information to be cached.
    :param path: The file path to the HDF5 store where stats will be saved.
    :param key: The key under which to store the stats in the HDF5 file.
    :return: None. The function performs file operations to cache the stats.
    """
    if not os.path.exists('.cache'):
        os.mkdir('.cache')

    if not os.path.exists(path):
        stats.to_hdf(path, key=key, mode='w')
        return

    with pd.HDFStore(path, mode='a') as store:
        if key in store:
            store.remove(key)
        stats.to_hdf(store, key=key)


@st.cache_data(show_spinner=False)
def get_cached_stats(path: str, key: str) -> pd.DataFrame | None:
    """
    Retrieve and return a pd.DataFrame from a cache file.

    :param path: The file path to the HDF5 file.
    :param key: The key within the HDF5 file to access the desired dataset.
    :return: Returns the dataset corresponding to the given key if it exists, otherwise returns None.
    """
    if os.path.exists(path):
        store = pd.HDFStore(path)
        if key in store:
            store.close()
            return pd.read_hdf(path, key)
        store.close()


def forget_stats(path: str, key: str) -> None:
    """
    Delete cached pd.DataFrame

    :param path: The file path to the HDF5 file where the data is stored.
    :param key: The key for the specific data set within the HDF5 file to be removed.
    :return: None
    """
    store = pd.HDFStore(path)
    if key in store:
        store.remove(key)
    store.close()
