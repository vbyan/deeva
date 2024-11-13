import streamlit as st

import itertools
import random
import threading
from functools import partial
from stqdm import stqdm
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .utils import check_image, get_color_distribution, _get_progress_string, truncate_string

from loaders.utils import *
from configs import configs
from loaders.converters import VOC2YOLO, yolo_to_voc
from loaders.parsers import parse_yolo, parse_voc
from utils import is_image, is_label, has_ext, get_name, complete_with_ext, get_ext


class DataMatch:
    """
    Class to manage and analyze data files in a structured directory.

    Parameters
    ----------
    data_path : str
       Path to the root directory containing the data
    """

    class Stats:
        """
        Handles the statistics related to a category ('images' or 'labels')

        Parameters
        ----------
        category : str
           Specifies the category of files being processed ('images' or 'labels').

        filenames : list of str
           List of filenames to be analyzed.
        """

        def __init__(self, category: str, filenames: list):
            self.category = category
            self.filenames = sorted(filenames)

            self.no_ext = []
            self.wrong_format = []
            self.duplicates = []
            self.correct = []

            self.stats_ = None

        def _is_correct_format(self) -> callable:
            if self.category == 'images':
                return is_image
            if self.category == 'labels':
                return is_label

        def get_stats(self) -> dict:
            if not self.stats_:
                stats = self._collect_stats()
                self.stats_ = stats

            return self.stats_

        def _collect_stats(self) -> dict:
            """
            Collect statistics by categorizing filenames into
                - no extension
                - wrong format
                - duplicates
                - correct.

            Returns
            -------
            dict
               A dictionary containing categorized filenames.
            """

            for filename in self.filenames:
                if not has_ext(filename):
                    self.no_ext.append(filename)
                    continue
                if not self._is_correct_format()(filename):
                    self.wrong_format.append(filename)
                    continue

                self.correct.append(filename)

            # collecting duplicates
            for i in range(1, len(self.filenames)):
                if get_name(self.filenames[i - 1]) == get_name(self.filenames[i]):
                    if all(duplicate not in self.no_ext + self.wrong_format
                           for duplicate in self.filenames[i - 1:i + 1]):
                        self.duplicates.append(self.filenames[i])

            stats = {'No extension': self.no_ext,
                    'Wrong format': self.wrong_format,
                    'Duplicates by filename': self.duplicates,
                    'Correct': self.correct}

            return stats

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.image_filenames, self.label_filenames = self._get_filenames(data_path)
        self.image_stats = self.Stats(category='images', filenames=self.image_filenames)
        self.label_stats = self.Stats(category='labels', filenames=self.label_filenames)

    @staticmethod
    def _get_filenames(data_dir: str):
        """
        Get corresponding filenames at root directory

        Returns
        -------
        tuple
            A tuple of two filename lists ('images' and 'labels')
        """
        images_path = os.path.join(data_dir, 'images')
        labels_path = os.path.join(data_dir, 'labels')

        image_filenames = os.listdir(images_path)
        label_filenames = os.listdir(labels_path)

        return image_filenames, label_filenames

    @st.cache_data(show_spinner=False)
    def get_layout(_self) -> dict:
        image_stats = _self.image_stats.get_stats()
        label_stats = _self.label_stats.get_stats()

        layout = dict(images=image_stats,
                      labels=label_stats)

        return layout

    @st.cache_data(show_spinner=False)
    def get_garbage(_self) -> dict:
        """Return all corrupted categories as a dict of dicts"""
        garbage = dict()
        garbage['images'] = dict(no_ext=_self.image_stats.no_ext,
                                 wrong_format=_self.image_stats.wrong_format,
                                 duplicates=_self.image_stats.duplicates)

        garbage['labels'] = dict(no_ext=_self.label_stats.no_ext,
                                 wrong_format=_self.label_stats.wrong_format,
                                 duplicates=_self.label_stats.duplicates)
        return garbage

    @st.cache_data(show_spinner=False)
    def get_matching(_self, garbage: dict) -> dict:
        """
        Match images and labels

        Returns
        -------
        dict
            Matching statistics with matched and unmatched categories
        """

        image_trash, label_trash = garbage['images'], garbage['labels']

        # filenames without extension
        image_names = set([get_name(image) for image in _self.image_filenames if
                           image not in itertools.chain(*image_trash.values())])
        label_names = set([get_name(label) for label in _self.label_filenames if
                           label not in itertools.chain(*label_trash.values())])

        lonely_images = image_names - label_names
        lonely_labels = label_names - image_names

        matched_images = image_names - lonely_images
        matched_labels = label_names - lonely_labels

        matched_images = complete_with_ext(os.path.join(_self.data_path, 'images'),
                                           matched_images)
        matched_labels = complete_with_ext(os.path.join(_self.data_path, 'labels'),
                                           matched_labels)
        lonely_images = complete_with_ext(os.path.join(_self.data_path, 'images'),
                                          lonely_images)
        lonely_labels = complete_with_ext(os.path.join(_self.data_path, 'labels'),
                                          lonely_labels)

        matching = {'Matched images': sorted(matched_images),
                    'Matched labels': sorted(matched_labels),
                    'Lonely images': lonely_images,
                    'Lonely labels': lonely_labels}

        return matching


class Annotations:
    """
    Class to manage and analyze annotations

    Parameters
    ----------
    data_path : str
       Path to the root directory containing the data

    labels : iterable of str
       Annotation filenames to be processed.

    labelmap : dict
       A dictionary mapping class IDs to class names
    """

    def __init__(self, data_path: str, labels: iter, labelmap: dict):
        self.data_path = data_path
        self.labelmap = labelmap
        self.labels = labels
        self.corrupted = []
        self.yolo = []
        self.voc = []
        self.background = []

    @st.cache_data(show_spinner=False)
    def categorize_by_formats(_self) -> dict:
        """
        Check annotation files for corruption and categorize them

        Returns
        -------
        dict
           A dictionary with categorized annotations
        """
        for label in _self.labels:
            check_correct = check_yolo if label.endswith('.txt') else check_voc
            bucket = _self.yolo if label.endswith('.txt') else _self.voc
            check_empty = check_yolo_empty if label.endswith('.txt') else check_voc_empty

            okay = check_correct(os.path.join(_self.data_path, 'labels', label))
            empty = check_empty(os.path.join(_self.data_path, 'labels', label))
            if okay:
                if empty:
                    _self.background.append(label)
                bucket.append(label)
            else:
                _self.corrupted.append(label)

        formats = dict(yolo=_self.yolo,
                       voc=_self.voc,
                       corrupted=_self.corrupted,
                       background=_self.background)

        return formats

    def bring_to(self, annotation_format_stats: dict, destination_format: str,
                 matching_dict: dict) -> None | list:
        """
        Convert annotations from one format to another (YOLO <-> VOC)

        Parameters
        ----------
        annotation_format_stats : dict
           A dictionary containing lists of filenames categorized by format.

        destination_format : str
           The format to which annotations should be converted ('yolo' or 'voc').

        matching_dict : dict
           A dictionary mapping annotation filenames to corresponding image filenames.

        Returns
        -------
        list of str
           A list of annotation filenames that could not be converted due to missing corresponding images.
        """
        voc_to_yolo = VOC2YOLO(labelmap=self.labelmap)

        convert = voc_to_yolo if destination_format == 'yolo' else yolo_to_voc
        write = write_yolo if destination_format == 'yolo' else write_voc
        bucket = annotation_format_stats['voc'] if destination_format == 'yolo' else annotation_format_stats['yolo']

        not_converted = []
        for label in stqdm(bucket, desc="Converting...   "):
            if label not in matching_dict.keys():
                not_converted.append(label)  # without corresponding image
                continue
            image_path = os.path.join(self.data_path, 'images', matching_dict[label])
            label_path = os.path.join(self.data_path, 'labels', label)

            converted_annotation, destination = convert(label_path, self.labelmap, image_path)
            write(converted_annotation, destination)
            os.remove(label_path)

        # labelmap generated using voc annotations
        if voc_to_yolo.labelmap_:
            with open(os.path.join(self.data_path, 'labelmap.txt'), 'w') as f:
                for class_name in voc_to_yolo.labelmap_:
                    f.write(f"{class_name}\n")

        return not_converted

    @st.cache_data(show_spinner=False)
    def get_stats(_self) -> pd.DataFrame:
        """
        Generate a DataFrame of annotation statistics

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame of annotation statistics
        """
        filenames = []
        yolo_instances = []
        for yolo_annotation in _self.yolo:
            file_path = os.path.join(_self.data_path, 'labels', yolo_annotation)
            instances = parse_yolo(file_path, labelmap=_self.labelmap)

            yolo_instances.extend(instances)
            filenames.extend([yolo_annotation] * len(instances))

        voc_instances = []
        for voc_annotation in _self.voc:
            file_path = os.path.join(_self.data_path, 'labels', voc_annotation)
            instances = parse_voc(file_path)

            voc_instances.extend(instances)
            filenames.extend([voc_annotation] * len(instances))

        instances = yolo_instances + voc_instances

        stats = pd.DataFrame(instances, columns=configs.ANNOTATION_STATS_COLUMNS)
        stats = stats.astype({'x_center': float, 'y_center': float, 'width': float, 'height': float})

        truncate_class_names = partial(truncate_string, limit=configs.CLASS_NAME_LENGTH)
        stats['class_name'] = stats['class_name'].apply(truncate_class_names)

        stats['box_size'] = stats['width'] * stats['height']
        stats['filename'] = filenames

        return stats


class Images:
    """
    Class to manage and analyze images

    Parameters
    ----------
    data_path : str
       Path to the root directory containing the data

    images : iterable of str
       List of image filenames to be processed.

    annotation_stats : pandas.DataFrame
       Output of Annotations.get_stats method
    """

    def __init__(self, data_path: str, images: iter, annotation_stats: pd.DataFrame):
        self.data_path = data_path
        self.image_paths = [os.path.join(data_path, 'images', image) for image in images]
        self.annotation_stats = annotation_stats
        self.verified_paths = self._get_verified_paths()
        self.n_images = len(images)

    @st.cache_data(show_spinner=False)
    def _get_verified_paths(_self) -> list:
        verified_paths = [image_path for image_path in _self.image_paths
                          if check_image(image_path)]
        return verified_paths

    def get_format_counts(self) -> dict:
        """
        Check image files for corruption and categorize them

        Returns
        -------
        dict
            A dictionary of image format statistics
        """
        verified_count = len(self.verified_paths)

        # Use a Counter to count extensions directly and avoid list manipulation
        format_counts = Counter(get_ext(path.split('/')[-1]) for path in self.verified_paths)

        # Add 'corrupted' count directly to the Counter
        format_counts['corrupted'] = self.n_images - verified_count

        return dict(format_counts)

    @staticmethod
    def _process_image(image_path: str, stop_event: threading.Event) -> list | None:
        """
        Process a single image to extract various properties.

        Parameters
        ----------
        image_path : str
            The file path of the image to be processed.

        Returns
        -------
        list
            A list of extracted properties
        """

        # stop execution if stop is triggered
        if stop_event.is_set():
            return

        image_array = cv2.imread(image_path)
        height, width, _ = image_array.shape

        # grayscale if channels are identical
        c1, c2, c3 = cv2.split(image_array)
        channels_identical = np.all(c1 == c2) and np.all(c2 == c3)
        mode = 'mono' if channels_identical else 'rgb'

        image_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # extracting contrast
        contrast_rms = image_gray.std()

        # extracting color distribution
        color_distribution = get_color_distribution(image_array, mono=channels_identical)
        # getting rid of color names
        color_distribution = list(itertools.chain(*color_distribution.values()))

        # extracting image brightness
        brightness = np.mean(image_gray)

        image_props = [image_path, height, width,
                       mode, contrast_rms, brightness]

        # color_distribution is a list of size 28 <-> 7(colors)x4(c,h,s,v)
        image_props.extend(color_distribution)

        return image_props

    @st.cache_data(show_spinner=False)
    def get_stats(_self, sample_size: int, cache_key: int) -> pd.DataFrame | None:
        """
        Analyze a random sample of images to gather detailed statistics on image properties.

        Parameters
        ----------
        sample_size : int
            The number of images to use
        cache_key : int
            Key to control st.cache_data

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the extracted image statistics
        """

        n_colors = len(configs.HUE_COLORS)
        n_stats = len(configs.IMAGE_STATS_COLOR_COLUMNS)
        n_overalls = len(configs.IMAGE_STATS_OVERALL_COLUMNS)

        index_level_0 = ['overall'] * n_overalls + [color for color in configs.HUE_COLORS for _ in range(n_stats)]
        index_level_1 = configs.IMAGE_STATS_OVERALL_COLUMNS + configs.IMAGE_STATS_COLOR_COLUMNS * n_colors

        # multilevel(2) columns to separate overall properties from color stats
        columns = pd.MultiIndex.from_arrays([index_level_0, index_level_1])

        sampled_paths = random.sample(_self.verified_paths, sample_size)

        image_stats = []
        bar = st.progress(0)  # Initialize the progress bar

        stop_event = threading.Event()
        with open(os.devnull, 'w') as null:  # Redirect to null, or use a log file if you want to capture output
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                # Submit tasks to the executor
                futures = {executor.submit(_self._process_image,
                                           path, stop_event): path for path in sampled_paths}

                # Track completed tasks, but redirect tqdm to a non-terminal output
                iterable = enumerate(as_completed(futures), start=1)
                with tqdm(iterable, total=len(futures), file=null) as pbar:
                    try:
                        for idx, future in pbar:
                            if st.session_state.images_cancel:
                                stop_event.set()  # Signal all threads to stop
                                # Cancel all pending futures
                                for f in futures:
                                    f.cancel()

                                # disable cancellation trigger
                                st.session_state.images_cancel = False
                                return

                            result = future.result()
                            image_stats.append(result)

                            progress = idx / sample_size
                            prefix = 'Scanning images ...'
                            progress_string = _get_progress_string(pbar, idx, sample_size, prefix)

                            bar.progress(progress, text=progress_string)

                    finally:
                        stop_event.set()
                        bar.empty()  # Clear the progress bar after completion

        image_stats = pd.DataFrame(image_stats, columns=columns)
        image_stats = image_stats.sort_values([('overall', 'filepath')])

        # merge with annotation_stats to connect images to the objects they contain
        annotations_key = _self.annotation_stats['filename'].apply(get_name)
        images_key = image_stats['overall']['filepath'].apply(lambda x: get_name(x.split('/')[-1]))
        combined = pd.merge(image_stats['overall'], _self.annotation_stats, left_on=images_key,
                            right_on=annotations_key, how='left')

        combined['class_name'] = combined['class_name'].fillna('background')

        # add class-objects column to image-stats using combined DataFrame
        image_stats['overall', 'class_objects'] = combined.groupby('filepath')[
            'class_name'].unique().sort_index().values

        return image_stats


    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_tone_counts(image_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Extract color tones using hsv

        Parameters
        ----------
        image_stats : pandas.DataFrame
            Output of Images.get_stats method

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the extracted color tones where each row represents a tone
        """
        color_cols = [c for (c, _, _) in configs.HUE_RANGES[:-1]]
        color_stats = image_stats.loc[:, color_cols].stack(level=0, future_stack=True).droplevel(1)

        tones = color_stats.groupby(['hue', 'sat', 'value'])['count'].sum().sort_index()
        tones = tones.reset_index()

        # shrink tone counts into 1-100 range
        scaler = MinMaxScaler((1, 100))
        tones['count'] = scaler.fit_transform(tones['count'].values.reshape(-1, 1)).astype('int32')

        # Expand hue_tones using the counts
        tones = tones.loc[tones.index.repeat(tones['count'])].reset_index(drop=True)

        sorted_hue_indices = tones.groupby('hue')['count'].sum().sort_values(ascending=False).index

        # Sort the DataFrame based on the sorted 'hue' groups
        tones = tones.set_index('hue').loc[sorted_hue_indices].reset_index()

        # converting to fractions
        tones['hue'] = tones['hue'] / 180
        tones['sat'] = tones['sat'] / 255
        tones['value'] = tones['value'] / 255

        return tones

    @property
    def n_verified(self) -> int:
        return len(self.verified_paths)


class Overlaps:
    """
    Class to manage and analyze overlaps between objects

    Parameters
    ----------
    annotation_stats : pandas.DataFrame
       Output of Annotations.get_stats method
    """

    def __init__(self, annotation_stats: pd.DataFrame, matching_dict: dict, data_path: str):
        self.annotation_stats = annotation_stats.sort_values(['filename', 'class_name'])
        self.matching_dict = matching_dict
        self.data_path = data_path

    @property
    def n_verified(self) -> int:
        return self.annotation_stats['filename'].nunique()

    def _get_image_filepath(self, filename: str) -> str | None:
        """For an annotation filename return corresponding image path"""
        if filename not in self.matching_dict:
            return None
        return os.path.join(self.data_path, 'images', self.matching_dict[filename])

    @staticmethod
    def _calculate_overlap(pair_index: tuple, data: pd.DataFrame):
        """
        Calculate the overlap area between 2 bounding boxes

        Parameters
        ----------
        pair_index : tuple
            A tuple containing the indices of the two bounding boxes to compare.

        data : pd.DataFrame
            The DataFrame containing the bounding box information for the images.

        Returns
        -------
        list
            A list containing two sublists with detailed information about the overlap between the two bounding boxes
        """
        index1, index2 = pair_index
        bbox1, bbox2 = data.loc[index1], data.loc[index2]

        # Calculate the distance between the centers
        dx = abs(bbox1.x_center - bbox2.x_center)
        dy = abs(bbox1.y_center - bbox2.y_center)

        # Calculate the overlap in each dimension
        overlap_width = max(0, bbox1.width / 2 + bbox2.width / 2 - dx)
        overlap_height = max(0, bbox1.height / 2 + bbox2.height / 2 - dy)

        # Calculate the area of each bounding box and the overlapping area
        overlap_size = min(bbox1.box_size, bbox2.box_size, overlap_width * overlap_height)

        # overlap info from the perspective of the first object
        relation = [bbox1.filename, bbox1.class_name, bbox2.class_name, index1, index2,
                    bbox1.box_size, bbox2.box_size, overlap_size, bbox1.x_center, bbox1.y_center,
                    bbox1.width, bbox1.height, bbox2.x_center, bbox2.y_center, bbox2.width, bbox2.height]

        # overlap info from the perspective of the second object
        reverse_relation = [bbox2.filename, bbox2.class_name, bbox1.class_name, index2, index1,
                            bbox2.box_size, bbox1.box_size, overlap_size, bbox2.x_center, bbox2.y_center,
                            bbox2.width, bbox2.height, bbox1.x_center, bbox1.y_center, bbox1.width, bbox1.height]

        return [relation, reverse_relation]

    @st.cache_data(show_spinner=False)
    def get_stats(_self, sample_size: int, cache_key: int) -> pd.DataFrame | None:
        """
        Sample a subset of images and compute the overlap statistics for all pairs of bounding boxes

        Parameters
        ----------
        sample_size : int
            The number of unique image filenames to sample and analyze.

        cache_key : int
            Key to control st.cache_data

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the overlap statistics for the sampled images
        """

        sampled_filenames = random.sample(_self.annotation_stats['filename'].unique().tolist(), sample_size)
        sampled_annotations = _self.annotation_stats[_self.annotation_stats['filename'].isin(sampled_filenames)]

        calculate_overlap_data = partial(_self._calculate_overlap, data=sampled_annotations)
        grouped = sampled_annotations.groupby('filename')

        # Create pairwise indices
        combinations = [list(itertools.combinations(group.index.values, 2)) for _, group in grouped]
        pairwise_indices = list(itertools.chain.from_iterable(combinations))

        overlaps = []
        pbar = stqdm(pairwise_indices, total=len(pairwise_indices), desc='Calculating overlaps ...')
        try:
            for pair in pbar:
                for item in calculate_overlap_data(pair):
                    # cancel triggered
                    if st.session_state.overlaps_cancel:
                        st.session_state.overlaps_cancel = False
                        return
                    overlaps.append(item)

        finally:
            pbar.close()

        overlap_stats = pd.DataFrame(overlaps, columns=configs.OVERLAP_STATS_COLUMNS)
        overlap_stats['size'] = sample_size  # number of images attached to overlap stats

        overlap_stats['image_filepath'] = overlap_stats['filename'].apply(_self._get_image_filepath)

        return overlap_stats


def render_layout(session_state, layout: dict, matching: dict, delete: bool=False) -> None:
    """
    Render layout configuration

    Parameters
    ----------
    session_state: st.session_state
        Current session_state
    layout: dict
        Output of DataMatch.get_layout method
    matching: dict
        Output of DataMatch.get_matching method
    delete: bool
        Delete if True, separate if false
    """

    for category in configs.CATEGORIES: # labels and images
        for attr, attr_short, attr_short_remove in zip(configs.DATAMATCH_ATTRIBUTES, configs.DATAMATCH_ATTRIBUTES_SHORT,
                                                       configs.DATAMATCH_ATTRIBUTES_SHORT_REMOVE):

            if not layout[category][attr]: # if count for attr is 0
                continue
            if session_state[attr_short_remove]: # if attr is set to be removed
                destination = os.path.join(session_state['data_path'], f'{attr_short}_{category}')

                for filename in layout[category][attr]:
                    if delete:
                        # removing altogether
                        os.remove(os.path.join(session_state['data_path'], category, filename))
                    else:
                        if not os.path.exists(destination):
                            os.mkdir(destination)
                        # moving to destination
                        os.replace(os.path.join(session_state['data_path'], category, filename),
                                   os.path.join(destination, filename))

        if session_state['remove_lnl']: # if lonely files are to be removed
            if not matching[f'Lonely {category}']:
                continue
            destination = os.path.join(session_state['data_path'], f'lonely_{category}')
            condemned = matching[f'Lonely {category}']
            for filename in condemned:
                if delete:
                    os.remove(os.path.join(session_state['data_path'], category, filename))
                else:
                    if not os.path.exists(destination):
                        os.mkdir(destination)
                    # moving to destination
                    os.replace(os.path.join(session_state['data_path'], category, filename),
                               os.path.join(destination, filename))


def add_background_labels(data_path: str, layout: dict,
                          matching: dict, actual_fill: bool=True) -> None | list:
    """
    Add empty files as background labels

    Parameters
    ----------
    data_path: str
        Path to the root directory containing the data
    layout: dict
        Output of DataMatch.get_layout method
    matching: dict
        Output of DataMatch.get_matching method
    actual_fill: bool
        A boolean flag indicating whether to actually add the background labels or just return file names
    """

    # find most common extension (annotation format)
    extensions = [get_ext(f) for f in layout['labels']['Correct']]
    extension_counts = Counter(extensions)
    most_common_ext = extension_counts.most_common(1)[0][0]

    destinations = []
    if actual_fill:
        get_empty_annotation = empty_yolo if most_common_ext == 'txt' else empty_voc
        write = write_yolo if most_common_ext == 'txt' else write_voc

        for filename in matching['Lonely images']:
            annotation, destination = get_empty_annotation(filename, data_path)
            write(annotation, destination)
            destinations.append(destination)

        return
    return [os.path.basename(destination) for destination in destinations]



