import os
import yaml
from lxml import etree

import streamlit as st
from configs import configs


def parse_yolo(annotation_path: str, labelmap: dict | None) -> list:
    """
    Parse YOLO file

    :param annotation_path: Path to YOLO file
    :param labelmap: Dictionary to map label indices to label names
    :return: List of parsed YOLO annotations with label mapping applied
    """
    instances = []
    with open(annotation_path, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            instance = line.strip().split(' ')
            instance = list(map(float, instance))
            instance[0] = int(instance[0])

            if labelmap:
                instance[0] = labelmap[str(instance[0])]

            instances.append(instance)
    return instances


def parse_voc(annotation_path: str, return_tree=False):
    """
    Parse VOC file

    :param annotation_path: Path to VOC file
    :param return_tree: Return parsed tree or instances
    :return: List of parsed VOC annotations
    """
    tree = etree.parse(annotation_path)
    if return_tree:
        return tree

    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    instances = []
    for obj in root.iter('object'):
        obj_class = obj.find('name').text
        bbox = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)

        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        w = (x_max - x_min) / width
        h = (y_max - y_min) / height

        instance = [obj_class, x_center, y_center, w, h]
        instances.append(instance)

    return instances

@st.cache_data(show_spinner=False)
def get_labelmap(data_path: str):
    """
    Locate and parse labelmap from dir

    :param data_path: Path to the directory containing labelmap file
    :return: A dictionary representing the labelmap if correctly formatted;
            'Labelmap not found' if no valid labelmap file is found;
            'Incorrect labelmap' if the file content does not match the expected pattern.
    """

    labelmap_file=None

    root_dir = os.listdir(data_path)
    for alias in configs.LABELMAP_ALIASES:
        if alias in root_dir:
            labelmap_file = alias
            break

    if not labelmap_file:
        return None, 'Labelmap not found'

    if labelmap_file == 'data.yaml':
        with open(os.path.join(data_path, labelmap_file), 'r') as f:
            stream = yaml.safe_load(f)

        if 'names' in stream.keys():
            lines = stream['names']
        else:
            return None, 'Labelmap not found'

    else:
        with open(os.path.join(data_path, labelmap_file), 'r') as f:
            lines = f.read().splitlines()

    for line in lines:
        if not line:
            return None, 'Incorrect labelmap'

    labelmap = {str(n): line.strip() for n, line in zip(range(len(lines)), lines)}

    return labelmap, None