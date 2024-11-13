from .parsers import parse_voc, parse_yolo

from utils import get_name
from lxml import etree
from PIL import Image
import os


def yolo_to_voc(yolo_path: str, labelmap: dict, image_path: str):
    """
    Convert YOLO annotation to VOC format

    :param yolo_path: path to YOLO annotation file
    :param labelmap: Dictionary mapping class indices to class names
    :param image_path: Path to the image file corresponding to the YOLO annotation
    :return: A tuple containing the parsed annotation tree and the destination filename for the VOC format XML file
    """
    img = Image.open(image_path)
    width, height = img.size
    depth = 3 if img.mode == 'RGB' else 1

    instances = parse_yolo(yolo_path, labelmap)

    annotation = etree.Element('annotation')
    etree.SubElement(annotation, 'folder').text = 'images'
    etree.SubElement(annotation, 'filename').text = os.path.basename(image_path)

    size = etree.SubElement(annotation, 'size')
    etree.SubElement(size, 'width').text = str(width)
    etree.SubElement(size, 'height').text = str(height)
    etree.SubElement(size, 'depth').text = str(depth)

    for instance in instances:
        obj_class = instance[0]
        if not labelmap:
            obj_class = int(obj_class)
        x_center, y_center, w, h = list(map(float, instance[1:]))

        x_min = int((x_center - w / 2) * width)
        y_min = int((y_center - h / 2) * height)
        x_max = int((x_center + w / 2) * width)
        y_max = int((y_center + h / 2) * height)

        x_min, x_max = (x_min - 1, x_max + 1) if x_min == x_max else (x_min, x_max)
        y_min, y_max = (y_min - 1, y_max + 1) if y_min == y_max else (y_min, y_max)

        x_min = max(0, x_min)
        x_max = min(x_max, width)
        y_min = max(0, y_min)
        y_max = min(y_max, height)

        obj = etree.SubElement(annotation, 'object')
        etree.SubElement(obj, 'name').text = str(obj_class)
        bbox = etree.SubElement(obj, 'bndbox')
        etree.SubElement(bbox, 'xmin').text = str(x_min)
        etree.SubElement(bbox, 'xmax').text = str(x_max)
        etree.SubElement(bbox, 'ymin').text = str(y_min)
        etree.SubElement(bbox, 'ymax').text = str(y_max)

    tree = etree.ElementTree(annotation)
    destination = get_name(yolo_path) + '.xml'
    return tree, destination



class VOC2YOLO:
    """
    Convert VOC annotation to YOLO format

    Args:
        labelmap: Dictionary mapping class names to class IDs
    """
    def __init__(self, labelmap: dict | None):
        self.labelmap = labelmap

    def __call__(self, voc_path: str, *args):
        """Parse voc file and replace class names with IDs"""
        instances = parse_voc(voc_path)

        # replace class names with labelmap values
        for i, instance in enumerate(instances):
            class_name = instance[0]
            if not self.labelmap:
                self.labelmap = {0: class_name}

            elif class_name not in self.labelmap.values():
                max_id = max(self.labelmap.keys())
                self.labelmap[max_id + 1] = class_name

            reverse_labelmap = {v: k for k, v in self.labelmap.items()}

            instances[i] = [int(reverse_labelmap[class_name]), *instance[1:]]

        destination = get_name(voc_path) + '.txt'

        return instances, destination

    @property
    def labelmap_(self):
        if not self.labelmap:
            return
        return list(self.labelmap.values())










