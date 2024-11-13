import os

from PIL import Image
from lxml import etree

from utils import get_name, check_int, check_float, is_image


def write_voc(tree, destination: str) -> None:
    """
    Write XML elementtree to a file

    :param tree: An XML tree structure that needs to be written to a file.
    :param destination: The file path or file-like object where the XML tree will be written.
    :return: None
    """
    tree.write(destination, pretty_print=True, xml_declaration=True, encoding="utf-8")


def write_yolo(instances: list, destination: str) -> None:
    """
    Write yolo annotation to a text file

    :param instances: A list of YOLO annotation instances (list of lists)
    :param destination: File path where the YOLO annotation should be written.
    :return: None
    """
    with open(destination, 'w') as file:
        for instance in instances:
            file.write(" ".join(map(str, instance)) + "\n")


def empty_yolo(image_filename: str, path: str):
    """
    Return empty yolo annotation and file destination

    :param image_filename: The filename of the image for which we need to create an empty YOLO label.
    :param path: The base directory where the 'labels' subdirectory is located.
    :return: A tuple containing an empty list representing an empty YOLO label and the path for the label file.
    """
    label_filename = get_name(image_filename) + '.txt'
    destination = os.path.join(path, 'labels', label_filename)

    yolo_empty = []

    return yolo_empty, destination


def empty_voc(image_filename: str, path: str):
    """
    Return empty voc annotation and file destination

    :param image_filename: The name of the image file for which the VOC XML annotation is to be created.
    :param path: The directory path where the image is located and where the VOC XML will be saved.
    :return: A tuple containing the generated XML annotation tree and the destination path for the XML file
    """
    label_filename = get_name(image_filename) + '.xml'
    image_path = os.path.join(path, 'images', image_filename)

    img = Image.open(image_path)
    width, height = img.size
    depth = 3 if img.mode == 'RGB' else 1

    annotation = etree.Element('annotation')
    etree.SubElement(annotation, 'folder').text = 'images'
    etree.SubElement(annotation, 'filename').text = os.path.basename(image_path)

    size = etree.SubElement(annotation, 'size')
    etree.SubElement(size, 'width').text = str(width)
    etree.SubElement(size, 'height').text = str(height)
    etree.SubElement(size, 'depth').text = str(depth)

    voc_empty = etree.ElementTree(annotation)
    destination = os.path.join(path, 'labels', label_filename)

    return voc_empty, destination


def check_voc(voc) -> bool:
    """
    Check voc annotation for corruption

    :param voc: The VOC XML data, either as a file path (str) or an etree parsed tree.
    :return: A boolean indicating whether the VOC XML data is valid.
    """

    def check_subtags(_root, necessary):

        return all(
            _root.find(_tag) is not None and (_root.find(_tag).text is not None or len(_root.find(_tag)) > 0)
            for _tag in necessary)

    if isinstance(voc, str):  # voc is a file path
        try:
            tree = etree.parse(voc)
        except etree.XMLSyntaxError:
            return False
        root = tree.getroot()
    else:  # voc is a parsed tree
        root = voc.getroot()

    # Check if 'annotation' is the root tag
    if root.tag != 'annotation':
        return False

    # Check if necessary tags exist
    if not check_subtags(root, ['filename', 'size']):
        return False

    if not is_image(root.find('filename').text):
        return False

    # Check 'size' tag
    size = root.find('size')
    if not check_subtags(size, ['width', 'height', 'depth']):

        return False
    for tag in ['width', 'height', 'depth']:
        if not check_int(size.find(tag).text):
            return False

    width = int(size.find('width').text)
    height = int(size.find('height').text)

    if not root.findall('object'):
        return True  # Backgrounds

    # Check 'object' tag
    objects = root.findall('object')
    for obj in objects:
        if not check_subtags(obj, ['name', 'bndbox']):

            return False
        bbox = obj.find('bndbox')
        if not check_subtags(bbox, ['xmin', 'xmax', 'ymin', 'ymax']):
            return False

        for msr in ['xmin', 'xmax', 'ymin', 'ymax']:
            if not check_int(bbox.find(msr).text):
                return False

        x_min = int(bbox.find('xmin').text)
        x_max = int(bbox.find('xmax').text)
        y_min = int(bbox.find('ymin').text)
        y_max = int(bbox.find('ymax').text)

        if not all((0 <= x_min < x_max <= width + 1, 0 <= y_min < y_max <= height + 1)):
            return False

    return True


def check_yolo(yolo: str | list, labelmap: dict = None) -> bool:
    """
    Check yolo annotation for corruption

    :param yolo: File path to YOLO formatted annotations or list of YOLO annotation instances.
    :type yolo: str or list

    :param labelmap: (Optional) Dictionary mapping class indices to class names.
    :type labelmap: dict, optional

    :return: True if the annotations are valid, False otherwise.
    :rtype: bool
    """
    if type(yolo) is str:  # yolo is a file path
        with open(yolo, 'r') as file:
            lines = file.readlines()
        instances = [line.strip().split() for line in lines]
    else:  # yolo is instances
        instances = yolo

    for instance in instances:

        # Check if there are 5 elements per line (class, x_center, y_center, width, height)
        if len(instance) != 5:
            return False

        if labelmap:
            if str(instance[0]) not in labelmap.keys():
                return False
        elif not check_int(instance[0]):
            return False

        # Check if all elements can be converted to float (necessary for bounding box parameters)
        for element in instance[1:]:
            if not check_float(element):
                return False
            if float(element) > 1 or float(element) < 0:
                return False

    # If all checks pass
    return True


def check_yolo_empty(yolo_path: str) -> bool:
    """
    Check if yolo file is empty

    :param yolo_path: Path to the file that needs to be checked.
    :return: True if the file is empty, False otherwise.
    """
    with open(yolo_path, 'r') as file:
        lines = file.readlines()
        return len(lines) == 0


def check_voc_empty(voc_path: str):
    """
    Check if VOC XML file is empty

    :param voc_path: Path to the VOC XML file that needs to be checked.
    :return: Boolean value, True if the VOC file has no 'object' elements; otherwise, False.
    """
    tree = etree.parse(voc_path)
    root = tree.getroot()

    if not root.findall('object'):
        return True
    return False
