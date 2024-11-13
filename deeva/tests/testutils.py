import os
from lxml import etree


def element_to_string(element):
    string = etree.tostring(element.getroot(), method='xml')
    string = string.replace(b"\n", b"").replace(b"\t", b"")
    return string


def listdir_absolute(directory):
    return sorted([os.path.join(directory, file) for file in os.listdir(directory)])
