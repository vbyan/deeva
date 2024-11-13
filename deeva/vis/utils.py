import io
import math
import colorsys

import cv2
import numpy as np
import pandas as pd
import plotly
from PIL import Image


def _crop_solid_edges(im_arr: np.ndarray) -> np.ndarray:
    """
    Remove solid edges from an image

    :param im_arr: Input image array.
    :return: Cropped image array with solid edges removed.
    """
    gray = cv2.cvtColor(im_arr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = im_arr[y:y + h, x:x + w]
        return crop
    else:
        # Return the original image if no contours are found
        return im_arr

def _make_symmetrical(im_arr: np.ndarray) ->  np.ndarray:
    """
    Make input image symmetrical

    :param im_arr: Input numpy array representing an image.
    :return: Symmetrical image as a numpy array, concatenated with its flipped version.
    """
    im_flip = cv2.flip(im_arr, 1)
    return np.concatenate((im_arr, im_flip), axis=1)

def get_planet_texture(im_arr: np.ndarray) -> np.ndarray:
    """
    Create a planet texture using input image

    :param im_arr: The input image array representing the planet's surface.
    :return: A square image array with symmetrical texture and cropped solid edges.
    """
    cropped = _crop_solid_edges(im_arr)
    symmetrical = _make_symmetrical(cropped)

    h, w = im_arr.shape[:2]
    square = cv2.resize(symmetrical, (h, h))

    return square

def generate_proportional_centroids(n: int) -> np.ndarray:
    """
    Return uniformly distributed n centroids

    :param n: Number of centroids to generate.
    :return: A numpy array containing the coordinates of the centroids.
    """
    if n == 1:
        return np.array([[0.5, 0.5]])  # Only one centroid at the center of the range [0, 1]

    # Generate (n-1)-sided polygon vertices with radius 0.5 and center at (0.5, 0.5)
    vertices = []
    angles = np.linspace(0, 2 * np.pi , n - 1, endpoint=False)
    for angle in angles:
        x = 0.5 + 0.5 * np.cos(angle)
        y = 0.5 + 0.5 * np.sin(angle)
        vertices.append((x, y))

    # Add the center
    vertices.append((0.5, 0.5))

    return np.array(vertices)


def calculate_circles_distance(r1: float, r2: float, a: float) -> float:
    """
    Calculate the distance between centers of two circles given their radii and overlap area.
    Uses numerical method to find the distance that gives the desired overlap area.

    Parameters:
    r1 (float): Radius of the first circle (larger one)
    r2 (float): Radius of the second circle (smaller one)
    a (float): Area of the overlapping region

    Returns:
    float: Distance between the centers of the circles
    """
    # Ensure r1 is the larger radius
    if r1 < r2:
        r1, r2 = r2, r1

    # Check edge cases
    if a == 0:
        return r1 + r2

    # Maximum possible overlap area is the area of the smaller circle
    max_overlap = math.pi * r2 ** 2
    if abs(a - max_overlap) < 1e-10:
        return 0 if r1 == r2 else r1 - r2

    def calculate_overlap(d):
        """Calculate the overlap area given the distance between centers."""
        if d >= r1 + r2:  # No intersection
            return 0
        if d <= abs(r1 - r2):  # One circle is contained within the other
            return math.pi * min(r1, r2) ** 2

        # Calculate angles using the law of cosines
        cos_alpha = (d * d + r1 * r1 - r2 * r2) / (2 * d * r1)
        cos_beta = (d * d + r2 * r2 - r1 * r1) / (2 * d * r2)

        # Handle floating point errors
        cos_alpha = min(1, max(-1, cos_alpha))
        cos_beta = min(1, max(-1, cos_beta))

        alpha = math.acos(cos_alpha)
        beta = math.acos(cos_beta)

        # Calculate overlap area using the formula
        area = (r1 * r1 * alpha + r2 * r2 * beta -
                d * r1 * math.sin(alpha))

        return area

    # Use binary search to find the distance that gives the desired overlap area
    left = abs(r1 - r2)  # Minimum possible distance
    right = r1 + r2  # Maximum possible distance

    for _ in range(50):  # Usually converges in fewer iterations
        mid = (left + right) / 2
        current_area = calculate_overlap(mid)

        if abs(current_area - a) < 1e-10:
            return mid
        elif current_area > a:
            left = mid
        else:
            right = mid

    return (left + right) / 2


def get_rect_vertices(img: np.ndarray, x: float, y: float, w: float, h: float) -> tuple:
    """
    Return bbox vertices coordinates

    :param img: Input image in the form of a NumPy array.
    :param x: x coordinate of the top-left corner of the bounding box as a relative value (0 to 1).
    :param y: y coordinate of the top-left corner of the bounding box as a relative value (0 to 1).
    :param w: Width of the bounding box as a relative value (0 to 1).
    :param h: Height of the bounding box as a relative value (0 to 1).
    :return: Two lists, x_rect and y_rect, containing the x and y coordinates of the vertices of the rectangle.
    """
    img_h, img_w = img.shape[:2]
    x_min, y_min, x_max, y_max = xywh2xyxy(x, y, w, h)

    x_rect = [x_min, x_max, x_max, x_min, x_min]
    y_rect = [y_max, y_max, y_min, y_min, y_max]

    x_rect = [v * img_w for v in x_rect]
    y_rect = [v * img_h for v in y_rect]

    return x_rect, y_rect


def _adjust_annotations(x: float, y: float, w: float, h: float,
                        aggregate_x: float, aggregate_y: float,
                        aggregate_w: float, aggregate_h: float) -> tuple:
    """
    Adjust annotations using aggregate box on a cropped image

    :param x: X-coordinate of the annotation.
    :param y: Y-coordinate of the annotation.
    :param w: Width of the annotation.
    :param h: Height of the annotation.
    :param aggregate_x: X-coordinate of the aggregate area.
    :param aggregate_y: Y-coordinate of the aggregate area.
    :param aggregate_w: Width of the aggregate area.
    :param aggregate_h: Height of the aggregate area.
    :return: Adjusted annotation coordinates and dimensions as a tuple (x_new, y_new, w_new, h_new).
    """
    x_new = 0.5 - (aggregate_x - x) / aggregate_w
    y_new = 0.5 - (aggregate_y - y) / aggregate_h

    w_new = w / aggregate_w
    h_new = h / aggregate_h

    return x_new, y_new, w_new, h_new



def _get_aggregate_box(x1, y1, w1, h1, x2, y2, w2, h2):
    """
    Return aggregate box of 2 bboxes

    :param x1: x-coordinate of the center of the first box
    :param y1: y-coordinate of the center of the first box
    :param w1: width of the first box
    :param h1: height of the first box
    :param x2: x-coordinate of the center of the second box
    :param y2: y-coordinate of the center of the second box
    :param w2: width of the second box
    :param h2: height of the second box
    :return: Aggregate box as a tuple (aggregate_x, aggregate_y, aggregate_w, aggregate_h)
    """
    left1, bottom1, right1, top1 = xywh2xyxy(x1, y1, w1, h1)
    left2, bottom2, right2, top2 = xywh2xyxy(x2, y2, w2, h2)

    aggregate_left = min(left1, left2)
    aggregate_right = max(right1, right2)
    aggregate_top = max(top1, top2)
    aggregate_bottom = min(bottom1, bottom2)

    aggregate_w = aggregate_right - aggregate_left
    aggregate_h = aggregate_top - aggregate_bottom
    aggregate_x = (aggregate_left + aggregate_right) / 2
    aggregate_y = (aggregate_bottom + aggregate_top) / 2

    return aggregate_x, aggregate_y, aggregate_w, aggregate_h


def crop_aggregate_box(img: np.ndarray, pair: pd.Series,
                       padding: float=0.1, square: bool=False) -> tuple:
    """
    Crop the aggregate bounding box for a pair of boxes

    :param img: The input image represented as a NumPy array.
    :param pair: A pd.DataFrame row containing bounding box annotations
    :param padding: A float representing the amount of padding to be added around the aggregate bounding box
    :param square: A boolean indicating whether to force the bounding box to be square. Default is False.
    :return: A tuple containing the cropped image and the updated pair with adjusted bounding box annotations.
    """
    img_height, img_width = img.shape[:2]

    bbox1 = (pair.relate_x, pair.relate_y, pair.relate_w, pair.relate_h)
    bbox2 = (pair.with_x, pair.with_y, pair.with_w, pair.with_h)

    aggregate_x, aggregate_y, aggregate_w, aggregate_h = _get_aggregate_box(*bbox1, *bbox2)
    if square:
        aggregate_w = aggregate_h = max(aggregate_w, aggregate_h)

    padding_left = min((aggregate_x - aggregate_w/2), padding/2)
    padding_right = min(1 - (aggregate_x + aggregate_w/2), padding/2)
    padding_bottom = min((aggregate_y - aggregate_h/2), padding/2)
    padding_top = min(1 - (aggregate_y + aggregate_h/2), padding/2)

    padding_w = padding_left + padding_right
    padding_h = padding_bottom + padding_top

    # Convert normalized coordinates to pixel coordinates
    left = int((aggregate_x - aggregate_w/2 - padding_left) * img_width)
    right = int((aggregate_x + aggregate_w/2 + padding_right) * img_width)
    bottom = int((aggregate_y - aggregate_h/2 - padding_bottom) * img_height)
    top = int((aggregate_y + aggregate_h/2 + padding_top) * img_height)

    # Perform the crop
    cropped_img = img[bottom:top, left:right]

    aggregates = (aggregate_x + (padding_right - padding_left)/2, aggregate_y + (padding_top - padding_bottom)/2,
                  aggregate_w + padding_w, aggregate_h + padding_h)

    pair.loc[['relate_x', 'relate_y', 'relate_w', 'relate_h']] = _adjust_annotations(*bbox1, *aggregates)
    pair.loc[['with_x', 'with_y', 'with_w', 'with_h']] = _adjust_annotations(*bbox2, *aggregates)

    return cropped_img, pair

def xywh2xyxy(x: float, y: float, w: float, h: float) -> tuple:
    """
    Convert xywh to xyxy

    :param x: The x-coordinate of the center of the box
    :param y: The y-coordinate of the center of the box
    :param w: The width of the box
    :param h: The height of the box
    :return: A tuple containing the coordinates of the top-left and bottom-right corners (x_min, y_min, x_max, y_max)
    """
    x_min = x - w / 2
    x_max = x + w / 2
    y_min = y - h / 2
    y_max = y + h / 2
    return x_min, y_min, x_max, y_max


def hex_to_rgba(hex_color: str, alpha: float, factor: float=1.0) ->  str:
    """
    Convert hex codes to rgba value

    :param hex_color: Hexadecimal color string (e.g., "#RRGGBB").
    :param alpha: Alpha value for the RGBA color (0.0 to 1.0).
    :param factor: Multiplication factor for the RGB values (default is 1).
    :return: RGBA color string in the format 'rgba(R, G, B, A)'.
    """
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) * factor for i in (0, 2, 4))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

def hsv_to_hex(h: float, s: float, v: float) -> str:
    """
    Convert HSV color values to hexadecimal color string

    :param h: Hue value (0.0 to 1.0).
    :param s: Saturation value (0.0 to 1.0).
    :param v: Value/Brightness value (0.0 to 1.0).
    :return: Hexadecimal color string in the format '#RRGGBB'.
    """
    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    # Convert RGB to HEX
    hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

    return hex_color

def plotly_fig2array(fig) -> np.ndarray:
    """
    Convert plotly figure into an image array

    :param fig: Input Plotly figure to be converted to a NumPy array.
    :type fig: plotly.graph_objs._figure.Figure
    :return: Image represented as a NumPy array.
    :rtype: numpy.ndarray
    """
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


class PlotFont:
    """A class for defining font properties for plot text styling."""

    def __init__(self, family, size, style, weight, color):
        self.family = family
        self.size = size
        self.style = style
        self.weight = weight
        self.color = color

    @property
    def font(self):
        """Return a dictionary of the font properties."""
        return dict(family=self.family,
                    size=self.size,
                    style=self.style,
                    weight=self.weight,
                    color=self.color)
