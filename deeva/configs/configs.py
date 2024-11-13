import mimetypes
import os

from screeninfo import get_monitors

from .utils import *


# setting working dir for entire project
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# main st.session_state vars
PARAMS_DEFAULT = dict(
    data_path=None,
    files_pref=None,
    remove_no_ext=False,
    remove_wf=False,
    remove_dp=False,
    remove_lnl=False,
    toggle_no_ext=False,
    toggle_wf=False,
    toggle_dp=False,
    toggle_lnl=False,
    toggle_mbg=False,
    render_final=False,
    choose_fp=None,
    remove_mbg=False,
    pass_forward=None,
    labelmap=None,
    scarce_threshold_backup=None,
    threshold_type_backup=None,
    max_classes_backup=None,
    show_all_checkbox_backup=None,
    stats_selectbox_backup=None,
    annotations_class_selectbox_backup=None,
    plot_type_selectbox_backup=None,
    images_class_selectbox_backup=None,
    images_include_selectbox_backup=None,
    medium_low_backup=None,
    medium_high_backup=None,
    box_size_slider_backup=None,
    convert_final=False,
    overlaps_inpute_page=True,
    overlaps_n_cases_backup=3,
    overlaps_n_images_backup=3,
    overlaps_object1_selectbox_backup=None,
    overlaps_object2_selectbox_backup=None,
    reset_toys=False,
    processing=False,
    images_cancel=False,
    overlaps_cancel=False,
    images_cache_key=0,
    overlaps_cache_key=0
)

PARAMS_DEFAULT_ANNOTATIONS = dict(
    medium_low=1,
    medium_high=2,
    box_size_slider=(0.0, 100.0)
)

# PARAMS_DEFAULT_OVERLAPS =

CATEGORIES = ['images', 'labels']
DATAMATCH_ATTRIBUTES_SHORT = ['no_ext', 'wrong_format', 'duplicates']
DATAMATCH_ATTRIBUTES_SHORT_REMOVE = ['remove_no_ext', 'remove_wf', 'remove_dp']
DATAMATCH_ATTRIBUTES_SHORT_TOGGLE = ['toggle_no_ext', 'toggle_wf', 'toggle_dp']
DATAMATCH_ATTRIBUTES = ['No extension', 'Wrong format', 'Duplicates by filename']


# set of all common extensions for all types of files
COMMON_EXTENSIONS = set()
for ext, _ in mimetypes.types_map.items():
    COMMON_EXTENSIONS.add(ext.split('.')[-1].lower())


monitor = get_monitors()[0]
vw = ViewPort(size=monitor.width)
vh = ViewPort(size=monitor.height)


# allowed extensions
IMAGE_EXTENSIONS = ('png', 'jpg', 'jpeg')
LABEL_EXTENSIONS = ('txt', 'xml')

# adding variables to PARAMS_DEFAULT for data caching
PARAMS_DEFAULT = add_caching_vars('images', PARAMS_DEFAULT)
PARAMS_DEFAULT = add_caching_vars('overlaps', PARAMS_DEFAULT)

# all backup vars from PARAMS_DEFAULT
BACKUPS = {k: v for k, v in PARAMS_DEFAULT.items() if k.endswith('_backup')}

# aliases to search for a labelmap more efficiently
LABELMAP_ALIASES = ['labelmap.txt', 'classes.txt', 'mapping.txt', 'label_classes.txt', 'data.yaml']

# max length of a class name
CLASS_NAME_LENGTH = 10

# assigned to classes
pastel_generator = ColorGenerator(saturation=0.75, lightness=0.45)
CLASS_COLORS = pastel_generator.generate_hex_colors(1000)

# hue values separated into ranges
HUE_RANGES = [
    ("mono", 0, 1),
    ("red", 1, 15),
    ("yellow", 15, 30),
    ("green", 30, 75),
    ("cyan", 75, 95),
    ("blue", 95, 140),
    ("pink", 140, 165),
    ("red", 165, 180)
]

HUE_COLORS = {k: v for k, v in zip(sorted((color for (color, _, _) in HUE_RANGES)), range(8))}
HUE_COLORS_REV = {k: v for v, k in HUE_COLORS.items()}
HUE_COLORS_MEAN = {k: (range_start + range_end) * .5 for (k, range_start, range_end) in HUE_RANGES}
HUE_COLORS_MEAN["red"] = 180

# Create a lookup table
HUE_RANGES_LOOKUP = np.empty(181, dtype=object)
for (color, min_hue, max_hue) in HUE_RANGES:
    HUE_RANGES_LOOKUP[min_hue:max_hue + 1] = HUE_COLORS[color]

COMMON_RESOLUTIONS = {
    '+': dict(w=8000, h=6000), # more than 8K
    '8k': dict(w=7680, h=4320), # 8k SHD
    '4K': dict(w=3840, h=2160),  # 4K UHD
    'QHD': dict(w=2560, h=1440),  # QHD / 1440p
    'Full HD': dict(w=1920, h=1080),  # 1080p
    'HD': dict(w=1280, h=720),  # 720p
    '480p': dict(w=854, h=480),  # SD (480p)
    '360p': dict(w=480, h=360),  # Low resolution
}

# n instances needed to collect stats
ANNOTATIONS_LIMIT = 5
IMAGES_LIMIT = 10
OVERLAPS_LIMIT = 10

# pd.DataFrame column names for different stats
IMAGE_STATS_COLOR_COLUMNS = ['count', 'hue', 'sat', 'value']
IMAGE_STATS_OVERALL_COLUMNS = ['filepath', 'height', 'width', 'mode',
                               'RMS_contrast', 'brightness']
ANNOTATION_STATS_COLUMNS = ['class_name', 'x_center',
                            'y_center', 'width', 'height']
OVERLAP_STATS_COLUMNS = ['filename', 'relate', 'relate_with', 'relate_index',
                         'with_index', 'relate_size', 'with_size', 'overlap_size',
                         'relate_x', 'relate_y', 'relate_w', 'relate_h', 'with_x',
                         'with_y', 'with_w', 'with_h']

# base64 Data URIs of corresponding images
URI = {'main_background': get_image_uri('assets/backgrounds/main.png'),
       'astronaut': get_image_uri('assets/other/astronaut.png'),
       'app': get_image_uri('assets/other/app.png'),
       'disk': get_image_uri('assets/other/disk.png'),
       'bin': get_image_uri('assets/other/hole.png'),
       'blue_planet': get_image_uri('assets/planets/blue.png'),
       'green_planet': get_image_uri('assets/planets/green.jpg'),
       'moon_planet': get_image_uri('assets/planets/moon.jpg'),
       'purple_planet': get_image_uri('assets/planets/purple.png'),
       'red_planet': get_image_uri('assets/planets/red.png'),
       }

HELP_DATA_PATH = (
    "- **Data must be organized into two folders \"images\" and \"labels\"**\n"
    "- **Supports YOLO and PASCAL VOC formats for labels**\n"
    "- **Also provide labelmap for better experience**"
)

HELP_MARK_BACKGROUNDS = "**If rendered will create empty labels for all lonely images**"

HELP_STRATEGY_SELECTBOX = """
                **Image Assigning Strategy**

                - **All**: All images having at least one instance of a specific class will be treated as images of that class.
                - **Most frequent**: If the specific class is the most frequent class in an image, the image is treated as an image of that class.
                """

HELP_MAX_CLASSES = "**Applies only to Class counts and Co-occurrence matrix**"

HELP_SCARCE_THRESHOLD = "**Proportion derived by comparing each value to the maximum or total within its category**"

CAPTIONS_FILES_PREF = ["***Move to distinct folders*** 🗃️",
                       "***Delete from source*** :wastebasket:"]

CAPTIONS_CONVERT = ["**Convert all annotations to VOC format**",
                    "**Convert all annotations to YOLO format**"]

TOAST_NUMBER_INPUTS = "**Invalid thresholds. low > high**"
TOAST_CONVERT = "**Choose a convert option**"
TOAST_NOT_A_DIRECTORY = "**The data path you provided is not a directory**"
TOAST_INVALID_DIRECTORY = "**Invalid directory. Labels and images must be in corresponding folders**"
TOAST_MIXED_FORMATS = "**Mixed formats not handled correctly**"
TOAST_RESET_TOYS = "**Toys have been reset**"

CONFIRMATION_WRITE_SOURCE = "Are you sure you want to make changes to the source?"
CONFIRMATION_RESET_TOYS = "Reset toys? Cache and changes will be lost"
CONFIRMATION_LONELY_FILES = """\n\n**⚠️ Conversion Warning**
                               \n**{0} files can not be converted.**
                               These files are missing corresponding images."""



ROLL_CONFIGS = dict(x=[29,26,26,26,26],
                     y=[77,73,73,73,73],
                     radius=[-33.5, -45, -55, -65, -75],
                     size=[3.7, 4.7, 5.3, 6.6, 7.4],
                     cruise_time=[30, 60, 90, 120, 180],
                     rotation_time=[3, 5, 7, 10, 15],
                     key=['blue_planet', 'red_planet', 'moon_planet', 'green_planet', 'purple_planet'])

# supply.html path parameters
D1 = ("M 250,0 L 250,90 S 250,190 300.0,190 L 390,190", "M240,0 a-10,10 0 0,0 20,0")
D2 = ("M 160,0 L 160,220 S 160,320 115.0,320 L 10,320", "M150,0 a-10,10 0 0,0 20,0")
