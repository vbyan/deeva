import pytest
from loaders.parsers import *
from loaders.converters import *
from loaders.utils import *
from tests.testconfigs import *
from tests.testutils import element_to_string, listdir_absolute


YOLO_ANNOTATIONS = listdir_absolute('tests/assets/yolo/labels')
VOC_ANNOTATIONS = listdir_absolute('tests/assets/voc/labels')
IMAGE_PATHS = listdir_absolute('tests/assets/yolo/images')

PARAMS = [(yolo, voc, image, LABELMAP) for yolo, voc, image in
          zip(YOLO_ANNOTATIONS, VOC_ANNOTATIONS, IMAGE_PATHS)]


class TestConverters:
    @pytest.fixture(params=PARAMS)
    def set_params(self, request):
        yolo_annotation, voc_annotation, image_path, labelmap = request.param
        return yolo_annotation, voc_annotation, image_path, labelmap

    @pytest.mark.run(order=1)
    @pytest.mark.parametrize(argnames='data_path, expected',
                             argvalues=PARAMS_GET_LABELMAP)
    def test_get_labelmap(self, data_path, expected):
        labelmap = get_labelmap(data_path)
        assert labelmap == expected

    @pytest.mark.run(order=2)
    @pytest.mark.parametrize(argnames='file_path, labelmap, expected',
                             argvalues=PARAMS_PARSE_YOLO)
    def test_parse_yolo(self, file_path, labelmap, expected):
        result = parse_yolo(file_path, labelmap)
        assert result == expected

    @pytest.mark.run(order=3)
    def test_check_default(self, set_params):
        yolo_annotation, voc_annotation, image_path, labelmap = set_params
        yolo_correct = check_yolo(yolo_annotation)
        voc_correct = check_voc(voc_annotation)

        assert yolo_correct
        assert voc_correct

    @pytest.mark.run(order=3)
    @pytest.mark.parametrize('file_path', argvalues=PARAMS_CHECK_CORRECT)
    def test_check_correct(self, file_path):
        if 'yolo' in file_path:
            correct = check_yolo(file_path)
        else:
            correct = check_voc(file_path)

        assert correct

    @pytest.mark.run(order=3)
    @pytest.mark.parametrize('file_path', argvalues=PARAMS_CHECK_CORRUPTED)
    def test_check_corrupted(self, file_path):
        if 'yolo' in file_path:
            corrupted = not check_yolo(file_path)
        else:
            corrupted = not check_voc(file_path)

        assert corrupted

    def test_yolo_to_voc(self, set_params):
        yolo_annotation, voc_annotation, image_path, labelmap = set_params
        result, destination = yolo_to_voc(yolo_annotation, labelmap, image_path)
        expected = parse_voc(voc_annotation, return_tree=True)
        assert check_voc(result)
        assert element_to_string(result) == element_to_string(expected)
        assert destination == voc_annotation.replace('voc', 'yolo')

    def test_voc_to_yolo(self, set_params):
        yolo_annotation, voc_annotation, image_path, labelmap = set_params

        voc_to_yolo = VOC2YOLO(labelmap)
        instances, destination = voc_to_yolo(voc_annotation)
        expected = parse_yolo(yolo_annotation, labelmap=None)

        assert check_yolo(instances, labelmap)
        assert instances == expected
        assert destination == yolo_annotation.replace('yolo', 'voc')

    @pytest.mark.parametrize(argnames="test_input, expected", argvalues=PARAM_CHECK_EMPTY_YOLO)
    def test_empty_yolo(self, test_input, expected):
        image_filename, path = test_input
        yolo_empty, destination = empty_yolo(image_filename, path)

        expected_yolo_empty, expected_destination = expected

        assert yolo_empty == expected_yolo_empty
        assert destination == expected_destination

    @pytest.mark.parametrize(argnames="test_input, expected", argvalues=PARAM_CHECK_EMPTY_VOC)
    def test_empty_voc(self, test_input, expected):
        image_filename, path = test_input
        expected_voc_empty, expected_destination = expected

        voc_empty, destination = empty_voc(image_filename, path)

        assert element_to_string(voc_empty) == expected_voc_empty
        assert destination == expected_destination



