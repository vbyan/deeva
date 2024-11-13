LABELMAP = {'0': 'first',
            '1': 'second'}

PARAMS_PARSE_YOLO = [('tests/assets/yolo/labels/test1.txt', LABELMAP,
                      [['first', 0.5, 0.5, 0.5, 0.5]]),
                     ('tests/assets/yolo/labels/test2.txt', LABELMAP,
                      [['first', 0.2, 0.1, 0.4, 0.2],
                       ['second', 0.2, 0.2, 0.2, 0.2]]),
                     ('tests/assets/yolo/labels/test3.txt', LABELMAP, []),
                     ('tests/assets/yolo/labels/test1.txt', None,
                      [[0, 0.5, 0.5, 0.5, 0.5]]
                      ),
                     ('tests/assets/yolo/labels/test2.txt', None,
                      [[0, 0.2, 0.1, 0.4, 0.2],
                       [1, 0.2, 0.2, 0.2, 0.2]]
                      ),
                     ('tests/assets/yolo/labels/test3.txt', None, [])]


PARAMS_GET_LABELMAP = [('tests/assets/labelmaps/correct',
                        (LABELMAP, None)
                        ),
                       ('tests/assets/labelmaps/missing',
                        (None, 'Labelmap not found')
                        ),
                       ('tests/assets/labelmaps/wrong',
                        (None, 'Incorrect labelmap'))]

PARAMS_CHECK_CORRECT = ['tests/assets/random_samples/voc/correct/test1.xml',
                        'tests/assets/random_samples/voc/correct/test2.xml',
                        'tests/assets/random_samples/voc/correct/test3.xml',
                        'tests/assets/random_samples/yolo/correct/test1.txt',
                        'tests/assets/random_samples/yolo/correct/test2.txt',
                        'tests/assets/random_samples/yolo/correct/test3.txt']

PARAMS_CHECK_CORRUPTED = ['tests/assets/random_samples/voc/corrupted/test1.xml',
                          'tests/assets/random_samples/voc/corrupted/test2.xml',
                          'tests/assets/random_samples/voc/corrupted/test3.xml',
                          'tests/assets/random_samples/voc/corrupted/test4.xml',
                          'tests/assets/random_samples/voc/corrupted/test5.xml',
                          'tests/assets/random_samples/voc/corrupted/test5.xml',
                          'tests/assets/random_samples/voc/corrupted/test6.xml',
                          'tests/assets/random_samples/voc/corrupted/test7.xml',
                          'tests/assets/random_samples/voc/corrupted/test8.xml',
                          'tests/assets/random_samples/voc/corrupted/test9.xml',
                          'tests/assets/random_samples/voc/corrupted/test10.xml',
                          'tests/assets/random_samples/yolo/corrupted/test1.txt',
                          'tests/assets/random_samples/yolo/corrupted/test2.txt',
                          'tests/assets/random_samples/yolo/corrupted/test3.txt',
                          'tests/assets/random_samples/yolo/corrupted/test4.txt',
                          'tests/assets/random_samples/yolo/corrupted/test5.txt']

PARAM_CHECK_EMPTY_YOLO = [(('foo.jpg', '/home/Desktop'), ([], '/home/Desktop/labels/foo.txt')),
                          (('foo.png', '/home/Desktop/foo'), ([], '/home/Desktop/foo/labels/foo.txt'))]

PARAM_CHECK_EMPTY_VOC = [(('test3.jpg', 'tests/assets/voc'), (
    b'<annotation><folder>images</folder><filename>test3.jpg</filename><size><width>1000</width><height>500</height><depth>3</depth></size></annotation>',
    'tests/assets/voc/labels/test3.xml'))]
