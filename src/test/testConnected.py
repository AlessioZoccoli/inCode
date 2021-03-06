from collections import defaultdict
from json import load
from os import path
from pprint import pprint
from cv2 import imshow, imread, waitKey, destroyAllWindows
from config import annotationsRichJSON, votesJSON, color_words
from src.lib.image2word import positions2chars


def testConnected():
    """
    Displays each color annotation inside an image, useful to verify connections between components
    :return: None
    """
    with open(annotationsRichJSON, 'r') as a:
        annotations = load(a)

    with open(votesJSON, 'r') as v:
        votes = load(v)

    # output
    img2chars = defaultdict(list)
    """
    images = ['040v/274_1193_33_54.png', '040v/1053_636_41_104.png', '040v/159_585_41_63.png',
              '040v/1243_1517_28_65.png', '040v/1240_309_46_98.png', '055r/442_1503_41_106.png',
              '040v/401_532_46_140.png', '040v/145_1688_48_122.png', '040v/1177_1187_37_68.png',
              '040v/1194_1139_34_65.png', '046r/120_1563_35_126.png', '046r/366_404_47_125.png',
              '046r/590_843_31_162.png', '040v/158_205_32_105.png', '046r/105_679_35_90.png',
              '056r_178_258_1393_1827/756_805_42_43.png', '056r_178_258_1393_1827/768_1024_47_181.png',
              '055v_631_241_1360_1839/290_1660_34_72.png', '060v/131_715_32_86.png']
    """

    # images = ['055v_631_241_1360_1839/290_1660_34_72.png'] three pieces 'm'
    images = ["056r_178_258_1393_1827/550_1196_38_73.png"]

    for myImage in images:
        imgPath = path.join(color_words, myImage)
        result = positions2chars(imgPath, annotations[myImage], votes[myImage], show=True)
        # pprint(annotations[myImage])
        img2chars[myImage] = result
    # pprint(img2chars)


if __name__ == '__main__':
    testConnected()
