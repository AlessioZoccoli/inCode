from json import load, dump
from os import path
from pprint import pprint

import cv2
import numpy as np

from config import *
from src.lib.image2word import getConnectedComponents
from src.utils.imageProcessing import maskByColors
from collections import defaultdict


def connectedComponents():
    """
    INPUT:
        RelativePath/imageName.png
    OUTPUT:
        dict of lists. Key=<input>, value=[<full word>, [<connected components>]].
        <connected components> are lists, one for each cc, containing charachters which make it up.


    In[]: getConnectedComponents('056r_178_258_1393_1827/768_1024_47_181.png', annotations, bwmask)
    {'056r_178_258_1393_1827/768_1024_47_181.png': [
                ['p', 'a', 'r', 'u', 'p', 'n', 'd', 'e'],
                [['p', 'a', 'r'],  ['u', 'p'],  ['n'],  ['d'],  ['e']]
                ]
    }

    """

    with open(wordsRichDoublesAndUppercase, 'r') as w, open(annotationsRichJSON, 'r') as a:
        words = load(w)
        annoted = load(a)

    result = defaultdict(dict)
    notKeys = []

    for im, ch2col in annoted.items():
        try:
            imagePath = path.join(color_words, im)
            colorsBGR = [np.flip(np.array(color, dtype=np.uint8), 0) for subl in ch2col.values() for color in subl]
            image = cv2.imread(imagePath)
            mask = maskByColors(image, colorsBGR)
            result.update(getConnectedComponents(im, words[im], mask))
        except KeyError:
            notKeys.append(im)

    print('NOT IN {}!    size: {}'.format(annotationsCleanJSON, len(notKeys)))
    pprint(notKeys)

    with open(connCompsRichJSON, 'w') as outFile:
        dump(result, outFile, indent=4, sort_keys=True)


if __name__ == '__main__':
    connectedComponents()
