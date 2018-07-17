from json import load, dump
from os import path, getcwd
import cv2
import numpy as np

from src.lib.image2word import getConnectedComponents
from src.utils.utils import mask_by_colors
from collections import defaultdict

if __name__ == '__main__':
    """
    INPUT:
        RelativePath/imageName.png
    OUTPUT:
        dict of lists. Key=<input>, value=[<full word>, [<connected components>]].
        <connected components> are lists, one for each cc, containing charachters which make it up.


    >>> getConnectedComponents('056r_178_258_1393_1827/768_1024_47_181.png', annotations, bwmask)
    {'056r_178_258_1393_1827/768_1024_47_181.png': [
                ['p', 'a', 'r', 'i_bis', 'u', 'p', 'n', 'd', 'e'],
                [['p', 'a', 'r', 'i_bis'],  ['u', 'p'],  ['n'],  ['d'],  ['e']]
                ]
    }

    """

    imagesPath = path.join(getcwd(), '../../../color_words/')
    dataPath = path.join(getcwd(), '../../../data/')

    with open(path.join(dataPath, 'words_clean.json'), 'r') as w, open(path.join(dataPath, 'anncolor_by_word.json'), 'r') as a:
        words = load(w)
        annoted = load(a)

    # getConnectedComponents(myImage, words[myImage], mask))
    result = defaultdict(dict)

    for im, ch2col in annoted.items():
        imagePath = path.join(imagesPath, im)
        colors = [np.flip(np.array(color, dtype=np.uint8), 0) for subl in ch2col.values() for color in subl]
        image = cv2.imread(imagePath)
        mask = mask_by_colors(image, colors)

        # composition = getConnectedComponents(im, words[im], mask).values()
        result.update(getConnectedComponents(im, words[im], mask))

    with open(path.join(dataPath, 'connectedComps.json'), 'w') as outFile:
        dump(result, outFile, indent=4, sort_keys=True)
