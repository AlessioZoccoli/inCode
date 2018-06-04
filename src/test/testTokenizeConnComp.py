from json import load, dump
from os import path, getcwd
import cv2
import numpy as np

from src.lib.image2word import getConnectedComponents
from src.utils.utils import mask_by_colors, bbxesCoverage
from pprint import pprint
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
                ['p', 'a', 'r', 'i_bis', 'u', 'i_bis', 'p', 'n', 'd', 'e'],
                [['p', 'a', 'r', 'i_bis'],  ['u', 'i_bis', 'p'],  ['n'],  ['d'],  ['e']]
                ]
    }
    
    """

    imagesPath = path.join(getcwd(), '../../../color_words/')
    dataPath = path.join(getcwd(), '../../data/')

    with open(path.join(dataPath, 'words.json'), 'r') as w, open(path.join(dataPath, 'anncolor_by_word.json'),'r') as a:
        words = load(w)
        annoted = load(a)


    myImage = '040v/401_532_46_140.png' # '056r_178_258_1393_1827/768_1024_47_181.png'
    myImagePath = path.join(imagesPath, myImage)

    colors = [np.flip(np.array(color, dtype=np.uint8), 0) for subl in annoted[myImage].values() for color in subl]
    image = cv2.imread(myImagePath)
    mask = mask_by_colors(image, colors)

    res = getConnectedComponents(myImage, words[myImage], mask)
    pprint(res)