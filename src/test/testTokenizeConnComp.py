from json import load
from os import path, getcwd
import cv2
import numpy as np
from config import *
from src.lib.image2word import getConnectedComponents, positions2chars
from src.utils.imageProcessing import maskByColors
from pprint import pprint


if __name__ == '__main__':

    """
    INPUT:
        RelativePath/imageName.png
    OUTPUT:
        dict of lists. Key=<input>, value=[<full word>, [<connected components>]].
        <connected components> are lists, one for each cc, containing charachters which make it up.
    
    
    >>> getConnectedComponents('056r_178_258_1393_1827/768_1024_47_181.png', annotations, bwmask)
    {'056r_178_258_1393_1827/768_1024_47_181.png': [
                ['p', 'a', 'r', 'u', 'p', 'n', 'd', 'e'],
                [['p', 'a', 'r'],  ['u', 'p'],  ['n'],  ['d'],  ['e']]
                ]
    }
    
    """

    with open(wordsDoublesAndUppercase, 'r') as wordsFile,\
            open(annotationsCleanJSON, 'r') as annotFile:
        words = load(wordsFile)
        annoted = load(annotFile)

    myImage = '058v/1072_314_46_167.png'  # '040v/401_532_46_140.png'
    myImagePath = path.join(color_words, myImage)


    colors = [np.flip(np.array(color, dtype=np.uint8), 0) for subl in annoted[myImage].values() for color in subl]
    image = cv2.imread(myImagePath)
    mask = maskByColors(image, colors)

    # pprint(words[myImage])
    print('\n\nconnected components:')
    res = getConnectedComponents(myImage, words[myImage], mask)
    pprint(res)
