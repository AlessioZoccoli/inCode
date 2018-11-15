from json import load
from os import path, getcwd
from pprint import pprint

import cv2
import numpy as np
from src.utils.imageProcessing import maskByColors, getMissingElements
from config import *


def displayMissing():

    imgs = ['040v/1015_1355_45_139.png', '048r/86_134_36_166.png', '056r_178_258_1393_1827/768_1024_47_181.png']

    with open(annotationsJSON, 'r') as ann:
        annot = load(ann)

    for i in imgs:
        imgPath = path.join(color_words, i)
        image = cv2.imread(imgPath)

        missings = getMissingElements(image, list(annot[i].values()))
        mask = maskByColors(image, missings['colors'])

        print('\n\n####  COLORS ####')
        pprint(missings['colors'])
        print('\n####  CENTROIDS ####')
        pprint(missings['centroids_area'])

        mask3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow("images", np.hstack([image, mask3ch]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    displayMissing()
