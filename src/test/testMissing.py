from json import load
from os import path, getcwd
import cv2
import numpy as np
from src.utils.imageProcessing import mask_by_colors, getMissingElements
from config import *

if __name__ == '__main__':
    # dataPath = path.join(getcwd(), '../../../data/')
    # imageName = '048r/86_134_36_166.png' # '056r_178_258_1393_1827/768_1024_47_181.png'

    i = '040v/1015_1355_45_139.png'
    imgPath = path.join(color_words, i)
    image = cv2.imread(imgPath)

    with open(annotationsJSON, 'r') as ann:
        annot = load(ann)

    missings = getMissingElements(image, list(annot[i].values()))
    # RGB -> BGR
    mask = mask_by_colors(image, np.flip(missings['colors'], 1))

    # pprint(missings['colors'])
    # pprint(missings['centroids_area'])
    # mask = mask_by_colors(image, colors)

    mask3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow("images", np.hstack([image, mask3ch]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()