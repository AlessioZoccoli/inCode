from config import color_words, images2ConncompEColors
from numpy import flip, concatenate
from json import load
from cv2 import imshow, imread, waitKey, destroyAllWindows
from src.utils.imageProcessing import mask_by_colors, cropByColor
from os import path

if __name__ == '__main__':
    with open(images2ConncompEColors, 'r') as ann:
        ch2col = load(ann)

    eq_BGRcolors = flip(concatenate(
        (ch2col['040v/1001_696_43_114.png']['annot']['e'], ch2col['040v/1001_696_43_114.png']['annot']['q']),
        axis=0), axis=1)

    imagePath = path.join(color_words, '040v/1001_696_43_114.png')
    img = imread(imagePath)
    mask = cropByColor(img, eq_BGRcolors)  # mask_by_colors(img, eq_BGRcolors)

    imshow('imgMask', mask)
    waitKey(0)
    destroyAllWindows()
