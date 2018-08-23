from os import path

from numpy import flip
from numpy.ma import concatenate
from config import color_words, images2ColorsBBxesJSON
from json import load
from cv2 import imshow, imread, waitKey, destroyAllWindows

from src.utils.imageProcessing import cropByColor, background

if __name__ == '__main__':


    with open(images2ColorsBBxesJSON, 'r') as ann:
        ch2col = load(ann)


    # 'onta': ['053r/404_209_44_156.png', ['o', 'n', 't', 'a']],
    imgName = '053r/404_209_44_156.png'
    imagePath = path.join(color_words, imgName)
    img = imread(imagePath)

    onta_RGBcolors = concatenate([ch2col[imgName]['col'][t] for t in ['o', 'n', 't', 'a']], axis=0)
    onta_BGRcolors = flip(onta_RGBcolors, axis=1)

    onta = cropByColor(img, onta_BGRcolors)
    # ontaImg = imshow('onta', onta)

    back = background()
    back[0:onta.shape[0], 0:onta.shape[1]] = onta

    imshow('maskWithBackground', back)

    waitKey(0)
    destroyAllWindows()
