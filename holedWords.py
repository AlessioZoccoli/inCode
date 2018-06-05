from collections import defaultdict
from json import load, dump
from os import path, getcwd
import numpy as np
from src.utils.utils import getMissingElements
from cv2 import imread

if __name__ == '__main__':

    imagesPath = path.join(getcwd(), '../color_words/')
    dataPath = path.join(getcwd(), 'data/')
    # imageName = '048r/86_134_36_166.png' # '056r_178_258_1393_1827/768_1024_47_181.png'

    with open(path.join(getcwd(), 'data/anncolor_by_word.json'), 'r') as f:
        annotatJSON = load(f)

    # np digits to python
    toScalar = (lambda x: (np.asscalar(x[0]), np.asscalar(x[1]), np.asscalar(x[2])))
    missings = defaultdict(dict)

    for imageName, val in annotatJSON.items():
        _image = imread(path.join(imagesPath, imageName))
        # RGB
        _colors = list(val.values())
        # numpy image, GBR -> BGR, (xCentroid, yCentroid, area)
        try:
            charInfo = getMissingElements(_image, _colors)
            if charInfo['colors'] != [] and charInfo['centroids_area'] != []:
                charInfo['centroids_area'] = list(map(lambda coord: toScalar(coord), charInfo['centroids_area']))
                charInfo['colors'] = charInfo['colors'].tolist()
                missings[imageName].update(charInfo)
        except ValueError:
            print(_colors)

    with open(path.join(dataPath, 'missings.json'), 'w') as fOut:
        dump(missings, fOut, indent=4, sort_keys=True)
