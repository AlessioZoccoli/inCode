from collections import defaultdict
from json import load
import numpy as np
import os.path
import src.utils.utils as utils
import cv2


def char2position(imgPath, charColors):
    image = cv2.imread(imgPath)
    cv2.imshow('image', image)

    mask = utils.mask_by_colors(image, charColors)
    mask3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow("images", np.hstack([image, mask3ch]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # [x, xBBoxEndings]
    return sorted([el[0] for el in utils.sorted_bbxs(mask)])


def chars2position(imgPath, char2colors):
    ch2pos = []
    for ch, colors in char2colors.items():
        # colors(RGB) -> colors(GBR)
        colorsGBR = np.flip(np.array(colors, dtype=np.uint8), 1)
        bboxesStats = char2position(imgPath, colorsGBR)
        ch2pos.append([(ch, coord) for coord in bboxesStats])

    return sorted([y for x in ch2pos for y in x], key=lambda el: el[1]) #sorted(ch2pos, key=lambda el: el[1])


if __name__ == '__main__':
    dataPath = os.path.join(os.getcwd(), '../../data/')
    jsonPath = os.path.join(dataPath, 'anncolor_by_word.json')
    print('\npath json exists: ', os.path.exists(jsonPath))

    with open(jsonPath, 'r') as f:
        anncolor = load(f)

    images = ['040v/1270_158_38_61.png']#['040v/1340_1687_13_12.png', '040v/401_532_46_140.png', '040v/408_309_42_71.png', '040v/1253_804_34_62.png']
    for i in images:
        imgPath = os.path.join(dataPath, i)
        #print('anncolor[i] ', anncolor[i])
        print(chars2position(imgPath, anncolor[i]))
