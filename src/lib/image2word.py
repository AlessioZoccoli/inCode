import cv2
import numpy as np

from src.utils import utils as utils


def char2position(imgPath, charColors):
    image = cv2.imread(imgPath)

    mask = utils.mask_by_colors(image, charColors)
    # mask3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("images", np.hstack([image, mask3ch]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # centroids + bboxes area
    charpos = utils.centroids_bbxes_areas(mask)
    # clustering single characters written separately
    maxArea = max(charpos, key=lambda x: x[1])[1]
    return list(filter(lambda el: el[1]/maxArea >= 0.9, charpos))


def chars2position(imgPath, char2colors):
    ch2pos = []
    for ch, colors in char2colors.items():
        # colors(RGB) -> colors(GBR)
        colorsGBR = np.flip(np.array(colors, dtype=np.uint8), 1)
        bboxesStats = char2position(imgPath, colorsGBR)
        if ch == 'semicolon':
            bboxesStats = [bboxesStats[0]]
        ch2pos.append([(ch, coord) for coord in bboxesStats])

    return sorted([y for x in ch2pos for y in x], key=lambda el: el[1][0])
