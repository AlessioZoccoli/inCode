import cv2
import numpy as np
from src.utils import utils as utils
from itertools import combinations


def char2position(imgPath, charColors):
    """
    returns the centroid and area of the bbox corresponding to the mask of charColors
    :param imgPath: string. file (relative) path, eg. 040v/159_585_41_63.png
    :param charColors: list of [G, B, R] colors
    :return: list of tuples. Each tuple consists in (xCentroid, yCentroid, bboxArea).
            this list exclude possibile disconnected fragments.
    """
    image = cv2.imread(imgPath)

    mask = utils.mask_by_colors(image, charColors)
    # mask3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("images", np.hstack([image, mask3ch]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # centroids + bboxes area
    charpos = utils.centroids_bbxes_areas(mask)

    if len(charpos) > 1:
        curr = charpos[0]
        taken = [curr]
        for bb in charpos[1:]:
            _min, _max = sorted([curr, bb], key=lambda x: x[2])
            # comparing xCentroid and areas ratio
            if not(abs(_max[0] - _min[0]) < 13.0 and _min[2]/_max[2] < 0.9):
                taken.append(bb)
        charpos = taken
    return charpos


def positions2chars(imgPath, char2colors):
    """
    This method associates to each word-image(file path) a list of its characters with corresponding xCentroid, area
    and vote
    :param imgPath: file path to the image
    :param char2colors: list of lists.
        Each list is a doubleton, two elements only:
            - char
            - dict, containing colors and vote
    :return: list of tuples in the form of    ((xCentroid, yCentroid, Area), char)
    """
    pos2ch = []
    for ch, colors in char2colors.items():
        # colors(RGB) -> colors(GBR)
        colorsGBR = np.flip(np.array(colors, dtype=np.uint8), 1)
        bboxesStats = char2position(imgPath, colorsGBR)
        if ch == 'semicolon':
            bboxesStats = [bboxesStats[0]]
        pos2ch.append([(coord, ch) for coord in bboxesStats])

    return sorted([y for x in pos2ch for y in x])


def disambiguate(img, charsList, votedChars):
    """
    Calling this method assumes there are overlapping (same centroid) chars.

    Given two charachters having the same centroid we may have one written character recordered as two distinct ones,
    We want to keep the one with more votes
    :param img: image path
    :param charsList: list of tuples ((<xCentroid, yCentroid, area>),<char>)
    :return: same input list except overlapping chars with lower votes
    """

    doubles = set()

    for thisChar, thatChar in combinations(charsList, 2):
        # max min
        thisVote = min(votedChars[img][1][thisChar[1]])
        thatVote = min(votedChars[img][1][thatChar[1]])
        if thisChar[0] == thatChar[0]:
            if thisVote < thatVote:
                doubles.add(thatChar)
            elif thisVote > thatVote:
                doubles.add(thisChar)

    return list(filter(lambda ch: ch not in doubles, charsList))
