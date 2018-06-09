from pprint import pprint

import cv2
import numpy as np
from src.utils import utils as utils
from itertools import combinations

from src.utils.utils import bbxesCoverage


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
    taken = set()
    omit = set()

    if len(charpos) > 1:
        for thisBB, thatBB in zip(charpos, charpos[1:]):
            # comparing areas
            _min, _max = sorted([thisBB, thatBB], key=lambda el: el[2])
            if abs(_max[0] - _min[0]) < 13.0 and _min[2]/_max[2] < 0.99:
                taken.add(_max)
                omit.add(_min)
            else:
                taken.add(_max)
                taken.add(_min)
            # print(abs(_max[0] - _min[0]) < 13.0, _min[2] / _max[2] < 0.99,'\n')
        charpos = sorted([el for el in taken-omit], key=lambda x: x[0])

    return charpos


def positions2chars(imgPath, char2colors):
    """
    This method ssociates to each word-image(file path) a list of its characters with corresponding xCentroid, area
    and vote
    :param imgPath: file path to the image
    :param char2colors: list of lists.
        Each list is a doubleton, two elements only:
            - char
            - colors
    :return: list of tuples in the form of    ((xCentroid, yCentroid, Area), char)
    """
    toScalar = (lambda el: (np.asscalar(el[0]), np.asscalar(el[1]), np.asscalar(el[2])))
    pos2ch = []
    for ch, colors in char2colors.items():
        # colors(RGB) -> colors(GBR)
        colorsGBR = np.flip(np.array(colors, dtype=np.uint8), 1)
        bboxesStats = char2position(imgPath, colorsGBR)
        if ch == 'semicolon':
            bboxesStats = [bboxesStats[0]]
        pos2ch.append([(toScalar(coord), ch) for coord in bboxesStats])

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
        # thisVote = min(votedChars[img][1][thisChar[1]])
        # thatVote = min(votedChars[img][1][thatChar[1]])
        thisVote = min(votedChars[img][0][thisChar[1]])
        thatVote = min(votedChars[img][0][thatChar[1]])
        if thisChar[0] == thatChar[0]:
            if thisVote < thatVote:
                doubles.add(thatChar)
            elif thisVote > thatVote:
                doubles.add(thisChar)

    return list(filter(lambda ch: ch not in doubles, charsList))


def getConnectedComponents(imageName, annotations, bwmask):
    """
    getConnectedComponents('056r_178_258_1393_1827/768_1024_47_181.png', words[imageName], bwmask)
    {'056r_178_258_1393_1827/768_1024_47_181.png': [
                ['p', 'a', 'r', 'i_bis', 'u', 'i_bis', 'p', 'n', 'd', 'e'],
                [['p', 'a', 'r', 'i_bis'],  ['u', 'i_bis', 'p'],  ['n'],  ['d'],  ['e']]
                ]
    }
    :param imageName: string. Relative path/image: dir/name.png
    :param annotations: words.json. Associates images to the relative transcribed word (as list of centroids and chars)
    :param bwmask: black and white mask
    :return: dict of list. Image to [full word, [connected components]]
    """
    fullWord = [c[1] for c in annotations]
    connectedCoords = bbxesCoverage(bwmask)
    # placeholders for each connected component chars list
    # placeholder = connectedCoords if fullWord[-1] != 'semicolon' else connectedCoords[:-1]
    connected = [[] for _ in connectedCoords]

    for centroid, ch in annotations:
        for i, bbox in enumerate(connectedCoords):
            start, end, ctr = bbox
            centroid = centroid[:2]
            ctr = ctr.tolist()
            if centroid == ctr or start <= centroid[0] < end:
                break
        connected[i].append(ch)

    # cv2.imshow(bwmask)
    # cv2.waitKey(0)
    return {imageName: [fullWord, connected]}
