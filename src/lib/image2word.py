from pprint import pprint

import cv2
import numpy as np
from src.utils import utils as utils
from itertools import combinations, product
from numpy import mean, diff, absolute
from src.utils.utils import bbxesCoverage, mask_by_colors


def char2position(imgPath, charColors):
    """
    returns the centroid and area of the bbox corresponding to the mask of charColors
    :param imgPath: string. file (relative) path, eg. color_words/040v/159_585_41_63.png
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
            # sort by area
            _min, _max = sorted([thisBB, thatBB], key=lambda el: el[2])
            if abs(_max[0] - _min[0]) < 13.2 and _min[2] / _max[2] < 0.99:
                # if we sum up _min and _max areas into newMax and we add this to taken
                # we will eventually have both _max and newMax, let's remove _max first
                omit.add(_max)
                omit.add(_min)

                newMax = (_max[0], _max[1], _min[2] + _max[2])
                # print(newMax)
                taken.add(newMax)
            else:
                taken.add(_max)
                taken.add(_min)
        charpos = taken - omit
    # print('TAKEN:', taken)
    # print('OMIT:', omit)
    return sorted([el for el in charpos], key=lambda x: x[0])


def positions2chars(imgPath, char2colors, votes=None):
    """
    This method associates to each word-image(file path) a list of its characters with corresponding xCentroid, area
    and vote
    :param imgPath: file path to the image
    :param char2colors: list of lists.
        Each list is a doubleton, two elements only:
            - char
            - colors
    :param votes: Mapping between annotated chars and their votes.
    :return: list of tuples in the form of    ((xCentroid, yCentroid, Area), char)
    """
    toScalar = (lambda el: (np.asscalar(el[0]), np.asscalar(el[1]), np.asscalar(el[2])))
    # output
    pos2ch = []
    ch2col = char2colors.items()

    for ch, colors in ch2col:
        # colors(RGB) -> colors(GBR)
        colorsGBR = np.flip(np.array(colors, dtype=np.uint8), 1)
        # print(ch)
        bboxesStats = char2position(imgPath, colorsGBR)
        if ch == 'semicolon':
            bboxesStats = [bboxesStats[0]]
        pos2ch.append([(toScalar(coord), ch) for coord in bboxesStats])

    pos2ch = sorted([y for x in pos2ch for y in x])

    if len(pos2ch) > 1:
        #
        #       overlappings handling
        #
        # searching chars with same color, shape and centroid first
        coords = [e[0] for e in pos2ch]
        votes = votes[0]
        i = 0

        while i < len(coords):
            try:
                doubleCharIndex = coords[i + 1:].index(coords[i]) + i + 1
                toRemove = min([
                    (i, min(votes[pos2ch[i][1]])),  # list index and minimum vote for the char
                    (doubleCharIndex, min(votes[pos2ch[doubleCharIndex][1]]))],
                    key=lambda x: x[1]
                )[0]  # just keep the list index
                del coords[toRemove]
                del pos2ch[toRemove]
            except ValueError:
                pass
            i += 1

        # if length of pos2ch is still >1 then evaluate other cases
        if len(pos2ch) > 1:
            # elements of this list will be omitted
            omit = set()
            meanCentroidXDistance = mean(absolute(diff([el[0][0] for el in pos2ch]))) # if len(pos2ch) > 1 else pos2ch[]

            for this, that in combinations(ch2col, 2):
                overlapping = [c for c in this[1] if c in that[1]]
                if overlapping:
                    overlGBR = np.flip(np.array(overlapping, dtype=np.uint8), 1)
                    ovrlBBexes = [coordOvr for coordOvr in char2position(imgPath, overlGBR)]

                    # grouping overlapping chars
                    thisBBxes, thatBBxes = [], []
                    for ch in pos2ch:
                        if ch[1] == this[0]:
                            thisBBxes.append(ch)
                        elif ch[1] == that[0]:
                            thatBBxes.append(ch)

                    # choosing bbox to delete by comparing all possible overlapping bboxes
                    for p in product(thisBBxes, thatBBxes):
                        #   centroids distance (x axis) -> too short  AND  overlBBx between two chars bbxes  AND   overlBBx area
                        #   is big ====>  bad overlapping, remove!
                        #
                        leftBB, rightBB = sorted([p[0][0], p[1][0]], key=lambda b: b[0])
                        maxOvrl = max(ovrlBBexes, key=lambda x: x[2])
                        meanNeighbourArea = np.mean([leftBB[2], rightBB[2]])

                        if (rightBB[0] - leftBB[0]) < meanCentroidXDistance and leftBB[0] <= maxOvrl[0] <= rightBB[0] \
                                and maxOvrl[2] / meanNeighbourArea > .50:
                            omitted = min(p, key=lambda b: b[0][2])
                            omit.add(omitted)

            word = [w for w in pos2ch if w not in omit]
        else:
            word = pos2ch
    else:
        word = pos2ch
    return word


def disambiguate(img, charsList, votedChars):
    """
    Calling this method assumes there are overlapping (same centroid) chars.

    Given two charachters having the same centroid we may have one written character recordered as two distinct ones,
    We want to keep the one with more votes
    :param img: image path
    :param charsList: list of tuples ((<xCentroid, yCentroid, area>),<char>)
    :param votedChars: mapping between annotated chars and votes for each word
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
    return [ch for ch in charsList if ch not in doubles]
    # return list(filter(lambda ch: ch not in doubles, charsList))


def getConnectedComponents(imageName, annotations, bwmask):
    """
    getConnectedComponents('056r_178_258_1393_1827/768_1024_47_181.png', words[imageName], bwmask)

    {'056r_178_258_1393_1827/768_1024_47_181.png': [['p','a','r','i_bis','u','p','n','d','e'],
                                                [['p', 'a', 'r', 'i_bis'],
                                                 ['u', 'p'],
                                                 ['n'],
                                                 ['d'],
                                                 ['e']]]
                                                 }

    :param imageName: string. Relative path/image: dir/name.png
    :param annotations: words.json. Associates images to the relative transcribed word (as list of centroids and chars)
    :param bwmask: black and white mask
    :return: dict of list. Image to [full word, [connected components]]
    """
    fullWord = [c[1] for c in annotations]
    connectedCoords = bbxesCoverage(bwmask)
    # placeholders for each connected component chars list
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
