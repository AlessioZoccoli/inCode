import cv2
import numpy as np
from src.utils.imageProcessing import bbxes_data, bbxesCoverage, mask_by_colors
from itertools import combinations, product
from numpy import mean, diff, absolute


def char2position(imgPath, charColors, character='', show=False):
    """
    returns the centroid and area of the bbox corresponding to the mask of charColors
    :param imgPath: string. file (relative) path, eg. color_words/040v/159_585_41_63.png
    :param charColors: list of [G, B, R] colors
    :param character: current character for which bbxes are required
    :param show: boolean. Display bbxes for debugging
    :return: list of tuples. Each tuple consists in (xCentroid, yCentroid, bboxArea, Width, Height, xStart, xEnd).
            this list exclude possibile disconnected fragments.
    """
    image = cv2.imread(imgPath)
    mask = mask_by_colors(image, charColors)

    if show:
        mask3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        print(character)
        cv2.imshow("images", np.hstack([image, mask3ch]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #         xCentr, yCentr, area, width, height, xStart, xEnd, yStart, yEnd
    charpos = bbxes_data(mask)
    nElem = len(charpos)
    validIndices, taken, omit = set(), set(), set()

    # 'u' is often written as two separate strokes of the same size, let's use stricter constraints
    xCentroidThrashold = 13.2 if character != 'u' else 11.0
    areaThrashold = 0.99 if character != 'u' else 1.1

    if len(charpos) > 1:
        for _curr, _next in zip(range(nElem), range(1, nElem)):
            # if _curr is already in omit, use the last one taken, otherwise there is poor aggregation of bbxes
            if _curr in omit:
                _curr = max(taken)

            # sort by area                  (bbx, index)
            _min, _max = sorted([(charpos[_curr], _curr),
                                 (charpos[_next], _next)],
                                key=lambda el: el[0][2])
            if abs(_max[0][0] - _min[0][0]) < xCentroidThrashold and (_min[0][2] / _max[0][2]) < areaThrashold:
                omit.add(_min[1])
                # picking the right bbxes boundaries  for the aggregated bbox -> xStart, xEnd, yStart, yEnd
                xStart = min([charpos[_curr][5], charpos[_next][5]])
                xEnd = max([charpos[_curr][6], charpos[_next][6]])

                yStart = min(charpos[_curr][7], charpos[_next][7])
                yEnd = max(charpos[_curr][8], charpos[_next][8])

                charpos[_max[1]] = ((_max[0][0] + _min[0][0]) / 2.0,              # xCentroid
                                    (_max[0][1] + _min[0][1]) / 2.0,              # yCentroid
                                    _min[0][2] + _max[0][2],                      # area
                                    _min[0][3] + _max[0][3],                      # width
                                    _max[0][4],                                   # height
                                    xStart, xEnd, yStart, yEnd)
                taken.add(_max[1])
            else:
                taken.add(_max[1])
                taken.add(_min[1])
        validIndices = taken - omit
    else:
        validIndices.add(0)
    return sorted([charpos[i] for i in validIndices], key=lambda x: x[0])


def positions2chars(imgPath, char2colors, votes=None, show=False):
    """
    This method associates to each word-image(file path) a list of its characters with corresponding xCentroid, area
    and vote
    :param imgPath: file path to the image
    :param char2colors: list of lists.
        Each list is a doubleton, two elements only:
            - char
            - colors
    :param votes: Mapping between annotated chars and their votes.
    :return: list of tuples in the form of    ((xCentroid, yCentroid, Area, Width, Height, xStart, xEnd, yStart, yEnd), char)
    """
    # toScalar is necessary for  serialization
    toScalar = (lambda vals: tuple(np.asscalar(el) for el in vals))
    # output
    pos2ch = []
    ch2col = char2colors.items()

    for ch, colors in ch2col:
        # colors(RGB) -> colors(GBR)
        colorsGBR = np.flip(np.array(colors, dtype=np.uint8), 1)
        bboxesStats = char2position(imgPath, colorsGBR, ch, show)
        if ch == 'semicolon':
            bboxesStats = [bboxesStats[0]]
        pos2ch.append([(toScalar(bbxdata), ch) for bbxdata in bboxesStats])

    pos2ch = sorted([y for x in pos2ch for y in x])

    """
    removing 'stroke's
    ... 'l_stroke', 'l' ... -> 'l'
    ... 'l', 'l_stroke' ... -> 'l'
    ... 'l_stroke' ... -> 'l'
    """
    stroked = set()
    for i in range(len(pos2ch)):
        if len(pos2ch[i][1]) == 1:
            try:
                if pos2ch[i][1][0] == pos2ch[i - 1][1][0] and len(pos2ch[i - 1][1]) > 1 and pos2ch[i - 1][1][1] == '_' \
                        and pos2ch[i - 1][1][0][2] / pos2ch[i][1][0][2] < 1.:
                    stroked.add(pos2ch[i])
            except IndexError:
                pass
            try:
                if pos2ch[i][1][0] == pos2ch[i + 1][1][0] and len(pos2ch[i + 1][1]) > 1 and pos2ch[i + 1][1][1] == '_' \
                        and pos2ch[i + 1][1][0][2] / pos2ch[i][1][0][2] < 1.:
                    stroked.add(pos2ch[i])
            except IndexError:
                pass

    pos2ch = [pc for pc in pos2ch if pc not in stroked]

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

                currCharLeft = pos2ch[i][1]
                currLeft = currCharLeft
                currCharRight = pos2ch[doubleCharIndex][1]
                currRight = currCharRight

                # doubles and uppercases checking
                try:
                    if currCharRight[1] == currCharRight[0]:
                        currRight = currCharRight[0]
                except IndexError:
                    pass
                if currCharRight.isupper():
                    currRight = currCharRight.lower()

                try:
                    if currCharLeft[1] == currCharLeft[0]:
                        currLeft = currCharLeft[0]
                except IndexError:
                    pass
                if currCharLeft.isupper():
                    currLeft = currCharLeft.lower()

                assert isinstance(currRight, str)
                assert isinstance(currLeft, str)

                isManualConn_Left = False
                isManualConn_Right = False

                if currLeft not in votes:
                    if currLeft + "_stroke" in votes:
                        currLeft += "_stroke"
                    elif currLeft + "_new" in votes:
                        currLeft += "_new"
                    else:
                        isManualConn_Left = True  # handmade connected comps = trustworthy

                if currRight not in votes:
                    if currRight + "_stroke" in votes:
                        currRight += "_stroke"
                    elif currRight + "_new" in votes:
                        currRight += "_new"
                    else:
                        isManualConn_Right = True  # handmade connected comps = trustworthy

                # Assuming isManualConn_Left and *_right are not True at the same time
                if not (isManualConn_Left or isManualConn_Right):
                    toRemove = min([
                        (i, min(votes[currLeft])),  # list index and minimum vote for the char
                        (doubleCharIndex, min(votes[currRight]))],
                        key=lambda x: x[1]
                    )[0]  # just keep the list index
                elif isManualConn_Left:
                    toRemove = i
                else:
                    toRemove = doubleCharIndex
                del coords[toRemove]
                del pos2ch[toRemove]
            except ValueError:
                pass
            i += 1

        # if length of pos2ch is still > 1 then evaluate other cases
        if len(pos2ch) > 1:
            # elements of this list will be omitted
            omit = set()
            meanCentroidXDistance = mean(absolute(diff([el[0][0] for el in pos2ch])))  # if len(pos2ch) > 1 else pos2ch[]

            for this, that in combinations(ch2col, 2):
                overlapping = [c for c in this[1] if c in that[1]]
                if overlapping:
                    overlGBR = np.flip(np.array(overlapping, dtype=np.uint8), 1)
                    ovrlBBexes = char2position(imgPath, overlGBR)

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

    [['p','a','r','u','p','n','d','e']
                ||
                V
    {'056r_178_258_1393_1827/768_1024_47_181.png':
                                                    [['p', 'a', 'r'],
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
    connectedCoords = bbxes_data(bwmask)  # bbxesCoverage(bwmask)
    # placeholders for each connected component
    connections = [[] for _ in connectedCoords]

    for centroid, ch in annotations:
        for i, bbox in enumerate(connectedCoords):
            xCtr, yCtr, xStart, xEnd, yStart, yEnd = bbox[0], bbox[1], bbox[5], bbox[6], bbox[7], bbox[8]
            centroid = centroid[:2]     # from annotations
            ctr = [xCtr, yCtr]          # from bwmask connected comps
            if centroid == ctr or (xStart <= centroid[0] < xEnd and yStart <= centroid[1] <= yEnd):
                break
        connections[i].append(ch)

    return {imageName: [comp for comp in connections if comp]}
