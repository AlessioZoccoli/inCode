from os import path
from pprint import pprint
from numpy.core.multiarray import zeros
from numpy.random import randint
from numpy import flip, uint8, invert
from config import color_words, images2ColorsBBxesJSON, ezechieleMinace
from json import load
from cv2 import imshow, imread, waitKey, destroyAllWindows, imwrite
from src.utils.imageProcessing import creatBackground, extractComponent
import ezechieleData


if __name__ == '__main__':

    with open(images2ColorsBBxesJSON, 'r') as ann:
        ch2col = load(ann)

    ccToTokens = ezechieleData.mapping
    phrase = ezechieleData.verse  # ['perc', 'he']


    def goesBelowLine(myString):
        below = 0
        if len(set(myString).intersection({'p', 'q', 'g', 'ss'})) > 0:
            below = 1
        elif 'h' in myString:
            below = 1/randint(2, 3)
        return below


    # mean width of a 1 char long tokens
    meanWidth = 17.30232622312978
    meanHeight = 17.274067142355833  # see goesBelowLine
    widthBackground = 1400
    heightBackground = 1900
    maxHeight = 69
    maxWidth = 698

    patches = []

    for subString in phrase:
        if subString != ' ':
            subStringImg, tokens = ccToTokens[subString][0], ccToTokens[subString][1]
            xStart, xEnd = 0., 0.
            xStartPos, xEndPos = 0, 0
            yStart, yEnd = [], []
            hasBigChar = 0.0
            groupedCols = []

            ending = (subString[0] == '_')

            # coordinates. This step consists in applying a mask on
            #  <ch2col[subStringImg]['tks']> to fit <tokens>
            found = False
            currStart = 0
            transcribed = ch2col[subStringImg]['tks']

            while not found and currStart <= len(transcribed) - len(tokens):
                window = transcribed[currStart:currStart + len(tokens)]
                windowChar = [tr[1] for tr in window]

                if tokens == windowChar:
                    found = True
                    xStart = window[0][0][5]
                    xEnd = window[-1][0][6]
                    yStart = min([tk[0][7] for tk in window])
                    yEnd = max([tk[0][8] for tk in window])
                    hasBigChar = int(max([(tk[0][4] - meanHeight)*goesBelowLine(tk[1])
                                          for tk in window if goesBelowLine(tk[1]) > 0] or [0.0]))
                else:
                    currStart += 1

            # colors. Colors may not be indexed with the same keys as bbxes
            for t in tokens:
                _token = t
                try:
                    if t not in ch2col[subStringImg]['col']:
                        if t.isupper():
                            _token = t.lower()
                        elif t[1] == t[0]:
                            _token = t[0]
                        elif t in ('us', 'ue'):
                            _token = 'semicolon'
                except IndexError:
                    pass

                print(subStringImg, _token)
                if _token:
                    groupedCols.extend(ch2col[subStringImg]['col'][_token])

            groupedCols = flip(groupedCols, axis=1)
            img = imread(path.join(color_words, subStringImg))
            patch = extractComponent(img, groupedCols, xStart, xEnd, yStart, yEnd)
            patches.append((patch, hasBigChar))

        else:
            spaceDim = (randint(round(meanWidth / 5), round(meanWidth / 4)), round(meanHeight))
            space = zeros(spaceDim, dtype=uint8)
            patches.append((space, 0.0))

    #
    #   Creating the image
    #

    back = creatBackground()
    xOff, xOffRand = 40, 0  # meanWidth = 17 circa
    # interRowOffset = 70
    row = 0

    for idx, patch2hbc in enumerate(patches):
        ptch, hbc = patch2hbc
        try:
            previousIsSpace = (phrase[idx - 1] == ' ')
        except IndexError:
            previousIsSpace = False

        randProximity = 0 if xOff == 0 or previousIsSpace else randint(0,
                                                                       4)  # random proximity to the last taken component

        interRowOffset = int(90 if row == 0 else 70)

        yOff = row * interRowOffset + round(interRowOffset - ptch.shape[0] + hbc)
        xOffRand = xOff - randProximity

        dy = yOff + ptch.shape[0]
        dx = xOffRand + ptch.shape[1]

        try:
            _next = patches[idx + 1][0].shape[1] if phrase[idx + 1] != ' ' else 0
        except IndexError:
            _next = 0

        if dx + _next <= widthBackground - 45:
            back[yOff:dy, xOffRand:dx] = ptch
            xOff += ptch.shape[1]
        elif dy < heightBackground - interRowOffset:
            row += 1
            yOff = row * interRowOffset + round(interRowOffset - ptch.shape[0] + hbc)
            xOff = 40
            xOffRand = xOff + randProximity
            dy = yOff + ptch.shape[0]
            dx = xOffRand + ptch.shape[1]
            back[yOff:dy, xOffRand:dx] = ptch
            xOff += ptch.shape[1]
        else:
            break

    toWhitePaper = invert(back)
    imshow('lectera minaces', toWhitePaper)
    waitKey(0)
    destroyAllWindows()

    """
    if not path.exists(ezechieleMinace):
        imwrite(ezechieleMinace, toWhitePaper)
    """
