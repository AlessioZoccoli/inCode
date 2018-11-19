from cv2 import imshow, waitKey, destroyAllWindows, imwrite, imread, IMREAD_GRAYSCALE, connectedComponentsWithStats
from os import path
from pprint import pprint
from numpy.core.umath import invert
from config import *
from json import load, dump

from src.lib.createMinacesLittera import createLetter
from src.utils.imageProcessing import bbxes_data, mergeBBxes

"""

    Creating the artificial tokens from the sketches

"""


def symbolsClass():
    """
    Produces 3 differt (vertical) threatening letters, one for each set of tokens:
    lowercases, simple uppercases and specials

    A vertical threatening letters for convenience, since it is easier than a horizontal arrangement.

    eg.
            a       _split()
            b       _s
            c       _s

            ;       _s
            .       _s
            ,       _s

    :return: None
    """
    with open(symbols, "r") as s:
        _symbols = load(s)

    ccToTokens = _symbols['symbol']
    sequence = _symbols['alphabetSeq']

    # SIMLE 1-grams
    simpleSeqUpper, simpleSeqLower, specialSeq = [], [], []
    for el in sequence:
        if el != " ":
            if el.isupper():
                simpleSeqUpper.append(el)
            elif (len(el) == 1 and el.islower and el.isalpha()) or (len(el) == 2 and el[0] == '_'):
                simpleSeqLower.append(el)
            else:
                specialSeq.append(el)

    litteraSimpleLowerTks = createLetter(ccToTokens, simpleSeqLower, vertical=True)
    litteraSimpleUpperTks = createLetter(ccToTokens, simpleSeqUpper, vertical=True)
    litteraSpecialTks = createLetter(ccToTokens, specialSeq, vertical=True)

    imshow('littera simple l', litteraSimpleLowerTks)
    waitKey(0)
    destroyAllWindows()
    print(simpleSeqLower)

    imshow('littera simple u', litteraSimpleUpperTks)
    waitKey(0)
    destroyAllWindows()
    print(simpleSeqUpper)

    imshow('littera specials', litteraSpecialTks)
    waitKey(0)
    destroyAllWindows()
    print(specialSeq)

    # Saving images with simple/special tokens
    if not path.exists(symbolsCarolingian_simpleLower):
        print("writing ", symbolsCarolingian_simpleLower)
        imwrite(symbolsCarolingian_simpleLower, litteraSimpleLowerTks)
    else:
        print(symbolsCarolingian_simpleLower)

    if not path.exists(symbolsCarolingian_simpleUpper):
        print("writing ", symbolsCarolingian_simpleUpper)
        imwrite(symbolsCarolingian_simpleUpper, litteraSimpleUpperTks)
    else:
        print(symbolsCarolingian_simpleUpper)

    if not path.exists(symbolsCarolingian_special):
        print("writing ", symbolsCarolingian_special)
        imwrite(symbolsCarolingian_special, litteraSpecialTks)
    else:
        print(symbolsCarolingian_special)


def getArtificialTokensFromList():
    """
    From the image containing all input fonts to single image for each one
    :return: None
    """

    lowerTks = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'm', 'n', 'o', 'p', 'q', 'r', 's', '_s', 't', 'u', 'x',
                'l']
    upperTks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']
    specialTks = ['comma', 'point', 'nt', 'per', 'pro', 'rum', 'semicolon', 'curl', 'con']

    lowerCaseTksImage = invert(imread(symbolsLower, IMREAD_GRAYSCALE))
    upperCaseTksImage = invert(imread(symbolsUpper, IMREAD_GRAYSCALE))
    specialTksImage = invert(imread(symbolsSpecial, IMREAD_GRAYSCALE))

    # extracting (and sorting) the bbxes stats and centroid, no background
    bbxesSortedY_lower = sorted(bbxes_data(lowerCaseTksImage), key=lambda d: d[1])
    bbxesSortedY_upper = sorted(bbxes_data(upperCaseTksImage), key=lambda d: d[1])
    bbxesSortedY_special = sorted(bbxes_data(specialTksImage), key=lambda d: d[1])

    assert len(lowerTks) == len(bbxesSortedY_lower)
    assert len(upperTks) == len(bbxesSortedY_upper)

    # ;
    bbxesSortedY_special[6] = mergeBBxes(bbxesSortedY_special[6], bbxesSortedY_special[7])
    del bbxesSortedY_special[7]
    assert len(specialTks) == len(bbxesSortedY_special)


    # LOWERCASE TOKENS
    for t, bb in zip(lowerTks, bbxesSortedY_lower):
        xStart, xEnd = bb[-4:-2]
        yStart, yEnd = bb[-2:]
        outPath = path.join(sketchTokens, t + '.png')
        if not path.exists(outPath):
            imwrite(outPath, lowerCaseTksImage[yStart:yEnd, xStart:xEnd])
            # imshow(t, lowerCaseTksImage[yStart:yEnd, xStart:xEnd])
            # waitKey(0)
            # destroyAllWindows()

    # UPPERCASE TOKENS
    for t, bb in zip(upperTks, bbxesSortedY_upper):
        xStart, xEnd = bb[-4:-2]
        yStart, yEnd = bb[-2:]
        outPath = path.join(sketchTokens, "".join(t*2).upper() + '.png')
        if not path.exists(outPath):
            imwrite(outPath, upperCaseTksImage[yStart:yEnd, xStart:xEnd])
            # imshow(t, upperCaseTksImage[yStart:yEnd, xStart:xEnd])
            # waitKey(0)
            # destroyAllWindows()

    # SPECIAL TOKENS
    for t, bb in zip(specialTks, bbxesSortedY_special):
        xStart, xEnd = bb[-4:-2]
        yStart, yEnd = bb[-2:]
        outPath = path.join(sketchTokens, t + '.png')
        if not path.exists(outPath):
            imwrite(outPath, specialTksImage[yStart:yEnd, xStart:xEnd])
            # imshow(t, specialTksImage[yStart:yEnd, xStart:xEnd])
            # waitKey(0)
            # destroyAllWindows()


if __name__ == '__main__':
    getArtificialTokensFromList()
