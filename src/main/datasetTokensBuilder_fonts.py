from cv2 import imshow, waitKey, destroyAllWindows, imwrite, imread, IMREAD_GRAYSCALE
from json import load
from os import path
from config import *
from src.lib.createMinacesLittera import createLetter
from numpy import invert
from src.utils.imageProcessing import bbxes_data, mergeBBxes

"""

            Fonts
            NOT USED ANYMORE


"""


def symbolsClass():
    """
    Produces 3 differt (vertical)
    threatening letters, one for each set of tokens: lowercases, simple uppercases and specials
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

    lowerTks = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 'q', 'r', '_s', 's', 't', 'u',
                'x']
    upperTks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']
    specialTks = [',', '.', 'Ca', 'ce', 'de', 'eg', 'epo', 'ex', 'fa', 'fi', 'gl', 'nt', 'pa', 'per', 'pro', 'prop',
                  'que', 'qui', 'rum', 'semicolon', 'si', 'curl', 'con', 'ca']

    lowerImg = invert(imread(symbolsLower, IMREAD_GRAYSCALE))
    upperImg = invert(imread(symbolsUpper, IMREAD_GRAYSCALE))
    specialImg = invert(imread(symbolsSpecial, IMREAD_GRAYSCALE))

    lowerBBXs = sorted(bbxes_data(lowerImg), key=lambda e: e[1])
    lowerBBXs[8] = mergeBBxes(lowerBBXs[8], lowerBBXs[9])  # i: dot + stroke
    lowerBBXs.pop(9)

    upperBBXs = sorted(bbxes_data(upperImg), key=lambda e: e[1])

    """
    Merging symbols/tokens written as disconnected components es. i, qui, 
    """
    specialBBXs = sorted(bbxes_data(specialImg), key=lambda e: e[1])
    # Ce
    specialBBXs[2] = mergeBBxes(specialBBXs[2], specialBBXs[3])
    specialBBXs[3] = (0, 0)
    # epo
    specialBBXs[7] = mergeBBxes(specialBBXs[7], specialBBXs[8])
    specialBBXs[8] = (0, 0)
    # pa
    specialBBXs[14] = mergeBBxes(specialBBXs[14], specialBBXs[15])
    specialBBXs[15] = (0, 0)
    # prop
    specialBBXs[18] = mergeBBxes(mergeBBxes(specialBBXs[18], specialBBXs[19]), specialBBXs[20])
    specialBBXs[19], specialBBXs[20] = (0, 0), (0, 0)
    # que
    specialBBXs[21] = mergeBBxes(mergeBBxes(specialBBXs[21], specialBBXs[22]), specialBBXs[23])
    specialBBXs[22], specialBBXs[23] = (0, 0), (0, 0)
    # qui
    specialBBXs[24] = mergeBBxes(specialBBXs[24],
                                 mergeBBxes(
                                     mergeBBxes(specialBBXs[25], specialBBXs[26]), specialBBXs[27]))
    specialBBXs[25], specialBBXs[26], specialBBXs[27] = (0, 0), (0, 0), (0, 0)
    # semicolon
    specialBBXs[29] = mergeBBxes(specialBBXs[29], specialBBXs[30])
    specialBBXs[30] = (0, 0)
    # si
    specialBBXs[31] = mergeBBxes(specialBBXs[31], specialBBXs[32])
    specialBBXs[32] = (0, 0)

    # clean up
    specialBBXs = [sb for sb in specialBBXs if sb != (0, 0)]

    def getBBXS(bbList, image, alphabet, store=False):
        bbxes = []
        for bb, alpha in zip(bbList, alphabet):
            y, dy = bb[-2:]
            x, dx = bb[-4:-2]
            patch = image[y: dy, x: dx]
            if store:
                if alpha.isupper():
                    _alpha = alpha * 2
                elif alpha == ",":
                    _alpha = "comma"
                elif alpha == ".":
                    _alpha = "fullStop"
                elif alpha[0] == "_":
                    _alpha = "ending_" + alpha[1:]
                elif alpha == 'Ca':
                    _alpha = "CCa"
                else:
                    _alpha = alpha

                currPath = path.join(fontsTokens, _alpha + ".png")
                if not path.exists(currPath):
                    imwrite(currPath, patch)

            bbxes.append(patch)
        return bbxes

    # getBBXS(lowerBBXs, lowerImg, lowerTk, store=True)
    # getBBXS(upperBBXs, upperImg, upperTk, store=True)
    getBBXS(specialBBXs, specialImg, specialTk, store=True)

    """
    for ind, symbol in enumerate(getBBXS(lowerBBXs, lowerImg, alphabet=lowerTk)):
        imshow(lowerTk[ind], symbol)
        waitKey(0)
        destroyAllWindows()

    for ind, symbol in enumerate(getBBXS(upperBBXs, upperImg, alphabet=upperTk)):
        imshow(lowerTk[ind], symbol)
        waitKey(0)
        destroyAllWindows()

    for ind, symbol in enumerate(getBBXS(specialBBXs, specialImg, alphabet=specialTk)):
        imshow(lowerTk[ind], symbol)
        waitKey(0)
        destroyAllWindows()
    """


if __name__ == "__main__":
    symbolsClass()
    getArtificialTokensFromList()
