from cv2 import imshow, waitKey, destroyAllWindows, imwrite
from os import path

import ezechieleData
from numpy.random import randint
from src.lib.createMinacesLittera import createLetter
from src.lib.indexing import getIndex, query


def textToClippings(text):
    """
    From string of text to a list containing sorted connected components and a dict containing the same connected components
    plus their respective image and tokens
    :param text: str. Input text to render as a menace letter
    :return:
            dict
                 ...
                 "wor": ["image.png", ["w", "o", "r"]],
                 ...

            list
                ["he", "ll", "o", " ", "wod","d"]

    """
    ix = getIndex(indexName='baselineIndex')
    char2Images, orederedComps = query(ix, text)
    return char2Images, orederedComps


def ezechieleTest256(ccToTokens=ezechieleData.mapping, phrase=ezechieleData.verse):
    littera = createLetter(ccToTokens, phrase, sketch=True)
    imshow('lectera minaces', littera)
    waitKey(0)
    destroyAllWindows()


if __name__ == '__main__':
    char2Images, orederedComps = textToClippings("cie di mi")
    ezechieleTest256(ccToTokens=char2Images, phrase=orederedComps)
