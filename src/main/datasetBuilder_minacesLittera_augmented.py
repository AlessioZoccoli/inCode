from json import load
from random import sample

from config import wordsRichDoublesAndUppercase, images2ColorsBBxesJSON, transcriptedWords_holesFree
from src.lib.createMinacesLittera import createLetter


#   Creates the dataset by using the menace letter (ml) mechanism.
#   Dataset: list of images composed like this
#
#                             A                                          B
#               condition or sketch (ml - separated tokens)    |    target (manuscript)
#
#   This particular dataset is similar to the one created with ''datasetBuilder_minacesLittera'' except for the fact
#   that it produces A-B pairing for substrings too.
from src.lib.indexing import getIndex


def renameSubImage(origName, length, off):
    """
    Given origName = "040v/599_532_32_88.png"
    length (as the length of the color word contained in the same image) = 4    <= "eccli" <= ["e", "cc", "l", "i"]
    off (as the offset of the sub word/string inside the original one) = 0      <= ["e", ...]

    output = length-off--page-position.png  = 4-0--040v-599_532_32_88.png

    :param origName: str. Name of the image
    :param length: int. Length of the word contained in the current image
    :param off: int. Offset or starting index of the current sub word inside the original word
    :return: str. Formatted name.
    """
    page, position = origName.split('/')
    return str(length) + '-' + str(off) + '--' + page + '-' + position


def datasetBuilderMinacesLittera_augmented(trainSetProp=0.9):
    index = getIndex(indexName='baselineIndex')

    with open(images2ColorsBBxesJSON, 'r') as w, open(transcriptedWords_holesFree, 'r') as hf:
        wordsNColors = load(w)
        holesFree = set(load(hf))  # no holes between tokens

        totNumWords = len(holesFree)
        imagesShuffled = sample(holesFree, totNumWords)
        trainingSetSize = round(totNumWords * trainSetProp)

