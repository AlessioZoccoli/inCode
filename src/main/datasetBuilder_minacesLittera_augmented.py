from json import load
from os import path
from random import sample

from cv2 import imwrite
from numpy import hstack

from config import wordsRichDoublesAndUppercase, images2ColorsBBxesJSON, transcriptedWords_holesFree, \
    menaceLetterDatasetTRAIN
from src.lib.createMinacesLittera import createLetter, SizeException


# TO DO  


#   Creates the dataset by using the menace letter (ml) mechanism.
#   Dataset: list of images composed like this
#
#                             A                                          B
#               condition or sketch (ml - separated tokens)    |    target (manuscript)
#
#   This particular dataset is similar to the one created with ''datasetBuilder_minacesLittera'' except for the fact
#   that it produces A-B pairing for substrings too.
from src.lib.indexing import getIndex, query


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

        # TRAIN/TEST SET SPLITTING
        imgIndices2len = [(hfImg, l) for hfImg in holesFree
                          for l in range(2 if len(wordsNColors[hfImg]["tks"]) > 1 else 1, len(wordsNColors[hfImg]["tks"])+1)]
        # totNumWords = len(holesFree)
        # imagesShuffled = sample(holesFree, totNumWords)
        #trainingSetSize = round(totNumWords * trainSetProp)

        print('\n\n#############################')
        print('TRAINING SET BUILDING')
        print('#############################\n')
        c = 1

        for imTrain in imagesShuffled[:trainingSetSize]:
            colors, tokensBBxs = wordsNColors["col"], wordsNColors["tks"]
            wlen = len(tokensBBxs)

            if wlen > 1:
                for substrLen in range(2, wlen + 1):
                    try:
                        # original color word
                        if substrLen == wlen:
                            chars = [t[1] for t in tokensBBxs]
                            targetWrd = "".join(chars)
                            # TARGET
                            #       tokens to crop (all)
                            targetChars2Imgs = {targetWrd: [imTrain, chars]}
                            #       menace letter
                            target = createLetter(targetChars2Imgs, [targetWrd], toWhitePaper=False, is256=True,
                                                  separate=False)

                            # CONDITION
                            condChar2Images, condOrderedComps = query(index, text=" ".join(chars), forceHead=len(chars) > 1)
                            condition = createLetter(condChar2Images, condOrderedComps, toWhitePaper=False, is256=True,
                                                     separate=False)
                            a2b = hstack((target, condition))
                            assert a2b.shape == (256, 256 * 2)

                            del target, condition
                            # WRITE OUT
                            thisImgName = renameSubImage(imTrain, wlen, 0)
                            a2bWriteStatus = imwrite(path.join(menaceLetterDatasetTRAIN, thisImgName), a2b)
                            assert  a2bWriteStatus

                            del a2b

                            if c % 50 == 0:
                                print(' -----  TRAINIG SET number of processed images: ', c)
                            c += 1

                        # SUBSTRINGS
                        else:


                    except SizeException as s:
                        print(s)
                        pass