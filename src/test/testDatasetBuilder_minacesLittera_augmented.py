from pprint import pprint
from json import load
from cv2 import imshow, waitKey, destroyAllWindows, imread, bitwise_or

from numpy import hstack, zeros, uint8

from config import wordsRichDoublesAndUppercase, images2ColorsBBxesJSON, color_words
from src.lib.createMinacesLittera import createLetter, meanHeight, goesBelowLine
from src.lib.indexing import getIndex, query
from src.main.datasetBuilder_minacesLittera_augmented import renameSubImage
from src.utils.imageProcessing import getAnnotatedBBxes


def testDatasetBuilderMinacesLittera_augmented():
    """
    Give a color word C of length L produces L(L-1)/2 + 1 black and white images. L(L-1)/2 for the substrings and one for the original.

    AnnotatedTks(C): ["e", "cc", "l", "i", "e"]
    "040v/599_532_32_88.png"
    #  2 (4)
            ecc [0, 1]
            ccl [1, 2]
            li [2, 3]
            ie [3, 4]
    #  3 (3)
            eccl [0, 2]
            ccli [1, 3]
            lie [2, 4]
    #  4 (2)
            eccli [0, 3]
            cclie [1, 4]
    #  5 (1)
            ecclie [0, 4]
    :return: None
    """
    inputImage = "040v/599_532_32_88.png"  # ecclie

    with open(images2ColorsBBxesJSON, "r") as f:
        data = load(f)[inputImage]

    colors, tokensBBxs = data["col"], data["tks"]

    wlen = len(tokensBBxs)

    # targetImage = imread(path.join(color_words, inputImage))

    print("###")
    pprint(tokensBBxs)
    producedSubstrings = set()

    index = getIndex(indexName='baselineIndex')

    for substrLen in range(2, wlen+1):
            print("\n# ", substrLen)
            # original color word
            if substrLen == wlen:
                chars = [t[1] for t in tokensBBxs]
                targetWrd = "".join(chars)
                producedSubstrings.add(targetWrd)
                print("       ", targetWrd)

                # TARGET
                targetChars2Imgs = {targetWrd: [inputImage, chars]}
                target = createLetter(targetChars2Imgs, [targetWrd], toWhitePaper=False, is256=True, separate=False)

                # CONDITION
                condChar2Images, condOrderedComps = query(index, text=" ".join(chars), forceHead=len(chars) > 1)
                condition = createLetter(condChar2Images, condOrderedComps, toWhitePaper=False, is256=True, separate=False)
                a2b = hstack((target, condition))
                del target, condition

                # DISPLAY
                thisImgName = renameSubImage(inputImage, wlen, 0)
                imshow(thisImgName, a2b)
                waitKey(0)
                destroyAllWindows()
            else:
                for offset in range(0, wlen - substrLen + 1):
                    _tksBBxes = tokensBBxs[offset: offset + substrLen]
                    chars = [t[1] for t in _tksBBxes]
                    targetWrd = "".join(chars)
                    print("       {}   [{}, {}]".format(targetWrd, offset, offset + substrLen - 1))

                    # TARGET
                    targetChars2Imgs = {targetWrd: [inputImage, chars]}
                    targetOrdComps = [targetWrd]
                    target = createLetter(targetChars2Imgs, targetOrdComps, toWhitePaper=False, is256=True)

                    # CONDITION
                    condChar2Images, condOrderedComps = query(index, text=" ".join(chars), forceHead=len(chars) > 1)
                    condition = createLetter(condChar2Images, condOrderedComps, toWhitePaper=False, is256=True,
                                             separate=False)
                    a2b = hstack((target, condition))
                    del target, condition

                    # DISPLAY
                    thisImgName = renameSubImage(inputImage, substrLen, offset)
                    imshow(thisImgName, a2b)
                    waitKey(0)
                    destroyAllWindows()

                    producedSubstrings.add(targetWrd)

    print("\n", producedSubstrings)
    assert len(producedSubstrings) == int(wlen*(wlen-1)/2.0)


if __name__ == "__main__":
    testDatasetBuilderMinacesLittera_augmented()
