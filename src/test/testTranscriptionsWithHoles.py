from json import load, dump

from pprint import pprint
from src.utils.imageProcessing import getMissingElements, getAnnotatedBBxes
from cv2 import imread, imshow, waitKey, destroyAllWindows
from config import color_words, annotationsRichJSON, wordsRichDoublesAndUppercase


def testTranscriptionsWithHoles():
    """

    :return:
    """
    with open(annotationsRichJSON, 'r') as fa, open(wordsRichDoublesAndUppercase, 'r') as wa:
        annots = load(fa)
        words = load(wa)

    #              has holes                    has holes                       has no holes
    images = ['/050v/1122_265_42_147.png', '/050v/1125_1370_47_95.png', '/049v_586_258_1366_1821/1140_1533_23_52.png']

    for im in images:
        _im = color_words + im
        _imImage = imread(_im)
        _imColors = annots[im[1:]]
        _imBBxes = words[im[1:]]

        _imMiss, _imMissBB = getMissingElements(_imImage, _imColors, _imBBxes, returnImage=True)
        # data and coordinates
        pprint(_imMissBB)
        print('\n')
        # show image
        imshow(im, _imMiss)
        waitKey(0)
        destroyAllWindows()


if __name__ == '__main__':
    testTranscriptionsWithHoles()