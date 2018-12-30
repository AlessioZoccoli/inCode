from os import path
from pprint import pprint
from src.lib.indexing import getIndex, query
from json import load
from cv2 import imread, imwrite, bitwise_or, imshow, waitKey, destroyAllWindows
from random import sample, randint
from config import *
from numpy import zeros, uint8, flip, hstack, ones
from src.utils.imageProcessing import extractComponent
from src.lib.createMinacesLittera import goesBelowLine, meanHeight, createLetter, SizeException


def getWord(image, colors, bbxes):
    blank = zeros(image.shape[:2], dtype=uint8)
    minXStart, maxXEnd = 500, 0
    minYStart, maxYEnd = 500, 0

    for bb, token in bbxes:
        t = token
        if token not in colors:
            if t.isupper():
                t = token.lower()
            elif t[1] == t[0]:
                t = token[0]
            elif t in ('us', 'ue'):
                t = 'semicolon'
            elif t is ".":  # just in case '.' color hasn't been annotated yet
                t = min(bbxes, key=lambda e: e[0][2])[1]
        _colors = flip(colors[t], axis=1)
        # coordinates
        xStart, xEnd, yStart, yEnd = bb[-4:]
        minXStart = min(minXStart, xStart)
        maxXEnd = max(maxXEnd, xEnd)
        minYStart = min(minYStart, yStart)
        maxYEnd = max(maxYEnd, yEnd)
        # sub-image containing s single component
        newToken = extractComponent(image, _colors, xStart, xEnd, yStart, yEnd) # extractComponent(image, _colors, xStart, xEnd - 1, yStart, yEnd - 1)
        blank[yStart:yEnd, xStart:xEnd] = bitwise_or(blank[yStart:yEnd, xStart:xEnd], newToken)

    return blank[minYStart:maxYEnd, minXStart:maxXEnd]


def datasetBuilderMinaceLetter(trainSetProp=0.9):
    index = getIndex(indexName='baselineIndex')

    with open(images2ColorsBBxesJSON, 'r') as w, open(transcriptedWords_holesFree, 'r') as hf:
        wordsNColors = load(w)
        holesFree = set(load(hf))  # no holes between tokens

        totNumWords = len(holesFree)
        imagesShuffled = sample(holesFree, totNumWords)
        trainingSetSize = round(totNumWords * trainSetProp)
        testSetSize = totNumWords - trainingSetSize

        def len2Image(imageName):
            page, image = imageName.split('/')
            return str(len(wordsNColors[imageName]['tks'])) + '##' + page + '#' + image

        # various = ['059v/840_1299_36_162.png', '048r/876_1667_34_25.png', '058r/852_979_31_73.png',
        #            '059r/949_832_36_36.png', '057r/328_137_34_40.png', '054r/953_1671_41_45.png',
        #            '051v/367_422_34_65.png', '060r_165_264_1424_1835/144_1249_30_48.png',
        #            '050r/295_1557_36_135.png', '050v/986_169_21_11.png', '060r_165_264_1424_1835/979_693_46_90.png',
        #            '059v/157_454_30_26.png', '055r/607_1046_27_18.png', '049v/470_340_31_92.png',
        #            '040v/1015_1075_49_128.png', '055v_631_241_1360_1839/876_972_37_41.png',
        #            '057v_542_281_1394_1819/792_1641_36_42.png', '060v/709_665_40_96.png',
        #            '055r/796_125_36_23.png', '055v_631_241_1360_1839/833_701_39_72.png', '040v/756_585_32_99.png',
        #            '053r/275_101_42_120.png', '051v/1262_265_33_57.png', '052r/242_1079_33_79.png',
        #            '053r/273_592_32_78.png']
        # variousWTSymbols = ['053r/681_1644_37_59.png', '060r_165_264_1424_1835/442_1588_36_46.png',
        #                     '054r/1180_460_28_35.png',
        #                     '054r/343_449_37_59.png', '060r_165_264_1424_1835/847_1422_35_69.png',
        #                     '040v/158_421_33_49.png',
        #                     '047r/1206_1281_35_56.png',
        #                     '053r/273_592_32_78.png', '053r/854_930_35_31.png', '051v/1073_1185_45_128.png',
        #                     '053r/922_1378_36_35.png', '054r/1086_285_49_137.png', '054r/743_565_46_99.png']

        #############################
        #                           #
        #       TRAINING SET        #
        #                           #
        #############################
        print('\n\n#############################')
        print('TRAINING SET BUILDING')
        print('#############################\n')
        c = 1
        for imTrain in imagesShuffled[:trainingSetSize]:
            try:
                imName = len2Image(imTrain)
                # TARGET = manuscript/real image
                targetColors = wordsNColors[imTrain]['col']
                targetBBxes = wordsNColors[imTrain]['tks']
                tarhetImage = imread(color_words + '/' + imTrain)
                target = getWord(tarhetImage, targetColors, targetBBxes)
                #         fitting the 256x256 format
                target256 = zeros((256, 256), dtype=uint8)
                hasBigChar = int(max([(tk[0][4] - meanHeight) * goesBelowLine(tk[1])
                                      for tk in targetBBxes] or [0.0]))
                yOff = 126 + round(3 - target.shape[0] + hasBigChar)
                xOff = round((256 - target.shape[1]) / 2)

                target256[yOff:yOff + target.shape[0], xOff:xOff + target.shape[1]] = \
                    bitwise_or(target256[yOff:yOff + target.shape[0], xOff:xOff + target.shape[1]], target)

                tokensList = [t[1] for t in targetBBxes]
                #
                # CONDITION = sketch/letter
                #       CONDITION 1 = minaces littera with singly taken tokens attached one another
                char2Images, orederedComps = query(index, text=" ".join(tokensList), forceHead=len(tokensList) > 1)
                conditionAttached = createLetter(char2Images, orederedComps, toWhitePaper=False,
                                                 sketch=True, separate=False)
                a2bAttached = hstack((target256, conditionAttached))
                #
                #       CONDITION 2 = minaces littera with singly taken tokens separated by spaces
                conditionSeparate = createLetter(char2Images, orederedComps, toWhitePaper=False, sketch=True,
                                                 separate=True)
                a2bSeparate = hstack((target256, conditionSeparate))

                # check sizes
                assert conditionAttached.shape == (256, 256)
                assert conditionSeparate.shape == (256, 256)
                assert target256.shape == (256, 256)
                assert a2bAttached.shape == (256, 256 * 2)
                assert a2bSeparate.shape == (256, 256 * 2)

                # imshow('lig', a2bAttached)
                # waitKey(0)
                # imshow('ligcond', conditionAttached)
                # waitKey(0)

                # imshow('sep', a2bSeparate)
                # waitKey(0)
                # imshow('sepcond', conditionSeparate)
                # waitKey(0)

                # imshow('target', target256)
                # waitKey(0)
                # destroyAllWindows()

                # writing out
                writtenTrainAttached = imwrite(path.join(trainDirAttached, imName), a2bAttached)
                writtenTrainAttachedUnpaired = imwrite(path.join(trainDirAttachedUnpaired, imName), conditionAttached)

                writtenTrainSeparate = imwrite(path.join(trainDirSeparate, imName), a2bSeparate)
                writtenTrainSeparateUnpaired = imwrite(path.join(trainDirSeparateUnpaired, imName), conditionSeparate)

                writtenTargetUnpaired = imwrite(path.join(targetDirUnpaired, imName), target256)

                if not (writtenTrainSeparate, writtenTrainSeparateUnpaired, writtenTrainAttached,
                        writtenTrainAttachedUnpaired, writtenTargetUnpaired):
                    print(imName)
                    raise Exception('Not written\n')

                # freeing memory
                del a2bAttached, a2bSeparate, target, target256, conditionAttached, conditionSeparate

                if c % 50 == 0:
                    print(' -----  TRAINIG SET number of processed images: ', c)
                c += 1

            except SizeException as s:
                print(s)
                pass

        #############################
        #                           #
        #          TEST SET         #
        #                           #
        #############################

        print('\n\n#############################')
        print('TEST SET BUILDING')
        print('#############################\n')
        c = 1
        for imTest in imagesShuffled[trainingSetSize:]:
            try:
                imName = len2Image(imTest)

                # TARGET = manuscript/real image
                targetColors = wordsNColors[imTest]['col']
                targetBBxes = wordsNColors[imTest]['tks']
                tarhetImage = imread(color_words + '/' + imTest)
                target = getWord(tarhetImage, targetColors, targetBBxes)
                #         fitting the 256x256 format
                target256 = zeros((256, 256), dtype=uint8)
                hasBigChar = int(max([(tk[0][4] - meanHeight) * goesBelowLine(tk[1])
                                      for tk in targetBBxes] or [0.0]))
                yOff = 126 + round(3 - target.shape[0] + hasBigChar)
                xOff = round((256 - target.shape[1]) / 2)

                target256[yOff:yOff + target.shape[0], xOff:xOff + target.shape[1]] = \
                    bitwise_or(target256[yOff:yOff + target.shape[0], xOff:xOff + target.shape[1]], target)

                tokensList = [t[1] for t in targetBBxes]
                #
                # CONDITION = sketch/letter
                #       CONDITION 1 = minaces littera with ligatures
                char2Images, orederedComps = query(index, text="".join(tokensList))
                conditionAttached = createLetter(char2Images, orederedComps, toWhitePaper=False,
                                                 sketch=True)
                a2bAttached = hstack((target256, conditionAttached))
                #
                #       CONDITION 2 = minaces littera without ligatures
                char2Images, orederedComps = query(index, text=" ".join(tokensList),
                                                   forceHead=len(tokensList) > 1)
                conditionSeparate = createLetter(char2Images, orederedComps, toWhitePaper=False,
                                                 sketch=True, separate=True)
                a2bSeparate = hstack((target256, conditionSeparate))

                # check sizes
                assert conditionAttached.shape == (256, 256)
                assert conditionSeparate.shape == (256, 256)
                assert target256.shape == (256, 256)
                assert a2bAttached.shape == (256, 256 * 2)
                assert a2bSeparate.shape == (256, 256 * 2)

                # print(imName, '\n')
                # imshow(imName+' lig', a2bAttached)
                # waitKey(0)
                # imshow(imName+' ligcond', conditionAttached)
                # waitKey(0)
                #
                # imshow(imName+' sep', a2bSeparate)
                # waitKey(0)
                # imshow(imName + ' sepcond', conditionSeparate)
                # waitKey(0)
                #
                # imshow(imName+' target', target256)
                # waitKey(0)

                # destroyAllWindows()

                # writing out
                writtenTestAttached = imwrite(path.join(testDirAttached, imName), a2bAttached)
                writtenTestAttachedUnpaired = imwrite(path.join(testDirAttachedUnpaired, imName), conditionAttached)

                writtenTestSeparate = imwrite(path.join(testDirSeparate, imName), a2bSeparate)
                writtenTestSeparateUnpaired = imwrite(path.join(testDirSeparateUnpaired, imName), conditionSeparate)

                writtenTargetUnpaired = imwrite(path.join(testDirUnpaire, imName), target256)

                if not (writtenTestAttached, writtenTestAttachedUnpaired, writtenTestSeparate,
                        writtenTestSeparateUnpaired, writtenTargetUnpaired):
                    print(imName)
                    raise Exception('Not written\n')

                # freeing memory
                del a2bAttached, a2bSeparate, target, target256, conditionAttached, conditionSeparate

                if c % 50 == 0:
                    print(' -----  TEST SET number of processed images: ', c)
                c += 1

            except SizeException as s:
                print(s)
                pass


if __name__ == '__main__':
    from datetime import datetime

    print(datetime.now())
    datasetBuilderMinaceLetter()
    print(datetime.now())
