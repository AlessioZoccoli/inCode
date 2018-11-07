from os import path
from cv2 import imread, IMREAD_GRAYSCALE, copyMakeBorder, BORDER_CONSTANT, bitwise_or, imshow, waitKey, \
    destroyAllWindows, error, imwrite
from numpy import flip, zeros, hstack, concatenate
from config import sketchTokens, color_words, images2ColorsBBxesJSON, larger256, trainDir, testDir, valDir, wordsSimple
from json import load
from random import sample, randint
from src.utils.imageProcessing import maskByColors, createBackground, scaleToBBXSize


getArtifToken = {',': [path.join(sketchTokens, 'comma.png')],
                 'U': [path.join(sketchTokens, 'UU.png')],
                 'que': [path.join(sketchTokens, 'q.png'), path.join(sketchTokens, 'semicolon.png')],
                 'h': [path.join(sketchTokens, 'h.png')],
                 'i': [path.join(sketchTokens, 'i.png')],
                 'O': [path.join(sketchTokens, 'OO.png')],
                 'od': [path.join(sketchTokens, 'o.png'), path.join(sketchTokens, 'd.png')],
                 'm': [path.join(sketchTokens, 'm.png')],
                 'A': [path.join(sketchTokens, 'AA.png')],
                 'l': [path.join(sketchTokens, 'l.png')],
                 'N': [path.join(sketchTokens, 'NN.png')],
                 'curl': [path.join(sketchTokens, 'curl.png')],  # "us", but if "b", ";" => "bus"!!!
                 'n': [path.join(sketchTokens, 'n.png')],
                 'o': [path.join(sketchTokens, 'o.png')],
                 'x': [path.join(sketchTokens, 'x.png')],
                 'T': [path.join(sketchTokens, 'TT.png')],
                 'fa': [path.join(sketchTokens, 'f.png'), path.join(sketchTokens, 'a.png')],
                 'R': [path.join(sketchTokens, 'RR.png')],
                 'G': [path.join(sketchTokens, 'GG.png')],
                 'ex': [path.join(sketchTokens, 'e.png'), path.join(sketchTokens, 'x.png')],
                 'es': [path.join(sketchTokens, 'e.png'), path.join(sketchTokens, 's.png')],
                 'ess': [path.join(sketchTokens, 'e.png'), path.join(sketchTokens, 's.png'),
                         path.join(sketchTokens, 's.png')],
                 'H': [path.join(sketchTokens, 'HH.png')],
                 'F': [path.join(sketchTokens, 'FF.png')],
                 'I': [path.join(sketchTokens, 'II.png')],
                 '.': [path.join(sketchTokens, 'point.png')],
                 '_s': [path.join(sketchTokens, '_s.png')],
                 'S': [path.join(sketchTokens, 'SS.png')],
                 'si': [path.join(sketchTokens, 's.png'), path.join(sketchTokens, 'i.png')],
                 'gl': [path.join(sketchTokens, 'g.png'), path.join(sketchTokens, 'l.png')],
                 'fi': [path.join(sketchTokens, 'f.png'), path.join(sketchTokens, 'i.png')],
                 'E': [path.join(sketchTokens, 'EE.png')],
                 'eg': [path.join(sketchTokens, 'e.png'), path.join(sketchTokens, 'g.png')],
                 'rum': [path.join(sketchTokens, 'rum.png')],
                 'prop': [path.join(sketchTokens, 'pro.png'), path.join(sketchTokens, 'p.png')],
                 'P': [path.join(sketchTokens, 'PP.png')],
                 'Q': [path.join(sketchTokens, 'QQ.png')],
                 'Qd': [path.join(sketchTokens, 'QQ.png'), path.join(sketchTokens, 'd.png')],
                 'de': [path.join(sketchTokens, 'd.png'), path.join(sketchTokens, 'e.png')],
                 'D': [path.join(sketchTokens, 'DD.png')],
                 'pa': [path.join(sketchTokens, 'p.png'), path.join(sketchTokens, 'a.png')],
                 'u': [path.join(sketchTokens, 'u.png')],
                 'b': [path.join(sketchTokens, 'b.png')],
                 'bprob': [path.join(sketchTokens, 'b.png'), path.join(sketchTokens, 'pro.png'),
                           path.join(sketchTokens, 'b.png')],
                 'c': [path.join(sketchTokens, 'c.png')],
                 'cu': [path.join(sketchTokens, 'c.png'), path.join(sketchTokens, 'u.png')],
                 'cus': [path.join(sketchTokens, 'c.png'), path.join(sketchTokens, 'curl.png')],
                 'ci': [path.join(sketchTokens, 'c.png'), path.join(sketchTokens, 'i.png')],
                 'B': [path.join(sketchTokens, 'BB.png')],
                 't': [path.join(sketchTokens, 't.png')],
                 'ta': [path.join(sketchTokens, 't.png'), path.join(sketchTokens, 'a.png')],
                 'tus': [path.join(sketchTokens, 't.png'), path.join(sketchTokens, 'curl.png')],
                 'M': [path.join(sketchTokens, 'MM.png')],
                 ';': [path.join(sketchTokens, 'semicolon.png')],  # attention curl is 'us' too!
                 'ue': [path.join(sketchTokens, 'semicolon.png')],  # q;
                 'ui': [path.join(sketchTokens, 'u.png'), path.join(sketchTokens, 'i.png')],
                 'iu': [path.join(sketchTokens, 'i.png'), path.join(sketchTokens, 'u.png')],
                 'ir': [path.join(sketchTokens, 'i.png'), path.join(sketchTokens, 'r.png')],
                 'it': [path.join(sketchTokens, 'i.png'), path.join(sketchTokens, 't.png')],
                 'semicolon': [path.join(sketchTokens, 'semicolon.png')],
                 'nt': [path.join(sketchTokens, 'nt.png')],
                 'ni': [path.join(sketchTokens, 'n.png'), path.join(sketchTokens, 'i.png')],
                 'a': [path.join(sketchTokens, 'a.png')],
                 'epo': [path.join(sketchTokens, 'e.png'), path.join(sketchTokens, 'p.png'),
                         path.join(sketchTokens, 'o.png')],
                 'ce': [path.join(sketchTokens, 'c.png'), path.join(sketchTokens, 'e.png')],
                 # TODO VERIFY
                 'qui': [path.join(sketchTokens, 'q.png'), path.join(sketchTokens, 'u.png'), path.join(sketchTokens, 'i.png')],
                 'd': [path.join(sketchTokens, 'd.png')],
                 's': [path.join(sketchTokens, 's.png')],
                 'Ca': [path.join(sketchTokens, 'CC.png'), path.join(sketchTokens, 'a.png')],
                 'ca': [path.join(sketchTokens, 'c.png'), path.join(sketchTokens, 'a.png')],
                 'per': [path.join(sketchTokens, 'per.png')],
                 'r': [path.join(sketchTokens, 'r.png')],
                 'e': [path.join(sketchTokens, 'e.png')],
                 'ee': [path.join(sketchTokens, 'e.png'), path.join(sketchTokens, 'e.png')],
                 'eee': [path.join(sketchTokens, 'e.png'), path.join(sketchTokens, 'e.png'),
                         path.join(sketchTokens, 'e.png')],
                 'ec': [path.join(sketchTokens, 'e.png'), path.join(sketchTokens, 'c.png')],
                 'ecc': [path.join(sketchTokens, 'e.png'), path.join(sketchTokens, 'c.png'),
                         path.join(sketchTokens, 'c.png')],
                 'g': [path.join(sketchTokens, 'g.png')],
                 'p': [path.join(sketchTokens, 'p.png')],
                 'pro': [path.join(sketchTokens, 'pro.png')],
                 'C': [path.join(sketchTokens, 'CC.png')],
                 'q': [path.join(sketchTokens, 'q.png')],
                 'f': [path.join(sketchTokens, 'f.png')],
                 'L': [path.join(sketchTokens, 'LL.png')],
                 'con': [path.join(sketchTokens, 'con.png')]
                 }


def buildDataset(trainSetSize=0.8, testSetSize=0.1, curriculum=True):
    """
    Builds the dataset in the form of a set of images containing original manuscript on the left and artificial
    images on the right

    :param trainSetSize: float. Percentage of the images to insert into the training set
    :param testSetSize: float. Percentage.
    :param curriculum: boolen. Use Curriculum Learning.
                Simple datapoints in the training set are presented/learnt first. The complexity of the data point is
                given by the length of the color word contained in it.
    :return: None
    """
    assert trainSetSize + testSetSize <= 1.0

    tooBig = larger256

    with open(images2ColorsBBxesJSON, 'r') as f, open(wordsSimple, 'r') as fws:
        wordsRich = load(fws)  # contains s_alta, s_mediana, s_ending
        img2ColsBB = load(f)

        #
        #   building training/test/validation sets
        #
        totWords = 10504 - len(tooBig)
        trainSetSize = int(totWords * trainSetSize)
        testSetSize = int(totWords * testSetSize)
        valSetSize = totWords - trainSetSize - testSetSize

        # randomizing keys
        # not necessary with sorting ?!
        totImages = sample(img2ColsBB.keys(), totWords)

        if curriculum:
            print('\n cv enabled \n')
            # Curriculum learning: increasing difficul sample to learn.
            #       We assume short words easier to learn.
            totImages = sorted(totImages, key=lambda tw: len(img2ColsBB[tw]['tks']))

        trainSet = totImages[:trainSetSize]

        if curriculum:
            tempSet = totImages[trainSetSize:]
            tempSet = sample(tempSet, len(tempSet))
            testSet = tempSet[:testSetSize]
            valSet = tempSet[testSetSize:]
        else:
            testSet = totImages[trainSetSize:trainSetSize+testSetSize]
            valSet = totImages[trainSetSize+testSetSize:]

        assert len(trainSet) == trainSetSize
        assert len(testSet) == testSetSize
        assert len(valSet) == valSetSize

        c = 0

        # browsing every directory and image
        for imagePath, _values in img2ColsBB.items():
            if imagePath not in tooBig:

                pageDir, image = imagePath.split('/')
                c += 1

                if pageDir[0].isnumeric():
                    tokens = _values['tks']
                    colorsBGR = []

                    for i, t in enumerate(tokens):
                        _token = t[1]
                        # colors
                        try:
                            if _token not in _values['col']:
                                if _token.isupper():
                                    _token = _token.lower()
                                # double
                                elif len(_token) in (2, 3):
                                    if _token[1] == _token[0]:
                                        _token = _token[0]
                                elif _token is ".":  # just in case '.' color hasn't been annotated
                                    _token = min(_values['tks'], key=lambda el: el[0][2])[1]

                        except IndexError:
                            print('IndexError ', pageDir, image, _token, len(_token))
                            break
                        try:
                            colorsBGR.extend(_values['col'][_token])
                        except KeyError:
                            print('ERROR in: ', _token, imagePath)
                            print('COLORS:   ', _values['col'].keys())
                            break
                    try:
                        colorsBGR = flip(colorsBGR, axis=1)
                    except ValueError:
                        print('color flip error: ', colorsBGR, colorsBGR.shape)
                        break

                    colorWord = imread(path.join(color_words, imagePath))
                    realImg = maskByColors(colorWord, colorsBGR)

                    # creating the fake image
                    w, h = realImg.shape[:2][::-1]
                    fakeImg = createBackground(width=w, height=h)

                    # enforcing ligatures
                    lastX = 0

                    for ind, ts in enumerate(tokens):
                        bbx, t = ts
                        # curl and semicolon
                        if t == 'us':
                            if tokens[ind - 1][1] != 'b' or len(tokens) == 1:
                                t = 'curl'
                            else:
                                t = 'semicolon'
                        elif t == 'ue' and ind > 0 and tokens[ind - 1][1] == 'q':
                            t = 'semicolon'
                        # ending
                        elif t[-1] == 's' and ind == len(tokens) - 1\
                                and wordsRich[imagePath][-1][1] in {'s_mediana', 's_ending'}:
                            t = '_s'

                        fakeTokenImage = zeros(realImg.shape)

                        try:
                            if len(t) > 1 and t[1] == t[0] == t[2]:
                                fakeTokenImage = imread(getArtifToken[t[0]][0], IMREAD_GRAYSCALE)
                                fakeTokenImage = concatenate((fakeTokenImage, fakeTokenImage, fakeTokenImage), axis=1)
                            elif len(t) > 1 and t[1] == t[0]:  # double
                                fakeTokenImage = imread(getArtifToken[t[0]][0], IMREAD_GRAYSCALE)
                                fakeTokenImage = concatenate((fakeTokenImage, fakeTokenImage), axis=1)
                            else:
                                s_end = False
                                if ind == len(tokens)-1 and t[-1] == 's' and wordsRich[imagePath][-1][1] in {'s_mediana', 's_ending'}:
                                    s_end = True
                                _artifTk = getArtifToken[t]
                                fakeTokenImage = imread(_artifTk[0], IMREAD_GRAYSCALE)

                                if len(_artifTk) > 1:
                                    subTokens = [imread(s, IMREAD_GRAYSCALE) for s in _artifTk]

                                    if s_end:
                                        subTokens[-1] = imread(getArtifToken['_s'], IMREAD_GRAYSCALE)

                                    maxHeight = max([sk.shape[0] for sk in subTokens])
                                    firstST = subTokens[0]
                                    if firstST.shape[0] == maxHeight:
                                        fakeTokenImage = firstST
                                    else:
                                        topPad = int((maxHeight-firstST.shape[0]) / 2)
                                        bottomPad = maxHeight - topPad - firstST.shape[0]
                                        fakeTokenImage = copyMakeBorder(fakeTokenImage, topPad, bottomPad, 0, 0,
                                                                        BORDER_CONSTANT, value=[0, 0, 0])
                                    for st in subTokens[1:]:
                                        topPad = int((maxHeight - st.shape[0]) / 2)
                                        bottomPad = maxHeight - topPad - st.shape[0]
                                        st = copyMakeBorder(st, topPad, bottomPad, 0, 0, BORDER_CONSTANT,
                                                            value=[0, 0, 0])
                                        fakeTokenImage = concatenate((fakeTokenImage, st), axis=1)

                        except KeyError:
                            print('not in getArtifToken', t, image)
                            break

                        # scaling
                        fakeTokenImage = scaleToBBXSize(fakeTokenImage, bbx)

                        # coordinates
                        y, dy = bbx[7:]
                        x, dx = bbx[5:7]


                        if ind > 1 and x-lastX < 10 and t[0] != 'g':
                            lb, ub = (2, 6) if ts == tokens[ind-1][1] else (0, 4)
                            randomBackOffset = randint(lb, ub)
                            while x - randomBackOffset < 0:
                                randomBackOffset = randint(0, x)
                            x -= randomBackOffset
                            dx -= randomBackOffset

                        # previous token updated
                        lastX = dx

                        try:
                            fakeImg[y:dy, x:dx] = bitwise_or(fakeImg[y:dy, x:dx], fakeTokenImage)
                        except ValueError as v:
                            print(v, '\n', pageDir, image)
                            break
                        except error:
                            print(fakeImg[y:dy, x:dx].shape, fakeTokenImage.shape)
                            print('x: ', x, 'y: ', y)
                            print('fakeImg shape ', fakeImg.shape)
                            print(dx > fakeImg.shape[1], dy > fakeImg.shape[0])
                            print(ts, imagePath)
                            return None

                    # to 256x256
                    topPad, bottomPad, leftPad, rightPad = 0, 0, 0, 0

                    # bigger then 256
                    if realImg.shape[0] > 256:
                        print('realImg height > 256   {}/{}'.format(pageDir, image))
                        realImg = realImg[0:255, 0:realImg.shape[1]]
                    if realImg.shape[1] > 256:
                        print('realImg width > 256   {}/{}'.format(pageDir, image))
                        realImg = realImg[0:realImg.shape[0], 0:255]

                    # smaller then 256
                    if realImg.shape[0] < 256:
                        topPad = int((256 - h) / 2)
                        bottomPad = 256 - topPad - h
                    if realImg.shape[1] < 256:
                        leftPad = int((256 - w) / 2)
                        rightPad = 256 - leftPad - w

                    realImg = copyMakeBorder(realImg, topPad, bottomPad, leftPad, rightPad, BORDER_CONSTANT,
                                             value=[0, 0, 0])
                    fakeImg = copyMakeBorder(fakeImg, topPad, bottomPad, leftPad, rightPad, BORDER_CONSTANT,
                                             value=[0, 0, 0])

                    # pairing real and fake
                    combinedImg = hstack((realImg, fakeImg))

                    assert realImg.shape == (256, 256)
                    assert fakeImg.shape == (256, 256)
                    assert combinedImg.shape == (256, 256 * 2)

                    # write out
                    imgName = pageDir + '#' + image
                    if curriculum:
                        imgName = str(len(tokens)) + '##' + imgName

                    if imagePath in trainSet:
                        status = imwrite(path.join(trainDir, imgName), combinedImg)
                        assert status
                    elif imagePath in testSet:
                        status = imwrite(path.join(testDir, imgName), combinedImg)
                        assert status
                    elif imagePath in valSet:
                        status = imwrite(path.join(valDir, imgName), combinedImg)
                        assert status

    print('processed {} images'.format(c))
    print('trainDir :', trainDir)
    print('testDir: ', testDir)
    print('valDir: ', valDir)


if __name__ == '__main__':
    buildDataset()
