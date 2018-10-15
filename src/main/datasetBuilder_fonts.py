from cv2 import imread, IMREAD_GRAYSCALE, imwrite, bitwise_or, error, copyMakeBorder, BORDER_CONSTANT, imshow, waitKey, \
    destroyAllWindows
from json import load
from numpy import flip, concatenate
from config import fontsTokens, color_words, images2ColorsBBxesJSON, datasetDirFake, datasetDirOriginal, \
    datasetDirCombined, larger256
from os import path
from src.utils.imageProcessing import maskByColors, createBackground, scaleToBBXSize
from numpy import zeros



"""
                            DATASET

    - Real images: one directory containing all images.
                       Each image is a black backround white text png file representing original handwritten documents.
    - Fake (artificial) images: one directory containing all images.
                                Each image is a black backround white text png file representing the same text as above
                                but written with artificial tokens.
                                
                                
    These images comes from those contained in color_words, this stores images in subdirectory (page) in the form of:
                                                <page>/<image>.png
    
    Since we want just one directory:
                                     <page>/<image>.png => <page>___<image>.png
                                     
"""

raise Exception

# mapping each token to the image containing its artificial version
getArtifToken = {',': path.join(fontsTokens, 'comma.png'),
                 'U': path.join(fontsTokens, 'UU.png'),
                 'que': path.join(fontsTokens, 'que.png'),
                 'h': path.join(fontsTokens, 'h.png'),
                 'i': path.join(fontsTokens, 'i.png'),
                 'O': path.join(fontsTokens, 'OO.png'),
                 'm': path.join(fontsTokens, 'm.png'),
                 'A': path.join(fontsTokens, 'AA.png'),
                 'l': path.join(fontsTokens, 'l.png'),
                 'N': path.join(fontsTokens, 'NN.png'),
                 'curl': path.join(fontsTokens, 'curl.png'),  # "us", but if "b", ";" => "bus"!!!
                 'n': path.join(fontsTokens, 'n.png'),
                 'o': path.join(fontsTokens, 'o.png'),
                 'x': path.join(fontsTokens, 'x.png'),
                 'T': path.join(fontsTokens, 'TT.png'),
                 'fa': path.join(fontsTokens, 'fa.png'),
                 'R': path.join(fontsTokens, 'RR.png'),
                 'G': path.join(fontsTokens, 'GG.png'),
                 'ex': path.join(fontsTokens, 'ex.png'),
                 'H': path.join(fontsTokens, 'HH.png'),
                 'F': path.join(fontsTokens, 'FF.png'),
                 'I': path.join(fontsTokens, 'II.png'),
                 '.': path.join(fontsTokens, 'fullStop.png'),
                 '_s': path.join(fontsTokens, 'ending_s.png'),
                 'ending_s': path.join(fontsTokens, 'ending_s.png'),
                 'S': path.join(fontsTokens, 'SS.png'),
                 'si': path.join(fontsTokens, 'si.png'),
                 'gl': path.join(fontsTokens, 'gl.png'),
                 'fi': path.join(fontsTokens, 'fi.png'),
                 'E': path.join(fontsTokens, 'EE.png'),
                 'eg': path.join(fontsTokens, 'eg.png'),
                 'rum': path.join(fontsTokens, 'rum.png'),
                 'prop': path.join(fontsTokens, 'prop.png'),
                 'P': path.join(fontsTokens, 'PP.png'),
                 'Q': path.join(fontsTokens, 'QQ.png'),
                 'de': path.join(fontsTokens, 'de.png'),
                 'D': path.join(fontsTokens, 'DD.png'),
                 'pa': path.join(fontsTokens, 'pa.png'),
                 'u': path.join(fontsTokens, 'u.png'),
                 'b': path.join(fontsTokens, 'b.png'),
                 'c': path.join(fontsTokens, 'c.png'),
                 'B': path.join(fontsTokens, 'BB.png'),
                 't': path.join(fontsTokens, 't.png'),
                 'M': path.join(fontsTokens, 'MM.png'),
                 ';': path.join(fontsTokens, 'semicolon.png'),  # attention curl is 'us' too!
                 'ue': path.join(fontsTokens, 'semicolon.png'),  # q;
                 'semicolon': path.join(fontsTokens, 'semicolon.png'),
                 'nt': path.join(fontsTokens, 'nt.png'),
                 'a': path.join(fontsTokens, 'a.png'),
                 'epo': path.join(fontsTokens, 'epo.png'),
                 'ce': path.join(fontsTokens, 'ce.png'),
                 'qui': path.join(fontsTokens, 'qui.png'),
                 'd': path.join(fontsTokens, 'd.png'),
                 's': path.join(fontsTokens, 's.png'),
                 'Ca': path.join(fontsTokens, 'CCa.png'),
                 'ca': path.join(fontsTokens, 'ca.png'),
                 'per': path.join(fontsTokens, 'per.png'),
                 'r': path.join(fontsTokens, 'r.png'),
                 'e': path.join(fontsTokens, 'e.png'),
                 'g': path.join(fontsTokens, 'g.png'),
                 'p': path.join(fontsTokens, 'p.png'),
                 'pro': path.join(fontsTokens, 'pro.png'),
                 'C': path.join(fontsTokens, 'CC.png'),
                 'q': path.join(fontsTokens, 'q.png'),
                 'f': path.join(fontsTokens, 'f.png'),
                 'L': path.join(fontsTokens, 'LL.png'),
                 'con': path.join(fontsTokens, 'con.png')
                 }

tooBig = larger256




def buildDataset():
    """
    Filling the directories with real and fake/aritificial images
    :return: None
    """

    with open(images2ColorsBBxesJSON, 'r') as f:
        img2ColsBB = load(f)
        c = 0

        # browsing every directory and image
        for imagePath, _values in img2ColsBB.items():
            if imagePath not in tooBig:

                pageDir, image = imagePath.split('/')
                c += 1

                if pageDir[0].isnumeric():
                    tokens = _values['tks']
                    colorsBGR = []

                    # colors
                    for i, t in enumerate(tokens):
                        _token = t[1]
                        try:
                            if _token not in _values['col']:
                                if _token.isupper():
                                    _token = _token.lower()
                                # double
                                elif len(_token) == 2:
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
                        #                                           area
                        elif t == 's' and ind == len(tokens) - 1 and bbx[2] >= 185.0555041628122:
                            t = 'ending_s'

                        fakeTokenImage = zeros(realImg.shape)

                        try:
                            if len(t) > 1 and t[1] == t[0]:  # double
                                fakeTokenImage = imread(getArtifToken[t[0]], IMREAD_GRAYSCALE)
                                fakeTokenImage = concatenate((fakeTokenImage, fakeTokenImage), axis=1)
                            elif len(t) > 1 and t[1] == t[0] == t[2]:
                                fakeTokenImage = imread(getArtifToken[t[0]], IMREAD_GRAYSCALE)
                                fakeTokenImage = concatenate((fakeTokenImage, fakeTokenImage, fakeTokenImage), axis=1)
                            else:
                                fakeTokenImage = imread(getArtifToken[t], IMREAD_GRAYSCALE)
                        except KeyError:
                            print('not in getArtifToken', t, image)
                            break

                        # scaling
                        fakeTokenImage = scaleToBBXSize(fakeTokenImage, bbx)

                        y, dy = bbx[7:]
                        x, dx = bbx[5:7]

                        try:
                            fakeImg[y: dy, x: dx] = bitwise_or(fakeImg[y: dy, x: dx], fakeTokenImage)
                        except ValueError as v:
                            print(v, '\n', pageDir, image)
                            break

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
                    combinedImg = concatenate((realImg, fakeImg), axis=1)

                    assert realImg.shape == (256, 256)
                    assert fakeImg.shape == (256, 256)
                    assert combinedImg.shape == (256, 256 * 2)

                    # write out
                    imgName = pageDir + '___' + image
                    outOriginalFile = path.join(datasetDirOriginal, imgName)
                    outFakeFile = path.join(datasetDirFake, imgName)
                    combinedFile = path.join(datasetDirCombined, imgName)


                    """
                    origWriteStatus = imwrite(outOriginalFile, realImg)
                    fakeWriteStatus = imwrite(outFakeFile, fakeImg)
                    combWriteStatus = imwrite(combinedFile, combinedImg)

                    assert origWriteStatus
                    assert fakeWriteStatus
                    assert combWriteStatus
                    """

    print('\n\nTotal number of images procecessed: ', c)
    print('original directory {}'.format(datasetDirOriginal))
    print('fake directory {}'.format(datasetDirFake))
    print('combined directory {}'.format(datasetDirCombined))


if __name__ == '__main__':
    buildDataset()
