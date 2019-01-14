from json import load
from os import path
from pprint import pprint
from random import randint
from math import e
from cv2 import imread, bitwise_or, error
from numpy import flip, uint8, invert, zeros_like, zeros

from config import images2ColorsBBxesJSON, color_words
from src.utils.imageProcessing import extractComponent, createBackground


richTokensConv = {
            's_mediana': '1',
            's_ending': '2',
            'd_stroke': '3',
            'l_stroke': '4',
            'b_stroke': '5',
            'curl': '6',
            'qui': '7',
            'con': '8',
            'nt': '9',
            'prop': '/',
            'pro': '$',
            'per': '%',
            'semicolon': '&'
        }


def goesBelowLine(evalString):
    """
    Adjusting vertical offset for chars with tipically long size
    :param evalString: str. String to be evaluated
    :param h: bool. Char 'h' is present in evalString? 'h' must be treaten separately
    :return: int. Final vertical offset.
    """
    below = 0
    h = 1
    if 'h' in evalString:
        below = 0.6 * e ** -(h / 25)
    elif len(set(evalString).intersection({'p', 'q', 'g', 'p', 'i', '1', '2', '7', '$', '%', 'x', '&', '/', '('})) > 0:  # 1 = s_med, 2 = s_end
        below = 1
    return below


# mean width of a 1 char long tokens
meanWidth = 17.30232622312978
meanHeight = 17.274067142355833  # see goesBelowLine
widthBackground = 1400
heightBackground = 1900
# maxHeight = 69
# maxWidth = 698


class SizeException(Exception):
    """
    each image must be 256x256 and when paired 256x512
    """
    pass


def createLetter(ccToTokens, phrase, toWhitePaper=True, vertical=False, width=widthBackground, height=heightBackground, showConComps=False, is256=False, separate=False):
    """
    Creates the menace letter from a given phrase and ccToTokens (data related to phrase)

    :param ccToTokens: dict.
            {
            'hello': ['dir1/image1.png', ['h', 'e', 'll', 'o']],
            'world': ['dirN/imageN.png', ['w', 'o', 'r', 'l', 'd']]
            ]
            Beign a standard dict, ccToTokens doesn't preserve the ordering of the original text.
    :param phrase: list. ['h', 'e', 'll', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
                    This list gives the ordering of the input phrase, preserving spaces from original text.
    :param toWhitePaper: bool. True if output will be black text on white background, False vice versa.
    :param vertical: True if the output letter is a vertical list
    :param width: int. Width of the letter
    :param height: int. Height of the letter
    :param showConComps: bool. Highlights each different bounding box
    :param is256: bool. Output will fit a 256x256 image and will center the string
    :param separate: bool. Output is a menace letter without legature/connected components,
                    just single tokens separataed by space
    :return: (height,width)-sized image representing the menace letter
    """
    if is256:
        width, height = 256, 256

    with open(images2ColorsBBxesJSON, 'r') as ann:
        ch2col = load(ann)

    patches = []

    for subString in phrase:
        if subString != ' ':
            subStringImg, tokens = ccToTokens[subString][0], ccToTokens[subString][1]

            xStart, xEnd = [], []
            yStart, yEnd = [], []
            hasBigChar = 0.0
            groupedCols = []

            # coordinates. This step consists in applying a mask on
            #  <ch2col[subStringImg]['tks']> to fit <tokens>
            found = False
            currStart = 0
            # all tokens in subStringImg
            transcribed = ch2col[subStringImg]['tks']

            while not found and currStart <= len(transcribed) - len(tokens):
                window = transcribed[currStart:currStart + len(tokens)]
                windowChar = [tr[1] for tr in window]
                if tokens == windowChar:
                    found = True
                    xStart = [wxs[0][5] for wxs in window]
                    xEnd = [wxe[0][6] for wxe in window]
                    yStart = [tk[0][7] for tk in window]
                    yEnd = [tk[0][8] for tk in window]
                    hasBigChar = int(max([(tk[0][4] - meanHeight)*goesBelowLine(tk[1])
                                          for tk in window] or [0.0]))   # if goesBelowLine(tk[1]) > 0
                else:
                    currStart += 1

            # colors. Colors may not be indexed with the same keys as bbxes
            for t in tokens:
                _token = t
                try:
                    if t not in ch2col[subStringImg]['col']:
                        if t.isupper():
                            _token = t.lower()
                        elif t[1] == t[0]:
                            _token = t[0]
                        elif t in ('us', 'ue'):
                            _token = 'semicolon'
                        elif t is ".":                          # just in case '.' color hasn't been annotated
                            _token = min(ch2col[subStringImg]['tks'], key=lambda e: e[0][2])[1]
                except IndexError:
                    pass

                if _token:
                    try:
                        groupedCols.append(flip(ch2col[subStringImg]['col'][_token], axis=1))
                    except KeyError:
                        print("Check for token and color to be properly mapped in: ", subStringImg, images2ColorsBBxesJSON)
                        print(ch2col[subStringImg]['col'].keys(), [x[1] for x in ch2col[subStringImg]['tks']], _token)
                        print("ccToTokens ---->  ", ccToTokens[subString], '\n')
                        return None
                    except ValueError as v:
                        print(v, 'n', groupedCols)
                        return None

            try:
                assert len(groupedCols) == len(xStart)
                assert len(groupedCols) == len(yStart)
                assert len(groupedCols) == len(xEnd)
                assert len(groupedCols) == len(yEnd)
            except AssertionError:
                print('GROUPED COLORS: ', len(groupedCols), len(xStart), len(yStart), len(yEnd), len(xEnd), subStringImg, tokens)
                return

            sortedXStart = sorted(xStart)
            sortedYStart = sorted(yStart)
            sortedXEnd = sorted(xEnd)
            sortedYEnd = sorted(yEnd)
            minYStart, maxYStart = sortedYStart[0], sortedYStart[-1]
            minXStart, maxXStart = sortedXStart[0], sortedXStart[-1]
            minYEnd, maxYEnd = sortedYEnd[0], sortedYEnd[-1]
            minXEnd, maxXEnd = sortedXEnd[0], sortedXEnd[-1]

            img = imread(path.join(color_words, subStringImg))
            patch = zeros(img.shape[:2], dtype=uint8)

            # this extracts requested components from target image and puts them side-by-side inside patch
            for idx, gc in enumerate(groupedCols):
                # coords for target image
                _xstart = xStart[idx]
                _xend = xEnd[idx]
                _ystart = yStart[idx]
                _yend = yEnd[idx]

                # guard for sketches
                if is256 and (_xend - _xstart > 256 or _yend - _ystart > 256):
                    raise SizeException('is256 but comp width {}, height {}! \n', _xend-_xstart, _yend-_ystart)
                # new connected component to be inserted
                newComp = extractComponent(img, gc, _xstart, _xend, _ystart, _yend)
                try:
                    patch[_ystart:_yend, _xstart:_xend] = bitwise_or(patch[_ystart:_yend, _xstart:_xend], newComp)
                except error:
                    print(_yend-_ystart, _xend-_xstart)
                    print('  patch[]: ', patch[_ystart:_yend, _xstart:_xend].shape, '  newComp: ', newComp.shape)
                    return

            try:
                patch = patch[minYStart:maxYEnd, minXStart: maxXEnd]
            except TypeError:
                print('Wrong Type(s) ', minYStart, maxYEnd, minXStart, maxXEnd)
                return
            if showConComps:
                patch = invert(patch)*randint(20, 190)
            patches.append((patch, hasBigChar))

        # sustring is space
        else:
            if not is256:
                spaceDim = (randint(3, 5), round(meanHeight))
                space = zeros(spaceDim, dtype=uint8)
                patches.append((space, 0.0))

    interLine = None
    if is256:
        # this is a special additive horizontal offset and is used in case it is is256 mode
        # since a is256 is a one-liner line*lineOffset is not suitable
        interLine = 3
    elif height < heightBackground:
        interLine = min([p[0].shape[0] for p in patches])

    # for sketches: if word is longer than 256 discard, otherwise take it and center it inside the image
    interTokenSpace = 0
    if is256:
        interTokenSpace = 5 if separate else 0
        wordWidth = sum([p[0].shape[1] for p in patches]) + (len(patches)-1)*interTokenSpace
        if wordWidth > 256:
            raise SizeException('is256 but {} > 256'.format(wordWidth))

    #
    #   Creating the image
    #
    def createLetterImage(heightPX=height, widthPX=width, interLineOffsetFixed=interLine, tokensSpace=interTokenSpace):
        """
        Creates the final image
        :param heightPX: int. Height
        :param widthPX: int. Width
        :param interLineOffsetFixed: distance between line is fixed if is256 (only one line is needed)
        :param separe: bool. Connected components or single tokens
        :param tokensSpace: int. space bitwin token (if is256)
        :return: np.ndarray of shape (heightPX, widthPX)
        """
        back = createBackground(heightPX, widthPX)
        # xOff is the last x of the previous token, xOffRand adds a some random pixels of space
        if widthPX >= widthBackground:
            xOff, xOffRand, lineOffset = 40, 0, 0
            # right space of the sheet
            rightBound = 45
        elif is256:
            xOff = round((256 - wordWidth)/2)
            # right space of the sheet
            rightBound = 0
            lineOffset = round((256 - interLineOffsetFixed)/2)
        else:
            xOff, xOffRand, lineOffset = 0, 0, 0
            # right space of the sheet
            rightBound = 0
        line = 0

        hasBlankSpace = True if len(phrase) == 1 or phrase[1] == ' ' else False

        for indx, patch2hbc in enumerate(patches):
            # patch, hasBigChar "penalty"
            ptch, hbc = patch2hbc
            try:
                previousIsSpace = (phrase[indx - 1] == ' ')
            except IndexError:
                previousIsSpace = False

            # random proximity to the last taken component
            # randProximity = 0 if xOff == 0 or previousIsSpace else (randint(0, 4) if phrase[idx][0] != 'g' else 10)
            randProximity = 0
            if not is256:
                if xOff == 0:
                    randProximity = 0
                elif previousIsSpace:
                    randProximity = 0 if phrase[idx][0] != 'g' else 2
                elif phrase[indx][0] != 'g':
                    if indx > 0 and phrase[indx-1][0] in {'s', 'f'}:
                        randProximity = randint(0, 1)
                    else:
                        randProximity = randint(0, 4)
                elif 'g' in {phrase[indx][0], phrase[indx][0]}:
                    randProximity = round(ptch.shape[1]/3) - randint(1, 3)

            interLineOffset = int(90 if line == 0 else 70) if not interLineOffsetFixed else interLineOffsetFixed
            if not is256:
                lineOffset = line * interLineOffset

            hasCurlMultip = 2 if hasBlankSpace and 2*indx < len(phrase) else 1
            yOff = lineOffset + round(interLineOffset - ptch.shape[0] + hbc) -\
                   (10 if phrase[indx*hasCurlMultip] == '6' else 0)
            xOffRand = xOff - randProximity

            dy = yOff + ptch.shape[0]
            dx = xOffRand + ptch.shape[1]

            try:
                _next = patches[indx + 1][0].shape[1] if phrase[indx + 1] != ' ' else 0
            except IndexError:
                _next = 0
            if dx + _next <= width - rightBound:
                back[yOff:dy, xOffRand:dx] = bitwise_or(back[yOff:dy, xOffRand:dx], ptch)
                xOff += ptch.shape[1]
                if is256 and tokensSpace > 0 and xOff+tokensSpace < width - rightBound:
                    xOff += tokensSpace

            elif dy < height - interLineOffset and not is256:
                print('newline ', line, ' ### yOff {}'.format(yOff))
                line += 1
                bigCharOffset = round(interLineOffset - ptch.shape[0] + hbc)
                yOff = line * interLineOffset + bigCharOffset
                xOff = 40
                xOffRand = xOff + randProximity
                dy = yOff + ptch.shape[0]
                dx = xOffRand + ptch.shape[1]
                back[yOff:dy, xOffRand:dx] = bitwise_or(back[yOff:dy, xOffRand:dx], ptch)
                xOff += ptch.shape[1]
            else:
                break

        assert back.shape == (widthPX, heightPX)
        return back


    def createVerticalImg(heightPX=heightBackground, widthPX=widthBackground):
        back = createBackground(heightPX, widthPX)
        yOff = 5
        for ix, patch2hbc in enumerate(patches):
            # patch, hasBigChar "penalty"
            ptch, hbc = patch2hbc
            punctOff = 5 if phrase[ix] in (',', '.', ';') else 0       # hint to separate punctuation from chars
            yOff += punctOff
            dy = yOff + ptch.shape[0]
            back[yOff:dy, 40:40+ptch.shape[1]] = ptch
            yOff += ptch.shape[0] + 5

        print('width ', widthPX, ' height ', heightPX)
        return back


    outImg = createVerticalImg() if vertical else createLetterImage()
    # print(outImg.shape)

    if toWhitePaper:
        return invert(outImg)
    return outImg
