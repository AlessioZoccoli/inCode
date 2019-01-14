import cv2
import numpy as np
from base64 import b64encode

from cv2 import bitwise_or
from numpy import uint8, flip
from numpy.core.multiarray import zeros


def maskByColors(word_img, colors):
    """
        funzione per creare una maschera su un'immagine,
        a partire da una lista di colori.
        :param word_img: np.ndarray dtype = uint8. shape = (height,width,channels)
        :param colors: np.ndarray dtype = uint8. Each element in the form [blue,green,red]
        :return: mask of shape (h,w) on the original one, which takes the value of 0 if colors are not present,
                255 otherwise.
    """
    mask = np.zeros((word_img.shape[0], word_img.shape[1]), dtype='uint8')

    for c in colors:
        colormask = cv2.inRange(word_img, lowerb=c, upperb=c)
        mask = cv2.bitwise_or(mask, colormask)

    return mask


def maskBlackWhite(word_img):
    """

    :param word_img: np.ndarray dtype = uint8. shape = (height,width,channels)
    :return: mask of shape (h,w) on the original one, which takes the value of 255 if there is any non white color,
                0 otherwise.
    """
    mask = np.zeros((word_img.shape[0], word_img.shape[1]), dtype='uint8')
    BLACK = np.array([0, 0, 0], dtype=np.uint8)
    NOTWHITE = np.array([244, 244, 244], dtype=np.uint8)
    colormask = cv2.inRange(word_img, lowerb=BLACK, upperb=NOTWHITE)
    mask = cv2.bitwise_or(mask, colormask)

    return mask


def findAllColors(img):
    """
        elenca tutti i colori di un'immagine.

        parametri:
            - img:
                numpy.ndarray di tipo uint8, shape (height,width,channels)

        return:
            una lista di valori [b,g,r], corrispondenti ai colori presenti
            nell'immagine.
    """
    hist = cv2.calcHist([img], [0, 1, 2], None, [256] * 3, [0, 256] * 3)
    allColors = np.argwhere(hist != 0)
    return allColors


def sortedBBxs(img):
    """
        elenca tutte le bounding box delle componenti connesse di un'immagine,
        ordinate dalla più piccola alla più grande.

        parametri:
            - img:
                numpy.ndarray di tipo uint8, shape (height,width). L'immagine deve
                essere binaria (0 = assenza di inchiostro, 255 = presenza).
        return:
            una lista di tuple (x,y,width,height,area), corrispondenti alle bounding boxes
            (coordinate del ritaglio) di ciascuna componente connessa presente
            nell'immagine.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(img)
    stats = sorted(stats, key=lambda s: s[4])  # ordino per area
    bbxs = stats[:-1]  # escludo la componente che rappresenta lo sfondo

    return bbxs


def bbxesCoverage(img):
    """

    :param img: numpy.ndarray di tipo uint8, shape (height,width). L'immagine deve
                essere binaria (0 = assenza di inchiostro, 255 = presenza).
    :return: una lista di tuple. [(inizioBBOX asse x, fineBBOX asse x)]
    """
    _, _, stats, centr = cv2.connectedComponentsWithStats(img)
    left, width = 0, 2
    # sorted on xCentroids
    return sorted(
        [(coord[left], coord[left] + coord[width], centroid) for coord, centroid in zip(stats[1:], centr[1:])],
        key=lambda s: s[2][0])


def centroids(img):
    """
    lists all centroids of the connected components (backround excluded)

    :param img: numpy.ndarray dtype = uint8, shape (height,width).
                Binary image (0 = no ink, 255 = inked).
    :return: centroids list (backgroud excluded = centr[1:]). Centroids type is (numpy.float64, numpy.float64)
    """
    _, _, _, centr = cv2.connectedComponentsWithStats(img)
    return centr[1:]


def centroids_bbxes_areas(img):
    """
    Lists all centroids and areas of the connected components (backround excluded), ordered by X axis of each centroid.

    :param img: numpy.ndarray dtype = uint8, shape (height,width).
                Binary image (0 = no ink, 255 = inked).
    :return: list of centroids and areas of the bboxes. List is ordered by x coordinate of each centroid.
             Tuples lists = [(xCentroid, yCentroid, area), ...]
    """
    _, _, stats, centr = cv2.connectedComponentsWithStats(img)
    return sorted([(cent[0], cent[1], area[4]) for cent, area in zip(centr[1:], stats[1:])])


def bbxes_data(img):
    """
    For each bbox (backround excluded), ordered by X axis of each centroid, bbxes_data returns
    centroid coordinates, area, width, height (approx) start and end of the bbox on x axis.
    :param img: numpy.ndarray dtype = uint8, shape (height,width).
                Binary image (0 = no ink, 255 = inked).
    :return: list of centroids and areas of the bboxes. List is ordered by x coordinate of each centroid.
                                0           1        2      3      4       5      6     7       8
             Tuples lists = [(xCentroid, yCentroid, area, width, height, xStart, xEnd, yStart, yEnd), ...]
    """
    _, _, stats, centr = cv2.connectedComponentsWithStats(img)
    #              (xCentr,  yCentr,   area,    width,  height,   xStart,   xEnd            , yStart,   yEnd           )
    return sorted([(cent[0], cent[1], stat[4], stat[2], stat[3], stat[0], stat[0] + stat[2], stat[1], stat[1] + stat[3])
                   for cent, stat in zip(centr[1:], stats[1:])])


def getMissingElements(img, cols, bbxesTKS, inputBGR=False, returnImage=False):
    """
    :param img: np.ndarray of type uint8
    :param cols: list of lists (python lists)
    :param bbxesTKS:
                [
                    [
                        [coordinates]
                        't'
                    ],
                    ...
                ]
    :param inputBGR: boolean. Is input
    :param returnImage: boolean. np.ndarray of type uint8, same shape of image. Contains missing elements
    :return: if returnImage np.ndarray of type uint8, same shape of image
    """
    usedMask = getAnnotatedBBxes(img, cols, bbxesTKS, keepSize=True)
    # missings
    allColors = [np.array(c) if inputBGR else np.flip(c) for c in findAllColors(img)]
    allColorsMask = np.invert(maskByColors(img, allColors))
    missingMask = np.bitwise_xor(allColorsMask, usedMask)
    missingBBxes = bbxes_data(missingMask)

    output = (missingMask, missingBBxes) if returnImage else missingBBxes
    return output


def getAnnotatedBBxes(img, cols, bbxes, keepSize=False):
    """
    Masks the color word by keeping annotated glyphs as white pixels and black for everything else.
    :param img: np.ndarray of uint8. Input color word
    :param cols: list of lists of ints. list of colors in RGB format.
    :param bbxes: list of lists. Each list contains coordinates and size as first element and the token as second
    :param keepSize: boolean. If true keeps the size of the output equal to the input image size.
    :return: np.ndarray. Output bw image containing glyphs having colors in cols
    """
    blank = np.zeros(img.shape[:2], dtype=np.uint8)
    minXStart, maxXEnd = 500, 0
    minYStart, maxYEnd = 500, 0

    for bb, token in bbxes:
        t = token
        if token not in cols:
            if t.isupper():
                t = token.lower()
            elif len(set(t)) == 1 and len(t) > 1:  # doubles
                t = token[0]
            elif t in ('us', 'ue'):
                t = 'semicolon'
            elif t is ".":  # just in case '.' color hasn't been annotated yet
                t = min(bbxes, key=lambda e: e[0][2])[1]
        _colors = np.flip(cols[t], axis=1)

        # coordinates
        xStart, xEnd, yStart, yEnd = bb[-4:]
        minXStart = min(minXStart, xStart)
        maxXEnd = max(maxXEnd, xEnd)
        minYStart = min(minYStart, yStart)
        maxYEnd = max(maxYEnd, yEnd)
        # sub-image containing s single component
        newToken = extractComponent(img, _colors, xStart, xEnd, yStart, yEnd)
        blank[yStart:yEnd, xStart:xEnd] = np.bitwise_or(blank[yStart:yEnd, xStart:xEnd], newToken)

    outImage = blank if keepSize else blank[minYStart:maxYEnd, minXStart:maxXEnd]
    return outImage


def cropByColor(img, cols):
    """
    Crops 'image' by keeping only areas associated with 'colors' via bounding box.
    Outputs a BW image where selected characters/colors are white and createBackground is black
    :param img: str.
    :param cols: numpy.ndarray. Colors as a numpy multidimentional list of BGR values of dtype uint8
    :return: numpy.array. Black and white image containing the connected component
    """

    mask = maskByColors(img, cols)

    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    compBBX = max(stats[1:], key=lambda s: s[4])  # in case of multiple matches
    # points of interest
    left = compBBX[0]
    right = left + compBBX[2]
    top = compBBX[1]
    bottom = top + compBBX[3]
    return mask[top: bottom, left: right]


def extractComponent(img, cols, fromX, toX, fromY, toY):
    """
    Crops 'image' by keeping only areas associated with 'colors' in a given range of pixels.
    Outputs a BW image where selected characters/colors are white and createBackground is black
    :param toY: bottom, pixel
    :param fromY: top pixel
    :param toX: rightmost pixel
    :param fromX: leftmost pixel
    :param img: np.ndarray
    :param cols: numpy.ndarray. Colors as a numpy multidimentional list of BGR values of dtype uint8
    :return: numpy.array. Black and white image containing the connected component
    """

    mask = maskByColors(img, cols)
    return mask[fromY: toY, fromX: toX]
    # return mask[fromY: toY+1, fromX: toX+1]


def createBackground(width=1400, height=1900, color=0):
    return np.zeros((height, width), dtype=np.uint8) if color == 0 else np.full((height, width), 255, dtype=np.uint8)


def mergeBBxes(thisBB, thatBB):
    """

    :param thisBB: Tuple. First bounding box. NO token
    :param thatBB: Tuple. Second bounding box NO token
    :return: Tuple. bounding box.

                 0           1        2      3      4       5      6     7       8
               (xCentroid, yCentroid, area, width, height, xStart, xEnd, yStart, yEnd)
    """
    xCentroid = np.mean([thisBB[0], thatBB[0]])
    yCentroid = np.mean([thisBB[1], thatBB[1]])
    area = thisBB[2] + thatBB[2]

    xStart = min([thisBB[5], thatBB[5]])
    xEnd = max([thisBB[6], thatBB[6]])
    width = xEnd - xStart

    yStart = min([thisBB[7], thatBB[7]])
    yEnd = max([thisBB[8], thatBB[8]])
    height = yEnd - yStart

    return xCentroid, yCentroid, area, width, height, xStart, xEnd, yStart, yEnd


def mergeBBxesWrapper(thisBB, thatBB):
    return mergeBBxes(thisBB[0], thatBB[0])


def scaleImageToEqualSize(sourceImg, targetImg):
    """
    Scales input image sourceImg to match targetImg.
    cv2.INTER_AREA is the suggest interpolator for downscaling
    :param targetImg: np.ndarray (3d matrix), from color words
    :param sourceImg: np.ndarray (3d matrix), artificial
    :return: np.array (3d matrix)
    """
    return cv2.resize(sourceImg,
                      None,  # no specific out dim
                      fx=targetImg.shape[1] / sourceImg.shape[1],  # scale factor Y
                      fy=targetImg.shape[0] / sourceImg.shape[0],  # scale factor X
                      interpolation=cv2.INTER_AREA)


def scaleToBBXSize(sourceImg, targetBBX):
    """
    Scales input image sourceImg to match targetBBX shape.
    cv2.INTER_AREA is the suggest interpolator for downscaling
    :param targetBBX: tuple, from color words.
                         0           1        2      3      4       5      6     7       8
        targetBBX =    (xCentroid, yCentroid, area, width, height, xStart, xEnd, yStart, yEnd)
    :param sourceImg: np.ndarray (3d matrix), artificial
    :return: np.array (3d matrix)
    """
    outImage = cv2.resize(src=sourceImg,
                          dsize=(targetBBX[3], targetBBX[4]),
                          interpolation=cv2.INTER_AREA)

    return outImage


def scale(source, target):
    """
    scale source image to target (object)
    :param source: np.ndarray (3d matrix)
    :param target: np.ndarray (3d matrix) or tuple (len 8)
    :return: np.array (3d matrix)
    """
    if isinstance(target, tuple) and len(target) == 9:
        out = scaleToBBXSize(source, target)
    elif isinstance(target, np.ndarray):
        out = scaleImageToEqualSize(source, target)
    else:
        print('target: ', type(target))
        raise TypeError
    return out


def getImageWidthHeight(path2img):
    """
    IT DOESN'T LOAD THE WHOLE FILE IN MEMORY!!
    :param path2img: string.
    :return: tuple (height, width)
    """
    from struct import unpack
    with open(path2img, 'rb') as f:
        metadata = f.read(25)
        return unpack('>LL', metadata[16:24])[::-1]


def countColoredHalves(image, bbx):
    """

    :param image: np.ndarray of (3d matrix). (height, width, channels=3)
    :param bbx: list ints. xCentroid, yCentroid, area, width, height, xStart, xEnd, yStart, yEnd
    :return:
    """
    subImage = maskBlackWhite(image)
    halfX = int(bbx[0][3] / 2)
    halfY = int(bbx[0][4] / 2)
    sx = subImage[:, :halfX]
    dx = subImage[:, halfX:]
    top = subImage[:halfY, :]
    bottom = subImage[halfY:, :]

    # colored pixels
    return np.count_nonzero(sx), np.count_nonzero(dx), np.count_nonzero(top), np.count_nonzero(bottom)


def img2base64str(image, ext=".png"):
    """
    Encodes image as a str in base64 representation.
            image -> cv2.imencoding -> b64encoding
    :param image: np.ndarray of uint8 (256, 256, 3)
    :param ext: image format of the image encoding (before b64 encoding)
    :return: bytes literal. Represent image encoding as base64
    """
    assert image.shape == (256, 256, 3)
    assert ext in {".png", ".jpg"}

    buffer = cv2.imencode('.png', image)[1]
    return b64encode(buffer)

