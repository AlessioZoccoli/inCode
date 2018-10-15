import cv2
import numpy as np


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
    :return: mask of shape (h,w) on the original one, which takes the value of 255 if there is ant non white color,
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
                numpy array di tipo uint8, shape (height,width,channels)

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
                numpy array di tipo uint8, shape (height,width). L'immagine deve
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

    :param img: numpy array di tipo uint8, shape (height,width). L'immagine deve
                essere binaria (0 = assenza di inchiostro, 255 = presenza).
    :return: una lista di tuple. [(inizioBBOX asse x, fineBBOX asse x)]
    """
    _, _, stats, centr = cv2.connectedComponentsWithStats(img)
    left, width = 0, 2
    # sorted on xCentroids
    return sorted([(coord[left], coord[left] + coord[width], centroid) for coord, centroid in zip(stats[1:], centr[1:])], key=lambda s: s[2][0])


def centroids(img):
    """
    lists all centroids of the connected components (backround excluded)

    :param img: numpy array dtype = uint8, shape (height,width).
                Binary image (0 = no ink, 255 = inked).
    :return: centroids list (backgroud excluded = centr[1:]). Centroids type is (numpy.float64, numpy.float64)
    """
    _, _, _, centr = cv2.connectedComponentsWithStats(img)
    return centr[1:]


def centroids_bbxes_areas(img):
    """
    Lists all centroids and areas of the connected components (backround excluded), ordered by X axis of each centroid.

    :param img: numpy array dtype = uint8, shape (height,width).
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
    :param img: numpy array dtype = uint8, shape (height,width).
                Binary image (0 = no ink, 255 = inked).
    :return: list of centroids and areas of the bboxes. List is ordered by x coordinate of each centroid.
                                0           1        2      3      4       5      6     7       8
             Tuples lists = [(xCentroid, yCentroid, area, width, height, xStart, xEnd, yStart, yEnd), ...]
    """
    _, _, stats, centr = cv2.connectedComponentsWithStats(img)
    #              (xCentr,  yCentr,   area,    width,  height,   xStart,   xEnd            , yStart,   yEnd           )
    return sorted([(cent[0], cent[1], stat[4], stat[2], stat[3], stat[0], stat[0] + stat[2], stat[1], stat[1] + stat[3])
                   for cent, stat in zip(centr[1:], stats[1:])])


def getMissingElements(image, annotations):
    """
    Returns colors and bounding boxes for missing elements
    :param image: uint8 numpy array, shape (height,width,channels)
    :param annotations: lists of lists, colors grouped bu char
    :return: colors and bounding boxes for missing elements
    """
    # BGR
    allColors = findAllColors(image)
    allColorsComp = set(tuple(sublist) for sublist in (allColors.tolist()))
    # RGB -> BGR
    annotColors = set((tuple(item[::-1])) for sublist in list(annotations) for item in sublist)
    annotColors.add((255, 255, 255))  # don't include white when applying difference

    differSet = allColorsComp - annotColors
    if differSet:
        difference = np.array([np.array(el, dtype=np.uint8) for el in differSet], dtype=np.uint8)
        missingsMask = maskByColors(image, difference)        # processing => BGR
        missings = centroids_bbxes_areas(missingsMask)          # [(xCentroid, yCentroid, area)]
        difference = np.flip(difference, 1)                     # storing => RGB
    else:
        difference = []
        missings = []
    return {'colors': difference, 'centroids_area': missings}


def cropByColor(image, colors):
    """
    Crops 'image' by keeping only areas associated with 'colors' via bounding box.
    Outputs a BW image where selected characters/colors are white and createBackground is black
    :param image: str.
    :param colors: numpy array. Colors as a numpy matrix of BGR values of dtype uint8
    :return: numpy.array. Black and white image containing the connected component
    """

    mask = maskByColors(image, colors)

    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    compBBX = max(stats[1:], key=lambda s: s[4])                # in case of multiple matches
    # points of interest
    left = compBBX[0]
    right = left + compBBX[2]
    top = compBBX[1]
    bottom = top + compBBX[3]
    return mask[top: bottom, left: right]


def extractComponent(image, colors, fromX, toX, fromY, toY):
    """
    Crops 'image' by keeping only areas associated with 'colors' in a given range of pixels.
    Outputs a BW image where selected characters/colors are white and createBackground is black
    :param toY:
    :param fromY:
    :param toX:
    :param fromX:
    :param image: str.
    :param colors: numpy array. Colors as a numpy matrix of BGR values of dtype uint8
    :return: numpy.array. Black and white image containing the connected component
    """

    mask = maskByColors(image, colors)
    return mask[fromY: toY, fromX: toX]


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


def scaleImageToEqualSize(sourceImg, targetImg):
    """
    Scales input image sourceImg to match targetImg.
    cv2.INTER_AREA is the suggest interpolator for downscaling
    :param targetImg: np.array (3d matrix), from color words
    :param sourceImg: np.array (3d matrix), artificial
    :return: np.array (3d matrix)
    """
    return cv2.resize(sourceImg,
                      None,                                             # no specific out dim
                      fx=targetImg.shape[1]/sourceImg.shape[1],         # scale factor Y
                      fy=targetImg.shape[0]/sourceImg.shape[0],         # scale factor X
                      interpolation=cv2.INTER_AREA)


def scaleToBBXSize(sourceImg, targetBBX):
    """
    Scales input image sourceImg to match targetBBX shape.
    cv2.INTER_AREA is the suggest interpolator for downscaling
    :param targetBBX: tuple, from color words.
                         0           1        2      3      4       5      6     7       8
        targetBBX =    (xCentroid, yCentroid, area, width, height, xStart, xEnd, yStart, yEnd)
    :param sourceImg: np.array (3d matrix), artificial
    :return: np.array (3d matrix)
    """
    if targetBBX[3] != targetBBX[6] - targetBBX[5]:
        targetBBX[3] = max(targetBBX[3], targetBBX[6] - targetBBX[5])
    if targetBBX[4] != targetBBX[8] - targetBBX[7]:
        targetBBX[4] = max(targetBBX[4], targetBBX[8] - targetBBX[7])

    outImage = cv2.resize(src=sourceImg,
                          dsize=(targetBBX[3], targetBBX[4]),
                          interpolation=cv2.INTER_AREA)

    return outImage


def scale(source, target):
    """
    scale source image to target (object)
    :param source: np.array (3d matrix)
    :param target: np.array (3d matrix) or tuple (len 8)
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
