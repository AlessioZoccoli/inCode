import cv2
import numpy as np


def mask_by_colors(word_img, colors):
    """
        funzione per creare una maschera su un'immagine,
        a partire da una lista di colori.
        parametri:
            - word_img:
                numpy array di tipo uint8, shape (height,width,channels)
            - colors:
                numpy array di tipo uint8. Ogni elemento è nella forma [blue,green,red].
        return:
            una maschera con shape (h,w) sull'immagine originaria,
            che vale 0 se color non c'è, 255 altrimenti.
    """
    mask = np.zeros((word_img.shape[0], word_img.shape[1]), dtype='uint8')

    for c in colors:
        colormask = cv2.inRange(word_img, lowerb=c, upperb=c)
        mask = cv2.bitwise_or(mask, colormask)

    return mask


def find_all_colors(img):
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


def sorted_bbxs(img):
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
    allColors = find_all_colors(image)
    allColorsComp = set(tuple(sublist) for sublist in (allColors.tolist()))
    # RGB -> BGR
    annotColors = set((tuple(item[::-1])) for sublist in list(annotations) for item in sublist)
    annotColors.add((255, 255, 255))  # don't include white when applying difference

    differSet = allColorsComp - annotColors
    if differSet:
        difference = np.array([np.array(el, dtype=np.uint8) for el in differSet], dtype=np.uint8)
        missingsMask = mask_by_colors(image, difference)        # processing => BGR
        missings = centroids_bbxes_areas(missingsMask)          # [(xCentroid, yCentroid, area)]
        difference = np.flip(difference, 1)                     # storing => RGB
    else:
        difference = []
        missings = []
    return {'colors': difference, 'centroids_area': missings}


def cropByColor(image, colors):
    """
    Crops 'image' by keeping only areas associated with 'colors' via bounding box.
    Outputs a BW image where selected characters/colors are white and creatBackground is black
    :param image: str.
    :param colors: numpy array. Colors as a numpy matrix of BGR values of dtype uint8
    :return: numpy.array. Black and white image containing the connected component
    """

    mask = mask_by_colors(image, colors)

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
    Outputs a BW image where selected characters/colors are white and creatBackground is black
    :param toY:
    :param fromY:
    :param toX:
    :param fromX:
    :param image: str.
    :param colors: numpy array. Colors as a numpy matrix of BGR values of dtype uint8
    :return: numpy.array. Black and white image containing the connected component
    """

    mask = mask_by_colors(image, colors)
    return mask[fromY: toY, fromX: toX]


def creatBackground(width=1400, height=1900, color=0):
    return np.zeros((height, width), dtype=np.uint8) if color == 0 else np.full((height, width), 255, dtype=np.uint8)


def mergeBBxes(thisBB, thatBB):
    """

    :param thisBB: Tuple. First bounding box
    :param thatBB: Tuple. Second bounding box
    :return: Tuple. bounding box.

                 0           1        2      3      4       5      6     7       8
               (xCentroid, yCentroid, area, width, height, xStart, xEnd, yStart, yEnd)
    """
    xCentroid = np.mean([thisBB[0], thatBB[0]])
    yCentroid = np.mean([thisBB[1], thatBB[1]])
    area = thisBB[2] + thatBB[2]
    width = max([thisBB[6], thatBB[6]]) - min([thisBB[5], thatBB[5]])
    height = max([thisBB[8], thatBB[8]]) - min([thisBB[7], thatBB[7]])
    xStart = min([thisBB[5], thatBB[5]])
    xEnd = xStart + width
    yStart = min([thisBB[7], thatBB[7]])
    yEnd = yStart + height

    return xCentroid, yCentroid, area, width, height, xStart, xEnd, yStart, yEnd
