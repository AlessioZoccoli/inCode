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
    :return: centroids list (backgroud excluded). Centroids type is (numpy.float64, numpy.float64)
    """
    _, _, _, centr = cv2.connectedComponentsWithStats(img)
    return centr[1:]


def centroids_bbxes_areas(img):
    """
    lists all centroids of the connected components (backround excluded), ordered by X axis of each centroid.

    :param img: numpy array dtype = uint8, shape (height,width).
                Binary image (0 = no ink, 255 = inked).
    :return: list of centroids and areas of the bboxes. List is ordered by x coordinate of each centroid.
             Tuples lists = [(xCentroid, yCentroid, area), ...]
    """
    _, _, stats, centr = cv2.connectedComponentsWithStats(img)
    return sorted([(cent[0], cent[1], area[4]) for cent, area in zip(centr[1:], stats[1:])])


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
        # processing => BGR
        missingsMask = mask_by_colors(image, difference)
        # [(xCentroid, yCentroid, area)]
        missings = centroids_bbxes_areas(missingsMask)
        # storing => RGB
        difference = np.flip(difference, 1)
    else:
        difference = []
        missings = []
    return {'colors': difference, 'centroids_area': missings}


def translateToken(token):
    charDict = {
        'semicolon': ['u', 'es'],
        'ues': ['u', 'es'],
        'curl': ['m', 'u', 's'],
        'curly_dash': []
    }
    try:
        return charDict[token]
    except KeyError:
        if len(token) > 1:
            if token[1] == '_':
                return [token[0]]
            else:
                return list(token)
        else:
            return [token]


def toBigrams(ccomps):
    """
    Connected component to bigram.
    :param ccomps: list of lists, as follows: [ [full word],
                                                [[ccomp0], [ccomp1], [ccomp2]] ]     <<<----- comps
                    each full word is followed by its connected components
    :return: list of bigrams
    """
    bgrams = []

    for _, comps in ccomps:
        for comp in comps:
            if comp:
                cleanComp = ['<s>'] + [t for char in comp for t in translateToken(char) if t] + ['</s>']
                newChain = [(cleanComp[i], cleanComp[i + 1]) for i in range(len(cleanComp) - 1)]
                bgrams.extend(newChain)
    return bgrams
