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
    all_colors = np.argwhere(hist != 0)

    return all_colors


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


def centroids(img):
    """
    elenca tutti i centroidi delle componenti connesse di img (sfondo escluso).
    :param img: numpy array di tipo uint8, shape (height,width). L'immagine deve
                essere binaria (0 = assenza di inchiostro, 255 = presenza).
    :return: lista di centroidi (sfondo escluso). list di valori di tipo numpy.float64
    """
    _, _, _, centr = cv2.connectedComponentsWithStats(img)
    return centr[1:]


def centroids_bbxes_areas(img):
    """
    elenca i centroidi e le bounding box delle componenti connesse di un'immagine, l'ordinamento avviene rispetto
    al centroide, da sinistra a destra (crescente).

    :param img: numpy array di tipo uint8, shape (height,width). L'immagine deve
                essere binaria (0 = assenza di inchiostro, 255 = presenza).
    :return: lista di centroidi e bounding box. Lista di tuple (centroide, (x, y, width, height, area))
    """
    _, _, stats, centr = cv2.connectedComponentsWithStats(img)
    #             (xCentroid, yCentroid, area)
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
        difference = np.array([np.array(el, dtype=np.uint8) for el in (differSet)], dtype=np.uint8)
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
