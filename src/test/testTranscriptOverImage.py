from os import path, getcwd, makedirs, errno
import cv2
from json import load
from src.lib.image2word import positions2chars
from src.utils.utils import getMissingElements, mask_by_colors
from numpy import  flip


if __name__ == '__main__':
    imagesPath = path.join(getcwd(), '../../../color_words/')
    dataPath = path.join(getcwd(), '../../data/')
    # imageName = '048r/86_134_36_166.png' # '056r_178_258_1393_1827/768_1024_47_181.png'

    i = '040v/1015_1355_45_139.png'

    imgPath = path.join(imagesPath, i)
    image = cv2.imread(imgPath)

    with open(path.join(dataPath, 'anncolor_by_word.json'), 'r') as annFile,\
            open(path.join(dataPath, 'words_voted.json'), 'r') as votesFile:
        annot = load(annFile)
        votes = load(votesFile)

    font = cv2.FONT_HERSHEY_PLAIN
    # Centroid, area to char
    centroidsChars = positions2chars(imgPath, annot[i], votes[i])

    """
    Show missing elements
    """
    missings = getMissingElements(image, list(annot[i].values()))
    # RGB -> BGR
    mask = mask_by_colors(image, flip(missings['colors'], 1))
    mask3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow('masked', mask3ch)
    cv2.waitKey(0)

    """
    show text over word image
    """
    for coord, ch in centroidsChars:
        fontSize = 1
        selectChar = ch
        if ch == 'i_bis':
            selectChar = 'i'
        elif ch[0] == 't':
            selectChar = 't'
        elif ch == 'semicolon':
            selectChar = 'sc'
            fontSize = 0.7
            coord = (int(coord[0]) - 4, int(coord[1]))
        elif ch[:2] == 's_' or ch[2:] == 'stroke':
            selectChar = ch[0:3:2]
            fontSize = 0.6
            coord = (int(coord[0]) - 4, int(coord[1]))
        elif ch == 'pro':
            fontSize = 0.6
            coord = (int(coord[0]) - 4, int(coord[1]))

        cv2.putText(image, selectChar, (int(coord[0] - 3), int(coord[1])), font, fontSize, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('image', image)
    cv2.waitKey(0)