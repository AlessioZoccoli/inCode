from config import color_words, images2ColorsBBxesJSON
from json import load
from cv2 import imshow, imread, waitKey, destroyAllWindows


if __name__ == '__main__':
    with open(images2ColorsBBxesJSON, 'r') as ann:
        ch2col = load(ann)
    """
    imagePath = path.join(color_words, '040v/1001_696_43_114.png')
    img = imread(imagePath)

    print(concatenate((ch2col['040v/1001_696_43_114.png']['annot']['e'],
                       ch2col['040v/1001_696_43_114.png']['annot']['q']), axis=0))


    eq_BGRcolors = flip(concatenate(
        (ch2col['040v/1001_696_43_114.png']['annot']['e'], ch2col['040v/1001_696_43_114.png']['annot']['q']),
        axis=0), axis=1)

    ptimo_BGRcolors = flip(concatenate(
        (ch2col['048v/308_1517_40_141.png']['annot']['p'],
         ch2col['048v/308_1517_40_141.png']['annot']['t'],
         ch2col['048v/308_1517_40_141.png']['annot']['i'],
         ch2col['048v/308_1517_40_141.png']['annot']['m'],
         ch2col['048v/308_1517_40_141.png']['annot']['o']
         ), axis=0
    ), axis=1)

    # component 'eq'
    eq = cropByColor(img, eq_BGRcolors)
    # component 'ptimo'


    back = background()
    back[0:eq.shape[0], 0:eq.shape[1]] = eq

    imshow('maskWithBackground', cropByColor(img, eq_BGRcolors))
    """
    waitKey(0)
    destroyAllWindows()
