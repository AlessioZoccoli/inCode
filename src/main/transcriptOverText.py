from os import path, getcwd, makedirs, errno
import cv2
from json import load


if __name__ == '__main__':
    imagesPath = path.join(getcwd(), '../../../color_words/')
    dataPath = path.join(getcwd(), '../../data/')

    # transcribed
    with open(path.join(dataPath, 'words.json'), 'r') as ann, open(path.join(dataPath, 'missings.json'), 'r')as m:
        annot = load(ann)
        missings = list(load(m).keys())

    font = cv2.FONT_HERSHEY_PLAIN

    # for each image insert transcription over text
    for im in missings:
        # creating directories
        dirName, imgName = im.split('/')
        outFolder = path.join(imagesPath, 'edited/', dirName)
        if not path.exists(outFolder):
            makedirs(outFolder)

        # Centroid, area to char
        imgPath = path.join(imagesPath, im)
        image = cv2.imread(imgPath)

        for coord, ch in annot[im]:
            fontSize = 1
            # reducing annotated char space
            if ch == 'i_bis' or ch[0] == 't':
                selectChar = 'i'
            elif ch[0] == 's' or ch == 'b_stroke':
                selectChar = ch[0:3:2]
                fontSize = 0.7
                coord = (int(coord[0])-4, int(coord[1]))
            else:
                selectChar = ch
            cv2.putText(image, selectChar, (int(coord[0]-3), int(coord[1])), font, fontSize, (0, 0, 0), 1, cv2.LINE_AA)

        outImagePath = path.join(outFolder, imgName)
        cv2.imwrite(outImagePath, image)