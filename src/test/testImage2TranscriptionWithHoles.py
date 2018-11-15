from json import load

from cv2 import imread
from os import path
from pprint import pprint
from config import wordsDoublesAndUppercase, annotationsJSON, color_words
from src.utils.imageProcessing import getMissingElements


def testTranscriptionWithHoles():
    imgs = ['040v/1015_1355_45_139.png', '048r/86_134_36_166.png', '056r_178_258_1393_1827/768_1024_47_181.png', '040v/1251_1137_44_83.png']

    imgsTranscription = {
        '040v/1015_1355_45_139.png': '~eligionis',
        '048r/86_134_36_166.png': 'concedimus',
        '056r_178_258_1393_1827/768_1024_47_181.png': 'parup~nd^e',
        '040v/1251_1137_44_83.png': '~^propr^~s'
    }

    with open(annotationsJSON, 'r') as ann, open(wordsDoublesAndUppercase, 'r') as w:
        annot = load(ann)
        words = load(w)

    for i in imgs:
        imgPath = path.join(color_words, i)
        image = imread(imgPath)
        missingsBBX = getMissingElements(image, list(annot[i].values()))['centroids_area']  # x, y, area
        filteredAreas = [(bb, '~') if bb[1] > 15 else (bb, '^') for bb in missingsBBX if bb[2] >= 50]

        wholeImage = sorted([bbxes for bbxes in words[i]]
                            + filteredAreas,
                            key=lambda b: b[0][0])

        chars = "".join([bbx[1] for bbx in wholeImage])

        pprint(wholeImage)
        assert imgsTranscription[i] == chars
        print(chars)
        print('\n')


if __name__ == '__main__':
    testTranscriptionWithHoles()
