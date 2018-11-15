from collections import defaultdict
from json import load, dump
from cv2 import imread
from os import path
from config import wordsDoublesAndUppercase, annotationsJSON, color_words, transcriptedWords_holes
from src.utils.imageProcessing import getMissingElements
import datetime


def transcriptionWithHoles():
    with open(annotationsJSON, 'r') as ann, open(wordsDoublesAndUppercase, 'r') as w:
        annot = load(ann)
        words = load(w)

    img2transcription = defaultdict(str)

    for i, bbxes in words.items():
        imgPath = path.join(color_words, i)
        image = imread(imgPath)
        missingsBBX = getMissingElements(image, list(annot[i].values()))['centroids_area']  # x, y, area
        filteredAreas = [(bb, '~') if bb[1] > 15 else (bb, '^') for bb in missingsBBX if bb[2] >= 50]

        wholeImage = sorted([bbxes for bbxes in words[i]]
                            + filteredAreas,
                            key=lambda b: b[0][0])

        chars = "".join([bbx[1] for bbx in wholeImage])

        # image: transcription with holes
        img2transcription[i] = chars

    with open(transcriptedWords_holes, 'w') as tr:
        dump(img2transcription, tr, sort_keys=True, indent=True)


if __name__ == '__main__':
    print(datetime.datetime.now())
    transcriptionWithHoles()
    print(datetime.datetime.now())
