from collections import defaultdict
from json import load, dump
from cv2 import imread
from os import path
from pprint import pprint

from config import images2ColorsBBxesJSON, color_words, transcriptedWords_holes
from src.utils.imageProcessing import getMissingElements


def transcriptionWithHoles():

    with open(images2ColorsBBxesJSON, 'r') as w:
        image2data = load(w)

    img2transcription = defaultdict(str)
    for i, data in image2data.items():
        colors = data['col']
        bbxes = data['tks']
        imgPath = path.join(color_words, i)
        image = imread(imgPath)
        # _imMiss, _imMissBB = getMissingElements(_imImage, _imColors, _imBBxes, returnImage=True)
        try:
            missingsBBX = getMissingElements(image, colors, bbxes, returnImage=False)
            filteredAreas = [(bb, '~') if bb[1] > 15 else (bb, '^') for bb in missingsBBX if bb[2] >= 50]
            # sorting all bbxes
            wholeImageBBxes = sorted(bbxes + filteredAreas, key=lambda b: b[0][0])
            chars = "".join([bbx[1] for bbx in wholeImageBBxes])

            # image: transcription with holes
            img2transcription[i] = chars
        except KeyError:
            print(i, colors.keys(), [b[1] for b in bbxes])

    with open(transcriptedWords_holes, 'w') as tr:
        dump(img2transcription, tr, sort_keys=True, indent=True)


if __name__ == '__main__':

    from datetime import datetime
    print(datetime.now())
    transcriptionWithHoles()
    print(datetime.now())
