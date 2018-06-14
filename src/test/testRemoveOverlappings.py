from itertools import combinations, product
from json import load
from os import path, getcwd
from numpy import mean, diff, absolute
from src.lib.image2word import positions2chars
from pprint import pprint

if __name__ == '__main__':

    dataPath = path.join(getcwd(), '../../../color_words/')
    with open(path.join(getcwd(), '../../data/anncolor_by_word.json'), 'r') as ann,\
            open('../../data/anncolor_by_word_noOverlappings.json', 'w') as annVotesJSON:

        annotations = load(ann)
        images = ['040v/1177_1187_37_68.png']  # ['040v/235_584_47_151.png','040v/1103_588_47_72.png', '040v/1164_1355_32_127.png']

        imgPath = path.join(dataPath, images[0])
        word = positions2chars(imgPath, annotations[images[0]])
        """
        meanCentroidXDistance = mean(absolute(diff([el[0][0] for el in word])))

        omit = []
        for this, that in combinations(annotations[images[0]].items(), 2):
            overlapping = [c for c in this[1] if c in that[1]]
            if overlapping:
                # grouping overlapping chars
                thisBBxes, thatBBxes = [], []
                for ch in word:
                    if ch[1] == this[0]:
                        thisBBxes.append(ch)
                    elif ch[1] == that[0]:
                        thatBBxes.append(ch)
                # choosing bbox to delete by comparing all possible overlapping bboxes
                for p in product(thisBBxes, thatBBxes):
                    # comparing centroids distance (x axis) -> too short = overlapping bboxes, omit the smaller one
                    if abs(p[0][0][0] - p[1][0][0]) < meanCentroidXDistance:
                        omit.append(min(p, key=lambda b: b[0][2]))

        word = [w for w in word if w not in omit]
        print('\n')
        """
        print(word)