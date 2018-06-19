from collections import defaultdict
from itertools import combinations
from json import load
from os import path, getcwd
from pprint import pprint

if __name__ == '__main__':

    # imagesPath = path.join(getcwd(), '../../color_words/')
    dataPath = path.join(getcwd(), '../data/')
    # imageName = '048r/86_134_36_166.png' # '056r_178_258_1393_1827/768_1024_47_181.png'

    with open(path.join(dataPath, 'anncolor_by_word.json'), 'r') as f:
        annotatJSON = load(f)

    overlappings = defaultdict(list)

    # cont = 0
    for im in annotatJSON:
        for thisCol, thatCol in combinations(annotatJSON[im].items(), 2):
            overlapping = [c for c in thisCol[1] if c in thatCol[1]]
            if overlapping and len(overlapping) != len(thisCol[1]) and len(overlapping) != len(thatCol[1]):
                overlappings[im].append(overlapping)

    pprint(overlappings)