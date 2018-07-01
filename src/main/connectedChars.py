from collections import defaultdict
from json import load, dump
from os import path, getcwd
from pprint import pprint

from src.lib.image2word import positions2chars
from src.utils.utils import translateToken

if __name__ == '__main__':

    imagesPath = path.join(getcwd(), '../../../color_words/')
    annotationsJSON = path.join(getcwd(), '../../data/anncolor_by_word.json')
    votesJSON = path.join(getcwd(), '../../data/words_voted.json')
    print('\npath json exists: {}\n'.format(path.exists(annotationsJSON)))

    with open(annotationsJSON, 'r') as f, open(votesJSON, 'r') as v:
        anncolor = load(f)
        votes = load(v)

    # output
    img2chars = defaultdict(list)

    c = 0
    for i, val in anncolor.items():
        imgPath = path.join(imagesPath, i)
        result = positions2chars(imgPath, val, votes[i])

        cleanResult = [(char[0], ''.join(translateToken(char[1]))) for char in result]
        img2chars[i] = cleanResult

    wordsPath = path.join(getcwd(), '../../data/words_clean.json')

    with open(wordsPath, 'w') as words:
        dump(img2chars, words, indent=4)
