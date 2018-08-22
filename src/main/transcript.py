from collections import defaultdict
from json import load, dump
from os import path, getcwd
from config import color_words, annotationsJSON, votesJSON, wordsClean
from pprint import pprint

from src.lib.image2word import positions2chars
from src.utils.textProcessing import translateToken


if __name__ == '__main__':


    print('\npath for images exists: {}\n'.format(path.exists(color_words)))
    print('\npath for annotations exists: {}\n'.format(path.exists(annotationsJSON)))
    print('\npath for votes exists: {}\n'.format(path.exists(votesJSON)))

    with open(annotationsJSON, 'r') as f, open(votesJSON, 'r') as v:
        anncolor = load(f)
        votes = load(v)

    # output
    img2chars = defaultdict(list)

    for i, val in anncolor.items():
            imgPath = path.join(color_words, i)
            result = positions2chars(imgPath, val, votes[i])
            lastChar = ''
            for char in result:
                newChar = translateToken(char[1], prev=lastChar)
                lastChar = newChar
                if newChar:
                    img2chars[i].append((char[0], newChar))


    with open(wordsClean, 'w') as words:
        dump(img2chars, words, indent=4, sort_keys=True)
