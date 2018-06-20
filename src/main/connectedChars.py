from collections import defaultdict
from json import load, dump
from os import path, getcwd
from src.lib.image2word import positions2chars, disambiguate


if __name__ == '__main__':
    dataPath = path.join(getcwd(), '../../../color_words/')
    annotationsJSON = path.join(getcwd(), '../../data/anncolor_by_word.json')
    votesJSON = path.join(getcwd(), '../../data/word_voted.json')
    print('\npath json exists: {}\n'.format(path.exists(annotationsJSON)))

    with open(annotationsJSON, 'r') as f, open(votesJSON, 'r') as v:
        anncolor = load(f)
        votes = load(v)

    # output
    img2chars = defaultdict(list)

    for i, val in anncolor.items():
        imgPath = path.join(dataPath, i)
        result = positions2chars(imgPath, val, votes[i])
        img2chars[i] = result

    wordsPath = path.join(getcwd(), '../../data/words.json')
    with open(wordsPath, 'w') as words:
        dump(img2chars, words, indent=4)
