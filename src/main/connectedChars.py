from collections import defaultdict
from json import load, dump
from os import path, getcwd
from src.lib.image2word import positions2chars, disambiguate


if __name__ == '__main__':
    dataPath = path.join(getcwd(), '../../../color_words/')
    annotationsJSON = path.join(getcwd(), '../../data/anncolor_by_word.json')
    print('\npath json exists: ', path.exists(annotationsJSON))

    with open(annotationsJSON, 'r') as f:
        anncolor = load(f)

    charsWord = path.join(dataPath, 'charsWord.json')

    with open(path.join(getcwd(), '../../data/word_voted.json'), 'r') as fvotes:
        votes = load(fvotes)

    # output
    img2chars = defaultdict(list)
    last = 'NULL'
    found = set()
    keys = set(anncolor.keys())

    for i, val in anncolor.items():
        imgPath = path.join(dataPath, i)
        result = list(map(lambda x: x, positions2chars(imgPath, val)))

        # doubles checking
        if len(set([ch[0] for ch in result])) < len(result):
            img2chars[i] = disambiguate(i, result, votes)
        else:
            img2chars[i] = result
        img2chars[i] = [ch[1] for ch in img2chars[i]]
        last = i
        found.add(i)

    wordsPath = path.join(getcwd(), '../../data/words.json')
    with open(wordsPath, 'w') as words:
        dump(img2chars, words, indent=4)

    notFound = keys - found
    if len(notFound) > 0:
        print('Not processed images: #{}, {}'.format(len(notFound), notFound))
