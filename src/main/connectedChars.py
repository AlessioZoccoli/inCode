from collections import defaultdict
from json import load, dump
from os import path, getcwd
from src.lib.image2word import positions2chars, disambiguate
from pprint import pprint
from copy import deepcopy

if __name__ == '__main__':
    dataPath = path.join(getcwd(), '../../data/')
    annotationsJSON = path.join(dataPath, 'anncolor_by_word.json')
    print('\npath json exists: ', path.exists(annotationsJSON))

    with open(annotationsJSON, 'r') as f:
        anncolor = load(f)

    images = ['040v/145_1688_48_122.png','040v/951_307_48_102.png', '040v/159_585_41_63.png',
              '040v/1243_1517_28_65.png', '040v/401_532_46_140.png', '040v/408_309_42_71.png',
              '040v/1253_804_34_62.png', '040v/1270_158_38_61.png']

    charsWord = path.join(dataPath, 'charsWord.json')

    with open(path.join(getcwd(), '../../data/word_voted.json'), 'r') as fvotes:
        votes = load(fvotes)


    img2chars = defaultdict(list)
    for i in images:
        imgPath = path.join(dataPath, i)
        #print('\n\n##########################################\n')
        result = list(map(lambda x: x, positions2chars(imgPath, anncolor[i])))

        # doubles checking
        if len(set([ch[0] for ch in result])) < len(result):
            img2chars[i] = disambiguate(i, result, votes)
        else:
            img2chars[i] = result
        #pprint(img2chars[i])
