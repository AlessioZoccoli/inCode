from collections import defaultdict
from json import load
from os import path, getcwd
from src.lib.image2word import disambiguate, positions2chars
from pprint import pprint

if __name__ == '__main__':
    dataPath = path.join(getcwd(), '../../../color_words/')
    annotationsJSON = path.join(getcwd(), '../../data/anncolor_by_word.json')
    votesJSON = path.join(getcwd(), '../../data/word_voted.json')
    print('\npath json exists: {}\n'.format(path.exists(annotationsJSON)))

    with open(annotationsJSON, 'r') as f:
        anncolor = load(f)

    with open(votesJSON, 'r') as v:
        votes = load(v)

    # output
    img2chars = defaultdict(list)

    images = ['040v/1270_158_38_61.png', '040v/1243_1517_28_65.png', '040v/159_585_41_63.png', '040v/951_307_48_102.png',
              '040v/401_532_46_140.png', '040v/145_1688_48_122.png']

    # imagePath: [char:[vote, [..., colors, ...]]]
    for myImage in images:
        imgPath = path.join(dataPath, myImage)
        result = list(map(lambda x: x, positions2chars(imgPath, anncolor[myImage])))

        # doubles checking
        if len(set([ch[0] for ch in result])) < len(result):
            img2chars[myImage] = disambiguate(myImage, result, votes)
        else:
            img2chars[myImage] = result

    pprint(img2chars)
    #print('\n', [(k,[ch[1] for ch in v]) for k,v in img2chars.items()])