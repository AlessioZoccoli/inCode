from collections import defaultdict
from json import load, dump
from os import path, getcwd
from src.lib.image2word import chars2position

if __name__ == '__main__':
    dataPath = path.join(getcwd(), '../../data/')
    jsonPath = path.join(dataPath, 'anncolor_by_word.json')
    print('\npath json exists: ', path.exists(jsonPath))

    with open(jsonPath, 'r') as f:
        anncolor = load(f)

    images = ['040v/951_307_48_102.png', '040v/159_585_41_63.png', '040v/1243_1517_28_65.png',
              '040v/401_532_46_140.png','040v/408_309_42_71.png', '040v/1253_804_34_62.png', '040v/1270_158_38_61.png']

    charsWord = path.join(dataPath, 'charsWord.json')
    with open(charsWord, 'w') as out:
        img2chars = defaultdict(list)
        for i in images:
            imgPath = path.join(dataPath, i)
            print('\n##########################################\n')
            print(i)
            img2chars[i] = list(map(lambda x: x[0], chars2position(imgPath, anncolor[i])))

        dump(img2chars, out, indent=4, sort_keys=True)
