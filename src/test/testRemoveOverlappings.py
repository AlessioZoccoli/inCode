from json import load
from os import path, getcwd
from src.lib.image2word import positions2chars
from pprint import pprint

if __name__ == '__main__':

    dataPath = path.join(getcwd(), '../../../color_words/')
    with open(path.join(getcwd(), '../../data/anncolor_by_word.json'), 'r') as ann, \
            open('../../data/anncolor_by_word_noOverlappings.json', 'w') as annVotesJSON:

        annotations = load(ann)

        images = ['060v/999_327_30_111.png']

        for i in images:
            print('\n\n#################################', i)
            imgPath = path.join(dataPath, i)
            pprint(positions2chars(imgPath, annotations[i]))
