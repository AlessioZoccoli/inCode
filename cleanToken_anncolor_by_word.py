from collections import defaultdict
from json import load, dump
from os import path
from pprint import pprint

from config import annotationsJSON, annotationsCleanJSON
from src.utils.textProcessing import translateToken


if __name__ == '__main__':

    with open(annotationsJSON, 'r') as fa:
        fullAnnot = load(fa)

    cleanOutput = defaultdict(lambda: defaultdict(list))

    count = 0
    for img, data in fullAnnot.items():
        if count > 5:
            break
        count += 1
        prevChar = ''
        for char, colors in data.items():
            cleanChar = translateToken(char, prev=prevChar)
            prevChar = cleanChar
            cleanOutput[img][cleanChar].extend(colors)

    pprint(cleanOutput)
    # with open(annotationsCleanJSON, 'w') as ac:
    #     dump(cleanOutput, ac, indent=4, sort_keys=True)
