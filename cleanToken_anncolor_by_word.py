from collections import defaultdict
from json import load, dump
from os import path
from pprint import pprint

from config import annotationsJSON, annotationsCleanJSON
from src.utils.textProcessing import translateToken


def cleanAnncolor():
    """
    Edit the file containing annotations to fit data cleaning on tokens
    :return: None
    """
    with open(annotationsJSON, 'r') as fa:
        fullAnnot = load(fa)

    cleanOutput = defaultdict(lambda: defaultdict(list))

    for img, data in fullAnnot.items():
        prevChar = ''
        for char, colors in data.items():
            cleanChar = translateToken(char, prev=prevChar)
            prevChar = cleanChar
            cleanOutput[img][cleanChar].extend(colors)

    # pprint(cleanOutput)
    # with open(annotationsCleanJSON, 'w') as ac:
    #     dump(cleanOutput, ac, indent=4, sort_keys=True)


if __name__ == '__main__':
    cleanAnncolor()