from collections import defaultdict
from json import load, dump
from os import path
from pprint import pprint

from config import annotationsJSON, annotationsRichJSON, annotationsCleanJSON
from src.utils.textProcessing import translateToken


def cleanAnncolorRich():
    """
    Notations for "clean" tokens plus some useful tokens es. "d_token", "s_mediana",  "s_ending".
    Basilically merging anncolor_by_word.json with anncolor_by_word_clean.json
    :return: None
    """

    richTokens = {
        's_mediana': '1',
        's_ending': '2',
        'd_stroke': '3',
        'l_stroke': '4',
        'b_stroke': '5',
        'curl': '6'
    }

    with open(annotationsJSON, 'r') as fa, open(annotationsCleanJSON, 'r') as ca:
        fullAnnot = load(fa)
        cleanAnnot = load(ca)

    for img, tk2colors in fullAnnot.items():
        for tk, colors in tk2colors.items():
            if tk in richTokens:
                try:
                    cleanAnnot[img].update({richTokens[tk]: colors})
                    if tk == 'curl':
                        del cleanAnnot[img]['us']
                except KeyError:
                    pass

    with open(annotationsRichJSON, 'w') as ar:
        dump(cleanAnnot, ar, indent=4, sort_keys=True)
        print('file  {}  written'.format(annotationsRichJSON))


if __name__ == '__main__':
    cleanAnncolorRich()
