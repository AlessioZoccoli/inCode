from collections import defaultdict
from json import load, dump
from os import path
from pprint import pprint

from config import annotationsJSON, annotationsRichJSON, annotationsCleanJSON, wordsDoublesAndUppercase
from src.utils.textProcessing import translateToken


def cleanAnncolorRich():
    """
    Notations for "clean" tokens plus some useful tokens es. "d_token", "s_mediana",  "s_ending".
    Basilically merging anncolor_by_word.json with anncolor_by_word_clean.json
    :return: None
    """

    richTokensConv = {
        's_mediana': '1',
        's_ending': '2',
        'd_stroke': '3',
        'l_stroke': '4',
        'b_stroke': '5',
        'curl': '6',
        'qui': '7',
        'con': '8',
        'nt': '9',
        'prop': '/',
        'pro': '$',
        'per': '%',
        'semicolon': '&',
        'rum': '('
    }

    with open(annotationsJSON, 'r') as fa, open(annotationsCleanJSON, 'r') as ca, open(wordsDoublesAndUppercase, 'r') as wdu:
        fullAnnot = load(fa)
        cleanAnnot = load(ca)
        words = load(wdu)

    for img, tk2colors in fullAnnot.items():
        for tk, colors in tk2colors.items():
            if tk in richTokensConv and img in cleanAnnot:
                try:
                    cleanAnnot[img].update({richTokensConv[tk]: colors})
                    # remove annecessary annotations. For s_ and _strokes we may have "aggregated" annotations
                    # so they are not deleted
                    if tk[:2] not in {'s_', 'd_', 'b_', 'l_', 'cu', 'se'} and tk in cleanAnnot[img]:
                        # qui, con, nt ...
                        del cleanAnnot[img][tk]
                except KeyError:
                    print(img, tk, cleanAnnot[img].keys(), '  ', richTokensConv[tk])
                    pass

    # 'prop' is specials since it does not appear in fullAnnot
    for im, tk2col in cleanAnnot.items():
        for tk, col in tk2col.items():
            if tk == 'prop':
                cleanAnnot[im].update({'/': col})
                del cleanAnnot[im]['prop']
            if tk == 'pro':
                cleanAnnot[im].update({'$': col})
                del cleanAnnot[im]['pro']

    with open(annotationsRichJSON, 'w') as ar:
        dump(cleanAnnot, ar, indent=4, sort_keys=True)
        print('file  {}  written'.format(annotationsRichJSON))


if __name__ == '__main__':
    cleanAnncolorRich()
