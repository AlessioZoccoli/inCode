from collections import defaultdict

from config import transcriptedWords_holes, transcriptedWords_holesFree
from json import load, dump
from re import search

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


def holesFreeTranscription():
    """
    Stores color words
    :return:
    """

    holesFree = defaultdict(str)
    counter = 0

    reverseRichConv = dict(zip(richTokensConv.values(), ["<{}>".format(rvr) for rvr in richTokensConv.keys()]))

    with open(transcriptedWords_holes, 'r') as t:
        transcriptions = load(t)
        for img, transcription in transcriptions.items():
            hasNoHoles = search(r"^[^a-zA-Z0-9]*[a-zA-Z0-9]+[^a-zA-Z0-9]*$", transcription)
            if hasNoHoles:
                holesFree[img] = "".join([el if el not in reverseRichConv else reverseRichConv[el] for el in hasNoHoles.group()])
                counter += 1

    with open(transcriptedWords_holesFree, 'w') as out:
        dump(holesFree, out, sort_keys=True, indent=4)

    print('\nWritten out to:     {}\n'.format(transcriptedWords_holesFree))
    print('counted {} without inner holes'.format(counter))


if __name__ == '__main__':
    holesFreeTranscription()
