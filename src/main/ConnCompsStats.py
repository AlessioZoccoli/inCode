from collections import defaultdict

from config import transcriptedWords_holesFree, connCompsRichJSON, connectedComponents_noHoles_stats
from json import load, dump

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


def connectedComponentsStats():

    reverseRichConv = dict(zip(richTokensConv.values(), ["<{}>".format(rvr) for rvr in richTokensConv.keys()]))

    with open(transcriptedWords_holesFree, 'r') as hf, open(connCompsRichJSON, 'r') as c:
        holesFree = load(hf)
        connectedComponents = load(c)

        conComps = defaultdict(int)
        freq2conComps = defaultdict(list)

        for comps in connectedComponents.values():
            for cmp in comps:
                cmpString = "".join([el if el not in reverseRichConv else reverseRichConv[el] for el in cmp])
                conComps[cmpString] += 1

        for cmp, freq in conComps.items():
            freq2conComps[freq].append(cmp)

    with open(connectedComponents_noHoles_stats, 'w') as fout:
        dump({'concomps2freq': conComps, 'freq2concomps': freq2conComps}, fout, sort_keys=True, indent=4)


if __name__ == '__main__':
    connectedComponentsStats()