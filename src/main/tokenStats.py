from os import getcwd, path
from json import load, dump
from src.utils import utils
from collections import Counter
from pprint import pprint


if __name__ == '__main__':

    dataPath = path.join(getcwd(), '../../data/')
    wordsFile = path.join(dataPath, 'connectedComps.json')
    with open(wordsFile, 'r') as f:
        tokens = load(f)

    ngramTokens = Counter()
    ngramTokensList = []

    for _, tks in tokens.values():
        ngramTokensList = []
        for t in tks:
            if len(t) in [1, 2, 3]:
                ngramTokensList.append(tuple(t))
            elif len(t) > 3:
                bigrams = [tuple(el) for el in list(zip(t, t[1:]))]
                trigrams = [tuple(el) for el in list(zip(t, t[1:], t[2:]))]
                ngramTokensList.extend(bigrams)
                ngramTokensList.extend(trigrams)
            ngramTokens.update(ngramTokensList)

    totalNGrams = float(sum(ngramTokens.values()))
    nGramProbability = sorted(list(map(lambda g: (g[0], g[1] / totalNGrams), ngramTokens.items())),
                              key=lambda x: x[1], reverse=True)

    # print(sum([el[1] for el in nGramProbability])) 1.0000000000000036

    with open(path.join(dataPath, 'ngramsTokens.json'), 'w') as outFile:
        dump(nGramProbability, outFile, indent=4)