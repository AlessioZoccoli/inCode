from os import getcwd, path
from json import load, dump
from src.utils import utils
from collections import Counter
from pprint import pprint


if __name__ == '__main__':

    wordsFile = path.join(getcwd(), '../../data/words.json')
    with open(wordsFile, 'r') as f:
        words = load(f)

    bigrams = Counter()
    trigrams = Counter()

    # BIGRAMS
    bigramsList = [list(zip(el, el[1:])) for el in words if len(el) > 2]
    for wordGrams in bigramsList:
        for bigram in wordGrams:
            bigrams.update([bigram])

    totalBG = float(sum(bigrams.values()))
    bigramsProbability = list(map(lambda g: (g[0], g[1]/totalBG), bigrams.items()))

    # TRIGRAMS
    trigramsList = [list(zip(el, el[1:], el[:2])) for el in words if len(el) > 3]
    for wordGrams in trigramsList:
        for trigram in wordGrams:
            trigrams.update([trigram])

    totalTG = float(sum(trigrams.values()))
    trigramsProbability = list(map(lambda g: (g[0], g[1]/totalTG), trigrams.items()))

    # print(sum([el[1] for el in bigramsProbability]))   # 0.9999999999999989
    # print(sum([el[1] for el in trigramsProbability]))  # 1.0000000000000007

    resPath = path.join(getcwd(), '../../data/ngrams_probability.json')

    with open(resPath, 'w') as ngramsProb:
        dump({'bigrams': bigramsProbability,
              'trigrams': trigramsProbability}, ngramsProb, indent=4)
