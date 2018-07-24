from json import load
from os import path, getcwd
from pickle import dump
from pprint import pprint

from src.utils.textProcessing import toBigrams
from src.lib.languageModel import LanguageModel

if __name__ == '__main__':

    dataPath = path.join(getcwd(), '../../../data/')
    connectedCompsFile = path.join(dataPath, 'connectedComps.json')
    ngramsFile = path.join(dataPath, 'ngrams.pkl')

    with open(connectedCompsFile, 'r') as cc:
        connectedComps = load(cc)

    ccomps = connectedComps.values()
    bgs = toBigrams(ccomps)
    lm = LanguageModel(bgs)

    # print(list(filter(lambda x: x[0] == '<s>' and x[1] == '</s>', bgs)))

    with open(ngramsFile, 'wb') as ngramsPickle:
        dump(lm, ngramsPickle)
