from json import load
from os import path, getcwd
from pickle import dump
from pprint import pprint

from src.lib.image2word import toBigrams
from src.lib.languageModel import LanguageModel

if __name__ == '__main__':

    dataPath = path.join(getcwd(), '../../data/')
    connectedCompsFile = path.join(dataPath, 'connectedComps.json')
    ngramsFile = path.join(dataPath, 'ngrams.pkl')

    with open(connectedCompsFile, 'r') as cc:
        connectedComps = load(cc)

    ccomps = connectedComps.values()
    lm = LanguageModel(toBigrams(ccomps))

    with open(ngramsFile, 'wb') as ngramsPickle:
        dump(lm, ngramsPickle)