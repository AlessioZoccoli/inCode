from pickle import dump
from pprint import pprint

from src.lib.ligatureModel import LigatureModel
from config import ligatureModelFile


if __name__ == '__main__':

    lm = LigatureModel()
    with open(ligatureModelFile, 'wb') as fpk:
        dump(lm, fpk)
