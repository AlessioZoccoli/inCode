from json import load, dump
from pprint import pprint

from config import *
from src.utils.doublesAndUppercaseDetection import doublesRules, upperCaseRules


DOUBLE = 2
SINGLE = 1
DELETE = 0


def doublesAndUppercase(wordsInput):

    for img, bbxes in wordsInput.items():
        deletable = []
        if img == '051r/540_1571_43_122.png':
            print(img)
            print(len(bbxes[0][1]) == 1, bbxes[0][1].islower(), upperCaseRules[bbxes[0][1]](bbxes[0]))
        if len(bbxes[0][1]) == 1 and bbxes[0][1].islower():
            char = bbxes[0][1]
            if upperCaseRules[bbxes[0][1]](bbxes[0]):
                bbxes[0][1] = char.upper()

        if len(bbxes) > 1:
            for index, elem in enumerate(bbxes[1:]):
                tailLength = len(bbxes[1:]) - 1
                if len(elem[1]) == 1 and elem[1].islower():
                    isDouble = doublesRules[elem[1]](elem)
                    # ending 'e's are usually bigger
                    if index == tailLength and elem[1] == 'e':
                        continue
                    elif isDouble == DOUBLE:
                        elem[1] = elem[1]*2
                    elif isDouble == DELETE:
                        deletable.append(elem)
        for el in deletable:
            bbxes.remove(el)

    return wordsInput


if __name__ == '__main__':

    with open(wordsClean, 'r') as w:
        words = load(w)

    words = doublesAndUppercase(words)
    # with open(wordsDoublesAndUppercase, 'w') as output:
    #    dump(words, output, indent=4, sort_keys=True)
