from os import path, getcwd
import pickle
from pprint import pprint

if __name__ == '__main__':

        pickleFile = path.join(getcwd(), 'data/ngrams.pkl')

        with open(pickleFile, 'rb') as p:
            langModel = pickle.load(p)

        lmCPDist = langModel.conditionalProbDist
        lmConditions = lmCPDist.conditions()

        # print(langModel.getComponentProb('ae'))
        # print('{}\n'.format(lmConditions))

        for cond in lmConditions:
            print('\n#######  condition: {}  #######\n'.format(cond))
            pprint([(s, lmCPDist[cond].prob(s)) for s in lmCPDist[cond].samples()])
