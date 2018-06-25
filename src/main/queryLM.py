from os import path, getcwd
import pickle


if __name__ == '__main__':

        pickleFile = path.join(getcwd(), 'data/ngrams.pkl')

        with open(pickleFile, 'rb') as p:
            langModel = pickle.load(p)

        # print(langModel.getComponentProb('ae'))
        print(langModel.conditionalProbDist.conditions())