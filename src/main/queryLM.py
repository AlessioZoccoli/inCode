import pickle
from pprint import pprint
from config import ligatureModelFile


if __name__ == '__main__':

        with open(ligatureModelFile, 'rb') as p:
            ligMod = pickle.load(p)

        print(ligMod.getComponentProb('ciao'), '\n\n')

        print("Followers of q (bigrams:")
        pprint(ligMod.getFollowers(token='q', gram=2))

        print('\n20 most frequent 3 grams')
        pprint(ligMod.mostFrequent(gram=3))
