import pickle
from pprint import pprint
from config import ligatureModelFile


if __name__ == '__main__':

        with open(ligatureModelFile, 'rb') as p:
            ligMod = pickle.load(p)


        print('\n\nprobability of {}: {} \n\n'.format('ciao', ligMod.getComponentProb('ciao')))

        print("Followers of q (bigrams):")
        pprint(ligMod.getFollowers(token='q', gram=2))

        print("\nFollowers of q (trigrams):")
        pprint(ligMod.getFollowers(token='q', gram=3))

        print('\n20 most frequent 3 grams')
        pprint(ligMod.mostFrequent(gram=3))

        print('\n20 most frequent 3 grams starting/ending flags')
        pprint(ligMod.mostFrequent(gram=3, flag=False))

