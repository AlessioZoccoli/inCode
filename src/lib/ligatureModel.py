from json import load

from nltk import ConditionalFreqDist, ConditionalProbDist, bigrams
from nltk.probability import MLEProbDist
from numpy import prod
from src.utils.textProcessing import translateToken, toNGrams
from config import connCompsJSON


class LigatureModel:

    def __init__(self):
        """
        on MLEProbDist
        The maximum likelihood estimate for the probability distribution of the experiment used to generate
        a frequency distribution. The “maximum likelihood estimate” approximates the probability of each sample
        as the frequency of that sample in the frequency distribution.
        """
        with open(connCompsJSON, 'r') as s:
            source = load(s)

        print('Creating the ligature model from: {}'.format(connCompsJSON))

        _bigrams = toNGrams(source.values(), isClean=True)
        _trigrams = [((first, sec), third) for first, sec, third in toNGrams(source.values(), n=3, isClean=True)]

        # Conditional Frequency distributions
        self.cfdBigrams = ConditionalFreqDist(_bigrams)
        self.cfdTrigrams = ConditionalFreqDist(_trigrams)

        # Conditional Probability distributions
        self.cpdBigrams = ConditionalProbDist(self.cfdBigrams, MLEProbDist)
        self.cpdTrigrams = ConditionalProbDist(self.cfdTrigrams, MLEProbDist)
        del _bigrams, _trigrams


    def getFollowers(self, token: str, gram=2):
        """
        shows tokens following gram. Use showProb to get tokens plus their related probabilities
        :param gram: number of token after 'token'
        :param token: token.
        :return: list.
        """

        maxLikehdBigrams = self.cpdBigrams[token]

        followers = [((token, sample), maxLikehdBigrams.prob(sample)) for sample in maxLikehdBigrams.samples()]

        if gram == 3:
            maxLikehdTrig = [(c[0], self.cpdTrigrams[c[0]]) for c in followers]
            followers = [((*cond, sample), ml.prob(sample)) for cond, ml in maxLikehdTrig for sample in ml.samples()]
        return followers


    def getComponentProb(self, component: str):
        """
        Markov's chain rule:

                P('pizza') =    P('p') * P('i'|'p') * P('z'|'i') * P('z'|'z') * P('a'|'z')
                                                           |
                                                          V
                                P('p'|'<s>') * P('i'|'p') * P('z'|'i') * P('z'|'z') * P('a'|'z') * P('a'|'</s>')

        :param component: string. String on which LM is queried to test probability of the same string plausibility.
        :return: float. Probability of component beign plausible for the model language in [0.0, 1.0)
        """
        comp2bigrams = bigrams([translateToken(char) if len(char) > 1 else char
                                for char in component])

        if len(component) == 1:
            prob = self.cpdBigrams['<s>'].prob(component) * self.cpdBigrams[component].prob('</s>')
        else:
            comp2bigrams = list(comp2bigrams)
            first = comp2bigrams[0][0]
            last = comp2bigrams[-1][1]

            prob = self.cpdBigrams['<s>'].prob(first) * prod(
                [self.cpdBigrams[cond].prob(evid) for evid, cond in comp2bigrams]) \
                * self.cpdBigrams[last].prob('</s>')
        return prob


    def mostFrequent(self, n=30, gram=2, flag=True):
        freqDist = self.cfdBigrams if gram == 2 else self.cfdTrigrams
        print('flag: ', flag)
        # if ('<s>' in condition[0] and flag) or ('<s>' not in condition[0])
        return sorted(
            [((*condition, sample[0]), sample[1]) for condition, samples in freqDist.items() for sample in samples.items()
             if (('<s>' in condition[0] and flag) or ('<s>' not in condition[0])) and
             (('</s>' in sample[0] and flag) or ('</s>' not in sample[0]))
             ],
            key=lambda x: -x[1])[:n]


    def conditionsAndSamples(self, gram=2):
        distrib = self.cpdBigrams if gram == 2 else self.cpdTrigrams
        output = dict()
        for cond in distrib.conditions():
            output.update({cond: [(s, distrib[cond].prob(s)) for s in distrib[cond].samples()]})

        return output
