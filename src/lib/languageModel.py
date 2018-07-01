from nltk import ConditionalFreqDist, ConditionalProbDist, bigrams
from nltk.probability import MLEProbDist
from numpy import prod

from src.utils.utils import translateToken


class LanguageModel:

    def __init__(self, bigrams):
        """
        :param bigrams: bigrams ('a', 'b') or even ('l_stroke', 't_new')

        on MLEProbDist
        The maximum likelihood estimate for the probability distribution of the experiment used to generate
        a frequency distribution. The “maximum likelihood estimate” approximates the probability of each sample
        as the frequency of that sample in the frequency distribution.
        """
        self.bigrams = bigrams
        self.conditionalFreqDist = ConditionalFreqDist(self.bigrams)
        self.conditionalProbDist = ConditionalProbDist(self.conditionalFreqDist, MLEProbDist)


    def getFollowers(self, token: str, showProb=False):
        """
        shows tokens following gram. Use showProb to get tokens plus their related probabilities
        :param token: token.
        :param showProb: boolean. Shows probability of token to be followed by tokenX.
                        token -> tokenX, 0.4567
        :return: list.
        """
        maxLikehd = self.conditionalProbDist[token]
        return [(sample, maxLikehd.prob(sample)) for sample in maxLikehd.samples()] if showProb else maxLikehd.samples()


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
            prob = self.conditionalProbDist['<s>'].prob(component) * self.conditionalProbDist[component].prob('</s>')
        else:
            comp2bigrams = list(comp2bigrams)
            first = comp2bigrams[0][0]
            last = comp2bigrams[-1][1]

            prob = self.conditionalProbDist['<s>'].prob(first) \
                * prod([self.conditionalProbDist[cond].prob(evid) for evid, cond in comp2bigrams]) \
                * self.conditionalProbDist[last].prob('</s>')
        return prob
