from collections import defaultdict

from config import wordsRichDoublesAndUppercase, wordsSimple, wordsDoublesAndUppercase
from json import load, dump


def enrichNotation():
    """
    In config.wordsDoublesAndUppercase does not containts characteristic tokens like "d_stroke" or "s_mediana"
    this method adds these tokens by alligning bbxes/tokens from config.words with the above file (based
    on the first lecter and xCoord)
    :return: None
    """

    richDoubleUppers = defaultdict(lambda: defaultdict(list))

    print('out file: {}\n'.format(wordsRichDoublesAndUppercase))

    with open(wordsSimple, 'r') as wS, open(wordsDoublesAndUppercase, 'r') as wDU:
        richNotation = load(wS)
        doublesUppers = load(wDU)

        richTokensConv = {
            's_mediana': '1',
            's_ending': '2',
            'd_stroke': '3',
            'l_stroke': '4',
            'b_stroke': '5',
            'curl': '6'
        }

        for image, bbxes in doublesUppers.items():
            # rich token in each wordImage
            #                  token, xStart, yStart
            richNotationTks = [(b[1], b[0][0], b[0][1]) for b in richNotation[image] if b[1] in richTokensConv]
            ind2newtokens = []
            for rtoken, rxStart, ryStart in richNotationTks:
                for ind, (bb, tk) in enumerate(bbxes):
                    if bb[5] <= rxStart <= bb[6] and bb[7] <= ryStart <= bb[8] and (tk[0] == rtoken[0] or rtoken == 'curl'):
                        ind2newtokens.append((ind, richTokensConv[rtoken]))
            # updating the tokens
            for idx, newtk in ind2newtokens:
                bbxes[idx] = [bbxes[idx][0], newtk]
            # update for output
            richDoubleUppers[image] = bbxes

    with open(wordsRichDoublesAndUppercase, 'w') as out:
        dump(richDoubleUppers, out, sort_keys=True, indent=4)


if __name__ == '__main__':
    enrichNotation()
