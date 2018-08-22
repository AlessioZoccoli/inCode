from string import printable
from unicodedata import normalize
from nltk import ngrams


def translateToken(token, prev=''):
    charDict = {
        'curl': 'us',
        'curly_dash': ''
    }
    try:
        return charDict[token]
    except KeyError:
        if len(token) > 1:
            if token == 'semicolon':
                if prev == 'q':
                    return 'ue'
                elif prev == 'b':
                    return 'us'
                else:
                    return token
            elif token[1] == '_':
                return token[0]
            else:
                return token
        else:
            return token


def toNGrams(ccomps, n=2, isClean=False):
    """
    Connected component to n-grams.
    :param ccomps: list of lists:

                    [
                        [
                            [ccomp0],     ->       ['h','e'],
                            [ccomp1],     ->       ['l','l'],
                            [ccomp2]      ->       ['o']]
                        ]
                    ]
                    ,
                    ....
    :param n: int. Gram size.
    :param isClean: boolean. If isClean is False then use translateToken
    :return: list of ngrams
    """
    grams = []

    for word in ccomps:
        for subcomp in word:
            chain = ['<s>']  # beginning of the word
            prevToken = ''
            for token in subcomp:
                if not isClean:
                    token = translateToken(token, prevToken)
                prevToken = token
                chain.append(token)
            chain.append('</s>')
            grams.extend(ngrams(chain, n))

    return grams


def filteringChars(text, subst=' ', isLower=True):
    """
    Given a text (str) only those characters for which a transcription exists are kept, unacceptable ones are replaced
    with a space
        'hello! vs hi!' -> 'hello   s hi '
    :param text: str. Input text.
    :param subst: str. Substitute for invalid chars
    :param isLower: boolean. Filtering out upper case chars.
    :return: str. Filtered text
    """
    alphabet = {'d', 'n', 'p', 't', 'b', 'c', 'x', 'l', 'm', 's', 'i', 'a', 'u', 'o', 'q', 'g', 'h', 'f', 'e', 'r', ' ',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'X'}

    filtering = ''.join(char if char in alphabet else subst for char
                        in (''.join(ch if ch in printable else subst for ch in normalize('NFKD', text))))
    return filtering.lower() if isLower else filtering
