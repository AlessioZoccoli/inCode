from string import printable
from unicodedata import normalize


def translateToken(token):
    charDict = {
        'semicolon': ['u', 'es'],
        'ues': ['u', 'es'],
        'curl': ['m', 'u', 's'],
        'curly_dash': []
    }
    try:
        return charDict[token]
    except KeyError:
        if len(token) > 1:
            if token[1] == '_':
                return [token[0]]
            else:
                return list(token)
        else:
            return [token]


def toBigrams(ccomps):
    """
    Connected component to bigram.
    :param ccomps: list of lists, as follows: [ [full word],
                                                [[ccomp0], [ccomp1], [ccomp2]] ]     <<<----- comps
                    each full word is followed by its connected components
    :return: list of bigrams
    """
    bgrams = []

    for _, comps in ccomps:
        for comp in comps:
            if comp:
                cleanComp = ['<s>'] + [t for char in comp for t in translateToken(char) if t] + ['</s>']
                newChain = [(cleanComp[i], cleanComp[i + 1]) for i in range(len(cleanComp) - 1)]
                bgrams.extend(newChain)
    return bgrams


def filteringChars(text, subst=' '):
    """
    Given a text (str) only those characters for which a transcription exists are kept, unacceptable ones are replaced
    with a space
        'hello! vs hi!' -> 'hello   s hi '
    :param text: str. Input text.
    :param subst: str. Substitute for invalid chars
    :return: str. Filtered text
    """
    alphabet = {'d', 'n', 'p', 't', 'b', 'c', 'x', 'l', 'm', 's', 'i', 'a', 'u', 'o', 'q', 'g', 'h', 'f', 'e', 'r', ' '}
    return ''.join(char if char in alphabet else subst for char
                   in (''.join(ch if ch in printable else subst for ch in normalize('NFKD', text)).lower()))
