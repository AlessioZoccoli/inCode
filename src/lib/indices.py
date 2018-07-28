from collections import defaultdict
from random import choice
from nltk.util import ngrams
from os import path, mkdir
from whoosh import qparser
from whoosh.analysis import SimpleAnalyzer
from whoosh.index import exists_in, create_in, open_dir
from whoosh.fields import Schema, ID, NGRAMWORDS

from config import dataPath, images2ConncompEColors
from json import load


def minaceSchema():
    """
    Defining a basic schema for the index.
    fields:
        - image = will be in the query output (stored) and is unique
        - ccomps = TEXT may be queried but not returned when retrieving results
    """
    return Schema(
        image=ID(stored=True, unique=True),
        ccomps=NGRAMWORDS(minsize=1, maxsize=5, sortable=True)
    )


def getIndex(indexName, schema=None):
    indexPath = path.join(dataPath, indexName)
    _schema = schema if schema else minaceSchema()

    if not path.exists(indexPath):
        mkdir(indexPath)
    if exists_in(indexPath):
        ix = open_dir(indexPath)
        print("Index {} already exists\n".format(indexName))
    else:
        ix = create_in(indexPath, schema=_schema)
        print("New index {}".format(indexName))
        fillIndex(ix)                               # indexing documents
        print("filling the index\n")
    return ix


def fillIndex(index):
    """
    Parameters of the indexed document:
    image: path
    spelledWord: 1-grams forming each word
    ccompsLong: "longer" connected components, 1-grams will be only searched in spelledWord

    :param index: index
    :return: None
    """
    with open(images2ConncompEColors, 'r') as icc:
        data = load(icc)

    writer = index.writer(procs=4, limitmb=128)
    for image, values in data.items():
        writer.update_document(
            image=image,
            ccomps=values['ccomps']
        )
    writer.commit()


def find(parser, searcher, pattern, taken):
    """

    :param parser: query parser
    :param searcher: index.searcher
    :param pattern: a.k.a token
    :param taken: already collected patterns
    :return: out. List of tuples (found pattern, image name)

                                            hello
                                        /           \
                                       he           llo
                                    /     \        /    \
                                  h        e      l      lo
                                                        / \
                                                       l   o
    """
    out = []

    def findRec(p):
        if p != '':
            if p in taken or p in out:
                out.append((p, '_'))    # '_' taken cc, but we want to keep cc ordering
            else:
                q = parser.parse(p + '*')
                result = searcher.search(q, limit=4)
                if result:
                    image = choice(list(result))['image']
                    out.append((p, image))
                else:
                    half = round(len(p) / 2)
                    findRec(p[:half])
                    findRec(p[half:])

    findRec(pattern)
    return out

"""
def query(index, text):
    char2Images = defaultdict(str)  # eg. 'a': 'path/image.png'
    orderedComps = []   # 'h', 'e', 'll', 'o'

    with index.searcher() as searcher:
        qp = qparser.QueryParser('ccomps', index.schema)
        qp.add_plugin(qparser.RegexPlugin())
        analyze = SimpleAnalyzer()

        for token in analyze(text):
            t = token.text
            if t not in char2Images.keys():
                result = find(qp, searcher, t, char2Images.keys())
                for r in result:
                    if r not in char2Images.keys() and r[1] != '_':  # check for duplicates
                        char2Images[r[0]] = r[1]  # 0=ccomp, 1=image
                    orderedComps.append(r[0])
            else:
                orderedComps.append(t)
            orderedComps.append(' ')  # space between words

    return char2Images, orderedComps
"""


def query(index, text):
    char2Images = defaultdict(str)  # eg. 'a': 'path/image.png'
    orderedComps = []   # 'h', 'e', 'll', 'o'

    with index.searcher() as searcher:
        qp = qparser.QueryParser('ccomps', index.schema)
        qp.add_plugin(qparser.RegexPlugin())
        analyze = SimpleAnalyzer()

        for token in analyze(text):
            t = token.text
            if t not in char2Images:
                # first, we search for all possible n-grams for a given token
                allGrams = []
                for n in range(len(t)):
                    for ngram in ngrams(t, len(t)-n):
                        allGrams.append(''.join(str(i) for i in ngram))

                """
                positional indices for grams
                this will be used to find the "not longest" substring
                (length substr, offset Left, substr)
                """
                indexGrams = zip(
                    [(n+1, j) for n in range(len(t)) for j in range(len(t)-n)[::-1]][::-1],
                    allGrams
                )

                # then we search the longest matching substring
                longestSubString = ''
                coord = None
                for lenStart, gram in indexGrams:
                    if gram not in char2Images:
                        q = qp.parse(gram + '*')
                        result = searcher.search(q, limit=4)
                        if result:
                            char2Images.update({gram: choice(list(result))['image']})
                            coord, longestSubString = lenStart, gram
                            break
                    else:
                        coord, longestSubString = lenStart, gram
                        break

                # rest of the string/token
                leftMiss = t[:coord[1]]
                rightMiss = t[coord[1]+coord[0]:]

                if leftMiss:
                    result = find(qp, searcher, leftMiss, char2Images)
                    for r in result:
                        if r not in char2Images.keys() and r[1] != '_':  # check for duplicates
                            char2Images[r[0]] = r[1]  # 0=ccomp, 1=image
                        orderedComps.append(r[0])

                orderedComps.append(longestSubString)

                if rightMiss:
                    result = find(qp, searcher, rightMiss, char2Images)
                    for r in result:
                        if r not in char2Images.keys() and r[1] != '_':  # check for duplicates
                            char2Images[r[0]] = r[1]  # 0=ccomp, 1=image
                        orderedComps.append(r[0])
            else:
                orderedComps.append(t)
            orderedComps.append(' ')  # space between words
        orderedComps.pop()  # removes last space

    return char2Images, orderedComps




"""
 [((7, 0), 'cammino'),
 ((6, 0), 'cammin'),
 ((6, 1), 'ammino'),
 ((5, 0), 'cammi'),
 ((5, 1), 'ammin'),
 ((5, 2), 'mmino'),
 ((4, 0), 'camm'),
 ((4, 1), 'ammi'),
 ((4, 2), 'mmin'),
 ((4, 3), 'mino'),
 ((3, 0), 'cam'),
 ((3, 1), 'amm'),
 ((3, 2), 'mmi'),
 ((3, 3), 'min'),
 ((3, 4), 'ino'),
 ((2, 0), 'ca'),
 ((2, 1), 'am'),
 ((2, 2), 'mm'),
 ((2, 3), 'mi'),
 ((2, 4), 'in'),
 ((2, 5), 'no'),
 ((1, 0), 'c'),
 ((1, 1), 'a'),
 ((1, 2), 'm'),
 ((1, 3), 'm'),
 ((1, 4), 'i'),
 ((1, 5), 'n'),
 ((1, 6), 'o')]
"""