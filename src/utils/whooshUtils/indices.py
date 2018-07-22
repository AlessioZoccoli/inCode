from collections import defaultdict
from os import path, mkdir
from whoosh import qparser
from whoosh.analysis import SimpleAnalyzer
from whoosh.index import exists_in, create_in, open_dir
from whoosh.fields import Schema, ID, TEXT, NGRAMWORDS

from config import dataPath, images2ConncompEColors
from pprint import pprint
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
        print("New index {}\n".format(indexName))
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
    :return: None
    """
    out = []

    def findRec(p):
        if p != '':
            if p in taken or p in out:
                out.append((p, '_'))    # '_' taken but we want to keep cc ordering
            else:
                q = parser.parse(p + '*')
                result = searcher.search(q)
                if result:
                    image = result[0]['image']
                    out.append((p, image))
                else:
                    half = round(len(p) / 2)
                    findRec(p[:half])
                    findRec(p[half:])

    findRec(pattern)
    return out


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

    pprint(char2Images)
    print(orderedComps)
