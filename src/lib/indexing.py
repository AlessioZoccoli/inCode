from collections import defaultdict
from pprint import pprint
from random import choice

import whoosh
from nltk.util import ngrams, everygrams
from os import path, mkdir
from whoosh import qparser
from whoosh.analysis import KeywordAnalyzer, RegexAnalyzer
from whoosh.index import exists_in, create_in, open_dir
from whoosh.fields import Schema, ID, KEYWORD, STORED
from whoosh.query import Term

from config import dataPath, connCompsRichJSON, transcriptedWords_holesFree
from json import load


richTokensConv = {
            's_mediana': '1',
            's_ending': '2',
            'd_stroke': '3',
            'l_stroke': '4',
            'b_stroke': '5',
            'curl': '6',
            'qui': '7',
            'con': '8',
            'nt': '9',
            'prop': '/',
            'pro': '$',
            'per': '%',
            'semicolon': '&'
        }


def minaceSchema():
    """
    Defining a basic schema for the index.
    fields:
        image: path
        ccompsHead: Components in the middle of a transcribed word, or single-grams if thei are not in ('s', 'e', 'l', 'm')
        ccompsTail: Ending tokens
        ccompsHeadTrace: Positional index of the tokens and their compounds

    """
    # allows ['$', '%', '&', '(', '/', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    takeAllSymbolsAnalyzer = RegexAnalyzer(expression=r"[^ ]+")
    return Schema(
        image=ID(stored=True, unique=True),
        ccompsHead=KEYWORD(stored=True, sortable=True, analyzer=takeAllSymbolsAnalyzer),
        ccompsTail=KEYWORD(stored=True, sortable=True, analyzer=takeAllSymbolsAnalyzer),
        ccompsHeadTrace=STORED,
    )


def getIndex(indexName, schema=None):
    indexPath = path.join(dataPath, indexName)
    _schema = schema if schema else minaceSchema()

    print('indexPath    ', indexPath)

    if not path.exists(indexPath):
        mkdir(indexPath)
    if exists_in(indexPath):
        ix = open_dir(indexPath)
        print("###  Index {} already exists\n".format(indexName))
    else:
        ix = create_in(indexPath, schema=_schema)
        print("###  New index {}".format(indexName))
        fillIndex(ix)
        print("###  Filling the index\n")
    return ix


def fillIndex(index):
    """
    :param index: index
    :return: None

        example:
            Indexing image '055r/526_989_46_120.png' which contains the word 'fiducia'.
            Let's assume some glyphs are written with a legature producing connected components:

                fi du cia => connected_components = [[f,i], [d, u], [c,i,a]]
            
            We want to index for each image both the single glyphs and the connected components.


                        ('055r/526_989_46_120.png',
                            (('du', (((1, 0), (1, 1)),)),
                            ('f', (((0, 0),),)),
                            ('fi', (((0, 0), (0, 1)),)),
                            ('i', (((0, 1),), ((2, 1),))),
                            ('c', (((2, 0),),)),
                            ('ci', (((2, 0), (2, 1)),)),
                            ('d', (((1, 0),),)),
                            ('cia', (((2, 0), (2, 1), (2, 2)),)),
                            ('ia', (((2, 1), (2, 2)),)),
                            ('a', (((2, 2),),)),
                            ('u', (((1, 1),),))),
                            (-1, -1))
                        )


            In the code above each connected component eg. 'du' is represented by the same string
            and the coordinates inside the connected_components data structure as (comp, char in position n)
            (-1,-1) beacause there is no special connected component (abbreviation) at the end.
            Some glyphs have a different appearence when written at the end.


            Each connected components is what we have called ccompsHead in the schema definition,
            while ccompsHeadTrace is associated with the coordinates.

    """

    with open(connCompsRichJSON, 'r') as icc, open(transcriptedWords_holesFree, 'r') as thf:
        data = load(icc)
        holesFree = load(thf)

    assert holesFree
    writer = index.writer(procs=4, limitmb=128)

    for image, annot in data.items():
        if image in holesFree:
            tail = ""
            aLength = len(annot)
            headFiltered = []

            """
                "060v/1244_1049_33_94.png": [
                                                [
                                                    "s",
                                                    "e"
                                                ],
                                                [              _          _
                                                    "d",        |   tail   |    head  
                                                    "e",        |          |
                                                                |         -          
                                                    "s"         |
                                                ]              -
                                            ],
    
            """

            # lastComp = [None]
            if image != '049v_586_258_1366_1821/476_742_47_93.png':  # 'eque u'
                # nont ending components/ ending component
                for i, comp in enumerate(annot):
                    if i < aLength - 1:
                        headFiltered.append(comp)
                    else:
                        if comp[-1] in ('s', 'e', 'l', 't', 'm', 'n', 'q&', 'b&', '1', '2'):  # these MUST be in tail
                            tail = comp[-1]
                            if comp[:-1]:
                                headFiltered.append(comp[:-1])
                        elif comp == ['&']:
                            if annot[i-1][-1] in {'b', 'q'}:
                                tail = ''+annot[i-1][-1]+'&'
                            else:
                                tail = '&'
                        elif comp[-2:] in (['q', '&'], ['b', '&']):
                            tail = "".join(comp[-2:])
                            if comp[:-1]:
                                headFiltered.append(comp[:-1])
                        else:
                            headFiltered.append(comp)
            else:
                headFiltered = ['e']
                tail = "que u"

            # mapping each non ending token to its position in the annotation
            headPositions = [
                [[g for g in gram] for gram in everygrams([(ind, i) for i in range(len(headComp))], max_len=10)]
                for ind, headComp in enumerate(headFiltered)]

            # possible combinations (ngrams) for connected components
            hCombinations = tuple(
                [tuple(["".join([headFiltered[comb[0]][comb[1]] for comb in combs]) for combs in combsCC]) for combsCC in
                 headPositions])

            # indexable string
            head = " ".join([" ".join([h for h in hcomp]) for hcomp in hCombinations])

            # each possible char/compound in head to its position
            # (char_i, position in values) -> cctrace = (char_i, [positionS in values])
            ch2position = [subdict for subdict in [list(zip(el[0], list(el[1])))
                                                   for el in zip(hCombinations, headPositions)]]
            ccTrace = defaultdict(list)
            for c2p in ch2position:
                for ch, pos in c2p:
                    ccTrace[ch].append(pos)

            writeArgs = {'image': image}
            if head:
                writeArgs.update({'ccompsHead': head})
                writeArgs.update({'ccompsHeadTrace': tuple(
                    (k, tuple(v)) for k, v in ccTrace.items() if v)})  # tuple are pickable, lists are not
            if tail:
                writeArgs.update({'ccompsTail': tail})

            writer.add_document(**writeArgs)
    writer.commit()


def query(index, text, forceHead=False):
    """

    :param index: whoosh index
    :param text: str. Text so search in the index
    :param forceHead: bool. Only hallow non ending tk to be searched
    :return: a map and a list.
            The map has as keys string or substrings of text retrieved by the index and as values the images
            that contains the glyphs/connected components and these last as strings.


            Input: query("ae")
            Output: 

    """
    assert isinstance(forceHead, bool)
    char2Images = defaultdict(list)  # eg. 'h': 'path/image.png'
    orderedComps = []  # 'h', 'e', 'll', 'o'

    with open(connCompsRichJSON, 'r') as ccfile:
        inputTextCC = load(ccfile)

    getSubTokens = (lambda imgNcoord: [inputTextCC[imgNcoord[0]][cmp][pos] for cmp, pos in imgNcoord[1]])

    with index.searcher() as searcher:
        # qpHead = qparser.QueryParser('ccompsHead', index.schema)
        # qpTail = qparser.QueryParser('ccompsTail', index.schema)
        # qpHead.remove_plugin_class(qparser.WildcardPlugin)
        # qpTail.remove_plugin_class(qparser.WildcardPlugin)
        # qpHead.add_plugin(qparser.RegexPlugin())
        # qpTail.add_plugin(qparser.RegexPlugin())
        analyze = KeywordAnalyzer()

        for token in analyze(text):
            t = token.text
            if t not in char2Images:
                # first, we search for all possible n-grams for a given token
                # allGrams is the power set of the letters that make up the word
                allGrams = []
                for n in range(len(t)):
                    for ngram in ngrams(t, len(t) - n):
                        allGrams.append(''.join(str(i) for i in ngram))

                """
                positional indices for grams
                this will be used to find the smaller substrings
                (length substr, offset Left, substr)

                """
                indexGrams = list(zip(
                    [(n + 1, j) for n in range(len(t)) for j in range(len(t) - n)[::-1]][::-1],
                    allGrams
                ))

                tmp_ordComp = []   # sublist orderedDomps for current token
                collectedChar = 0  # whole word taken
                i = 0

                while collectedChar < len(t) and i < len(indexGrams):
                    lenStart, gram = indexGrams[i]
                    _length, _start = lenStart
                    prune = True
                    inTail = False
                    endGram = ""

                    if gram not in char2Images:
                        # tail or head, -3 = prevent from taking wronk tokens
                        if _start in range(len(t) - 3, len(t)) and gram in (
                                's', 'e', 'l', 't', 'm', 'n', 'q&', 'b&', '&', '1', '2') and not forceHead:
                            q = Term("ccompsTail", gram)
                            inTail = True
                            endGram = "_" + gram
                        elif _start in range(len(t) - 3, len(t)) and gram in ('q&', 'b&', '&', '1', '2') and forceHead:
                            q = Term("ccompsTail", gram)
                            inTail = True
                            endGram = "_" + gram
                        else:
                            q = Term("ccompsHead", gram)
                        result = searcher.search(q)

                        # if gram in richTokensConv.values():
                        #     print('\n//// ', gram, '\n', result, '\n\n')

                        # handling results
                        if result and endGram not in char2Images:
                            randchoice = choice(list(result))

                            if inTail:
                                char2Images[endGram] = [randchoice['image'], list(gram)]
                                tmp_ordComp.append((lenStart, endGram))
                            else:
                                try:
                                    positions = choice(dict(randchoice['ccompsHeadTrace'])[gram])
                                except KeyError:
                                    dict(randchoice['ccompsHeadTrace'])
                                    return
                                try:
                                    char2Images[gram] = [randchoice['image'], getSubTokens((randchoice['image'], positions))]
                                except IndexError:
                                    print('randchoice ', randchoice)
                                    print('positions', positions)
                                    print(inputTextCC[randchoice['image']])
                                    print('_________\n\n')
                                    raise IndexError
                                tmp_ordComp.append((lenStart, gram))

                        elif endGram in char2Images:
                            tmp_ordComp.append((lenStart, endGram))
                            break
                        else:
                            prune = False
                    else:
                        # already taken
                        tmp_ordComp.append((lenStart, gram))

                    if prune:
                        collectedChar += _length
                        pruned = [el for el in indexGrams[i + 1:]
                                  if not (_start <= el[0][1] < _length + _start or               # start
                                          _start <= el[0][0] + el[0][1] - 1 < _length + _start)  # end
                                  ]
                        indexGrams = indexGrams[:i + 1] + pruned

                    i += 1

                orderedComps.extend([oc[1] for oc in sorted(tmp_ordComp, key=lambda x: x[0][1])])
            else:
                orderedComps.append(t)
            orderedComps.append(' ')  # space between words

        orderedComps.pop()  # removes last space
        return char2Images, orderedComps


#
#       2B tested
#
#


def find(defaultParser, searcher, pattern, taken, secondaryParser=None):
    """

    :param secondaryParser:
    :param defaultParser: query parser
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

    On picking a token:

                defaultdict(list,
                    {'per': [((0, 0),)],
                     'pert': [((0, 0), (0, 1))],
                     'pertu': [((0, 3),), ((0, 0), (0, 1), (0, 2))],
                     'pertupertu': [((0, 0), (0, 1), (0, 2), (0, 3))],
                     'r': [((1, 0),), ((1, 1),)],
                     'rr': [((1, 0), (1, 1))],
                     't': [((0, 1),)],
                     'tu': [((0, 1), (0, 2))],
                     'tupertu': [((0, 1), (0, 2), (0, 3))],
                     'u': [((0, 2),)],
                     'upertu': [((0, 2), (0, 3))]})

        choice(ccTrace['pertu']):
            ((0, 3),)    or  ((0, 0), (0, 1), (0, 2))
    """
    out = []

    def findRec(p):

        if p != '':
            if p in taken or p in out:
                out.append((p, '_'))  # '_' taken cc, but we want to keep cc ordering
            else:
                q = defaultParser.parse(p)
                result = searcher.search(q)

                if result:
                    randchoice = choice(list(result))
                    # if defaultParser is tailParser and there is a result. Tail is True
                    metadata = [randchoice['image'], [(-1, -1)]] if secondaryParser \
                        else [randchoice['image'], choice(dict(randchoice['ccompsHeadTrace'])[p])]
                    out.append((p, *metadata))

                elif not result and secondaryParser:  # not in tail, search in head
                    q = secondaryParser.parse(p)
                    result = searcher.search(q)
                    if result:
                        randchoice = choice(list(result))
                        out.append((p, randchoice['image'], choice(dict(randchoice['ccompsHeadTrace'])[p])))
                    else:
                        half = round(len(p) / 2)
                        findRec(p[:half])
                        findRec(p[half:])
                else:
                    half = round(len(p) / 2)
                    findRec(p[:half])
                    findRec(p[half:])

    findRec(pattern)
    return out


def queryRec(index, text):
    """
    >> query(MyIndex, 'ciao')

            defaultdict(<class 'list'>,
                      {'cia': ['059r/413_780_36_104.png', ((1, 2), (1, 3), (1, 4))],
                      'o': ('051r/751_1468_23_33.png', (-1, -1))})                      # clearly tail

            ['cia', 'o']

    :param index: whoosh index
    :param text: String. Input text
    :return: defaultditc, list.

    """
    char2Images = defaultdict(list)  # eg. 'a': 'path/image.png'
    orderedComps = []  # 'h', 'e', 'll', 'o'

    with open(connCompsRichJSON, 'r') as ccfile:
        inputTextCC = load(ccfile)

    getSubTokens = (lambda imgNcoord: [inputTextCC[imgNcoord[0]][cmp][pos] for cmp, pos in imgNcoord[1]])

    with index.searcher() as searcher:
        qpHead = qparser.QueryParser('ccompsHead', index.schema)
        qpTail = qparser.QueryParser('ccompsTail', index.schema)
        qpHead.add_plugin(qparser.RegexPlugin())
        qpTail.add_plugin(qparser.RegexPlugin())
        analyze = KeywordAnalyzer()

        for token in analyze(text):
            t = token.text
            if t not in char2Images:
                # first, we search for all possible n-grams for a given token
                allGrams = []
                for n in range(len(t)):
                    for ngram in ngrams(t, len(t) - n):
                        allGrams.append(''.join(str(i) for i in ngram))

                """
                positional indices for grams
                this will be used to find the "not longest" substring
                (length substr, offset Left, substr)
                """
                indexGrams = zip(
                    [(n + 1, j) for n in range(len(t)) for j in range(len(t) - n)[::-1]][::-1],
                    allGrams
                )

                # then we search the longest matching substring
                longestSubString = ''
                coord = None

                for lenStart, gram in indexGrams:
                    if gram not in char2Images:
                        q = qpHead.parse(gram)
                        result = searcher.search(q)
                        if result:
                            randchoice = choice(list(result))
                            positions = choice(dict(randchoice['ccompsHeadTrace'])[gram])
                            char2Images[gram] = [randchoice['image'], getSubTokens((randchoice['image'], positions))]
                            coord, longestSubString = lenStart, gram
                            break
                    else:
                        coord, longestSubString = lenStart, gram
                        break

                # rest of the string/token
                leftMiss = t[:coord[1]]
                rightMiss = t[coord[1] + coord[0]:]

                if leftMiss:
                    result = find(qpHead, searcher, leftMiss, char2Images)
                    for r in result:
                        if r[0] not in char2Images.keys() and r[1] != '_':  # duplicates?
                            # 0=ccomp, 1=image, 2=headtrace
                            char2Images[r[0]] = [r[1], getSubTokens((r[1], r[2]))]
                        orderedComps.append(r[0])

                # middle of the word
                orderedComps.append(longestSubString)

                if rightMiss:
                    result = find(qpTail, searcher, rightMiss, char2Images, qpHead)
                    for r in result:
                        # ('al', 'dir/name.png', ((0, 1), (0, 2)))
                        if r[0] not in char2Images.keys() and r[1] != '_':
                            if r[2] == [(-1, -1)]:
                                char2Images['_' + r[0]] = [r[1], [r[0]]]
                                orderedComps.append('_' + r[0])
                            else:
                                char2Images[r[0]] = [r[1], getSubTokens((r[1], r[2]))]
                                orderedComps.append(r[0])
                        else:
                            orderedComps.append(r[0])
            else:
                orderedComps.append(t)
            orderedComps.append(' ')  # space between words

        orderedComps.pop()  # removes last space

    return char2Images, orderedComps
