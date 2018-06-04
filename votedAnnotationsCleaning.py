from collections import defaultdict
from json import dump
import pandas as pd


def getColors(colorsString):
    """
    "[""155,205,114,255"",""164,89,68,255""]" => [[155,205,114], [164,89,68]]
    :param colorsString: str. Represents a list of colors
    :return: a list of lists of int.
    """
    rgbAlpha = pd.eval(colorsString.replace('["', '[[').replace('","', '],[').replace('"]', ']]'))
    return list(map(lambda col: col[:3], rgbAlpha))


def aggregateVotes(listCharVotes):
    """
    aggregateVotes flattens a list of dicts in the form of

                [{ <char1>:<voteX>}, { <char2>:<voteY>}, { <char1>:<voteZ>} ...]

    so that we have

                [{ <char1>:[<voteX>, <voteZ>]}, { <char2>:[<voteY>]} ...]

    :param listCharVotes: list of dicts. Associate a vote to each transcribed char
    :return: a list of dicts. Each dict associates a list of votes to the character
    """
    flattened = defaultdict(list)
    for entry in listCharVotes:
        key, val = list(entry.items())[0]
        flattened[key].append(val)

    return flattened


if __name__ == '__main__':
    with open('./data/annotations_gt1.csv', 'r') as annVotesCSV, open('./data/word_voted.json', 'w') as annVotesJSON:

        annotations = pd.read_csv(annVotesCSV)
        annotations.dropna(inplace=True)

        annotations.title = annotations.title.apply(lambda x: x.split(' ')[1])
        # path extraction: eg "img/images/color_words_1_abbr/040v/1340_1687_13_12.png" => "040v/1340_1687_13_12.png"
        annotations.path = annotations.path.apply(lambda x: '/'.join(x.split('/')[-2:]))
        annotations.answer = annotations.answer.apply(lambda x: getColors(x))

        imageVotes = annotations.groupby(['path', 'title']).votes.agg(['sum']).rename(columns={'sum': 'votes'}).reset_index()
        imageVotes['char2vote'] = imageVotes.apply(lambda x: {x[1]: x[2]}, axis=1)
        imageVotes = imageVotes.groupby('path').agg({'votes': ['sum'], 'char2vote': (lambda x: aggregateVotes(list(x)))})
        imageVotes.reset_index(inplace=True)
        imageVotes.columns = ['image', 'charsVotes', 'totalVotes']

        out = imageVotes.set_index('image').T.to_dict('list')
        dump(out, annVotesJSON, indent=4, sort_keys=True)
