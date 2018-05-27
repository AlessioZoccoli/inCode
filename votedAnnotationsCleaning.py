from collections import defaultdict
from json import dump
import pandas as pd


def getColors(colorsString):
    """
    "[""155,205,114,255"",""164,89,68,255""]" => [[155,205,114,255], [164,89,68,255]]
    :param colorsString: str. Represents a list of colors
    :return: a list of lists of int.
    """
    return colorsString.replace('["', '[[').replace('","', '],[').replace('"]', ']]')


if __name__ == '__main__':
    with open('./data/annotations_gt1.csv', 'r') as annVotesCSV, open('./data/anncolor_by_word_voted.json',
                                                                      'w') as annVotesJSON:
        annotations = pd.read_csv(annVotesCSV)
        annotations.title = annotations.title.apply(lambda x: x.split(' ')[1])
        # path extraction: eg "img/images/color_words_1_abbr/040v/1340_1687_13_12.png" => "040v/1340_1687_13_12.png"
        annotations.path = annotations.path.apply(lambda x: '/'.join(x.split('/')[-2:]))
        annotations.answer = annotations.answer.apply(lambda x: getColors(str(x)))

        # this will be the output data structure
        annotations2votes = defaultdict(lambda: defaultdict(list))

        for row in annotations.itertuples():
            annotations2votes[row.path].update({row.title: [row.answer, row.votes]})

        dump(annotations2votes, annVotesJSON, indent=4, sort_keys=True)
