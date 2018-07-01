"""
    Comparing 050v manual transcription with automatically generated ones

    - labelledData.csv:   contains manual transcription for 050v
    - words_clean.json:   contains automatically generated transcriptions

"""
from _csv import reader
from csv import DictWriter
from itertools import product
from json import load
from os import path, getcwd
from nltk import edit_distance


if __name__ == '__main__':

    cont = 0

    dataPath = path.join(getcwd(), '../../data/')
    automPath = path.join(dataPath, 'words_clean.json')
    labelledPath = path.join(dataPath, 'labelledData.csv')
    testPath = path.join(dataPath, 'testLabelledData.csv')

    with open(labelledPath, 'r') as lf, open(automPath, 'r') as af, open(testPath, 'w') as tst:
        labelled = reader(lf)
        autom = load(af)
        fieldnames = ("Image", "Manual", "Automatic", "Distance")
        resWriter = DictWriter(tst, fieldnames=fieldnames, extrasaction="ignore")

        # result of the test
        result = []
        dictFormat = (lambda t: {fld: el for fld, el in zip(fieldnames, t)})

        for row in labelled:
            image, word = row
            try:
                autoGen = [char[1] for char in autom[image]]
                # 'ues' token is an aggregate for both 'us' and 'ue'. Thus it must be split

                if 'ues' in autoGen:
                    # esIndices is the list of all 'ues' instances
                    uesIndices = [index for index, value in enumerate(autoGen) if value == 'ues']
                    permutations = map(lambda prod: list(zip(uesIndices, prod)), product(['ue', 'us'], repeat=3))

                    possibilities = []
                    for perm in permutations:
                        _case = autoGen[:]
                        for p in perm:
                            _case[p[0]] = p[1]
                        possibilities.append(''.join(_case))

                    # associating Levenshteine's edit_distance to each possible word
                    # then picking the most similar
                    bestMatch = min([(image, word, ''.join(p), edit_distance(''.join(p), word)) for p in possibilities],
                                    key=lambda s: s[1])
                    result.append(dictFormat(bestMatch))
                else:
                    autoGenString = ''.join(autoGen)
                    result.append(dictFormat((image, word, autoGenString, edit_distance(autoGenString, word))))

            except KeyError:
                print('not found-> ', image)  # 050v/346_212_42_83.png
                continue

        print('\n####   {} labels to test   ####\n'.format(len(result)))

        # write out
        resWriter.writeheader()
        resWriter.writerows(result)
