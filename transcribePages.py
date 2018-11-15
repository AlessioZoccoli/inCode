from os import path
from config import transcriptedWords_holes, pagesTranscriptionsJSON, pagesTranscriptionsTXT
from math import floor
from pprint import pprint
from collections import defaultdict
from json import load, dump
from scipy.stats import mode


def imageName2coordinates(imgName):
    """
    imgNames contains the coordinates of the image inside the page, this function unpacks the name.
    :param imgName: a string formatted like 040v/1234_456_34_36
    :return: tuple. (page, x, y, height, width)
    """
    page, coords = imgName.split('/')
    x, y, height, width = coords[:-4].split('_')    # removes extension too
    return page, int(x), int(y), int(height), int(width)


def transcribePages(outJSON=True, outTXT=True):
    """
    Transcribes the content of each page keeping the ordering of the words.
    :param outJSON: output to JSON file.
    :param outTXT: output to txt file.
    :return: None
    """
    pages = defaultdict(list)
    with open(transcriptedWords_holes, 'r') as f:
        colorWords = load(f)

        # group by page
        for img, wrd in colorWords.items():
            pageNcoords = imageName2coordinates(img)
            pages[pageNcoords[0]].append((*pageNcoords[1:], wrd))

        # calculate rows height and yOffset for each page
        X, Y, HEIGHT, WIDTH, WORD = 0, 1, 2, 3, -1
        dataXpage = dict()
        for p, wrds in pages.items():
            rowHeight = mode([w[HEIGHT] for w in wrds])[0][0]
            yOffset = min(wrds, key=lambda e: e[Y])[Y]
            dataXpage.update({p: (rowHeight, yOffset)})


        # mapping word to its row
        pagesRows = defaultdict(lambda: defaultdict(list))
        for p, wrds in pages.items():
            for w in wrds:
                rowHeight = dataXpage[p][0]
                yOffset = dataXpage[p][1]/rowHeight
                row = floor(w[Y]/rowHeight-yOffset)
                pagesRows[p][row].append((w[X], w[WIDTH], w[WORD]))

        del pages


        """
        sort words on X-axis
        
           pagesRows = {
                040v:
                    {
                        1: (x, width, 'hello')
                    }
            }
        """
        # from cordiates and word to lines of text
        pagesText = defaultdict(list)

        for kpage, row2coordwords in pagesRows.items():
            pageContent = []

            for row, rwords in row2coordwords.items():
                sortedRow = sorted(rwords, key=lambda el: el[X])
                rowContent = []
                for ind, rword in enumerate(sortedRow[:-1]):
                    x, width, word = rword
                    try:
                        rowContent.append(word)
                        rowContent.append(' '*(int((sortedRow[ind+1][X] - x+width)/20)))     # spacing
                    except IndexError:
                        rowContent.append(word)
                rowContent.append(sortedRow[-1][-1])
                pageContent.append("".join(rowContent))
            pagesText[kpage] = pageContent

        del pagesRows

        # writing to file
        if outJSON:
            with open(pagesTranscriptionsJSON, 'w') as ftranscr:
                dump(pagesText, ftranscr, sort_keys=True, indent=4)
                print('### {} written'.format(ftranscr.name))

        if outTXT:
            for page, rows in pagesText.items():
                with open(path.join(pagesTranscriptionsTXT, page + '.txt'), 'w') as pagetxt:
                    for r in rows:
                        rout = r + '\n'
                        pagetxt.write(rout)

            print('### txt files written')


if __name__ == '__main__':
    transcribePages()
