from json import load, dump
from config import *


if __name__ == '__main__':

    """
    Joining <annotationsCleanJSON> and <wordsDoublesAndUppercase> so that a file associating sorted, clean tokens to its
    colors is produced
    
    
     output:
        
            'path/image.png': {
                    'ccomps': ['i':[xCentr, yCentr, area, width, height, xStart, xEnd, yStart, yEnd], ...],
                    'colors': {'i': [[161,203,240], ...], ...}
                    }
                    
    """


    with open(wordsDoublesAndUppercase, 'r') as annFile, open(annotationsCleanJSON, 'r') as acl:
        annotWords = load(annFile)
        annotColors = load(acl)

        output = dict()
        for image, val in annotWords.items():
            entry = {
                image: {'tks': val, 'col': annotColors[image]}
            }
            output.update(entry)

        print("Storing output to {}".format(images2ColorsBBxesJSON))

        with open(images2ColorsBBxesJSON, 'w') as agg:
            dump(output, agg, indent=4, sort_keys=True)
