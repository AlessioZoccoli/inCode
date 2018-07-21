from json import load, dump
from config import *

if __name__ == '__main__':

    """
    Joining anncolor_by_word.json and connectedComps.json so that a file associating images to its
    connected components and colors to chars is produced
    
    
    Counting connected components length and their frequency:
        1: 17343,
        2: 5469,
        3: 1940,
        4: 764,
        5: 380,
        6: 132,
        7: 64,
        8: 21,
        9: 10,
        10: 2
    """

    with open(annotationsJSON, 'r') as annFile, open(connCompsJSON, 'r') as ccFile:
        annotations = load(annFile)
        connComps = load(ccFile)

        """
        'path/image.png': {
                'ccomps': 'ful l W or d',
                'annot': { 'a': [rgb1, rgb2, ..., }  
                }
        """
        output = dict()

        for image, val in connComps.items():
            entry = {
                image: {
                    'ccomps': ' '.join([''.join(comp) for comp in val[1] if len(comp) > 0]),
                    'annot': annotations[image]
                }
            }
            output.update(entry)

        with open(images2ConncompEColors, 'w') as agg:
            dump(output, agg, indent=4, sort_keys=True)
