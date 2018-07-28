from src.lib.indices import *
from pprint import pprint

from src.utils.textProcessing import filteringChars


"""
          STATS

    length      count
    of cc
         1:     17343,
         2:     5469,
         3:     1940,
         4:     764,
         5:     380,
         6:     132,
         7:     64,
         8:     21,
         9:     10,
         10:    2
         
"""


def minaceLecter():
    ix = getIndex(indexName='minacesIndex')
    text = filteringChars("Il cammino dell'uomo timorato è minacciato da ogni parte dalle iniquità degli esseri egoisti e dalla tirannia degli uomini malvagi. Benedetto sia colui che nel nome della carità e della buona volontà conduce i deboli attraverso la valle delle tenebre, perché egli è in verità il pastore di suo fratello e il ricercatore dei figli smarriti. E la mia giustizia calerà sopra di loro con grandissima vendetta e furiosissimo sdegno su coloro che si proveranno ad ammorbare e infine a distruggere i miei fratelli. E tu saprai che il mio nome è quello del Signore quando farò calare la mia vendetta sopra di te")
    char2Images, orederedComps = query(ix, text)

    pprint(char2Images)
    print(orederedComps)


if __name__ == '__main__':
    minaceLecter()