from whoosh.reading import IndexReader

from src.lib.indexing import *
from pprint import pprint
from src.utils.textProcessing import filteringChars


"""
          STATS(old)

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
         
         
    Counting connected components length and their frequency (doubles counts as 2 while specials tokens as 1, the latter
    are unsplittable)

            special tokens: {'Ca', 'Ga', 'ce', 'con', 'curl', 'curly_dash', 'de', 'eg', 'epo', 'ex', 'fa',
                            'fi', 'gl', 'nt', 'pa', 'per', 'pro', 'prop', 'que', 'qui', 'rum', 'semicolon',
                             'si', 'ue', 'us'}

                    length            freq

                    1:                16885,
                    2:                5171,
                    3:                1894,
                    4:                750,
                    5:                309,
                    6:                133,
                    7:                54,
                    8:                16,
                    9:                6,
                    10:               2
         
"""


def minaceLecter():
    ix = getIndex(indexName='baselineIndex')
    text = filteringChars("Il cammino dell'uomo timorato è minacciato da ogni parte dalle iniquità degli esseri egoistie dalla tirannia degli uomini malvagi. Benedetto sia colui che nel nome della carità e della buona volontà conduce i deboli attraverso la valle delle tenebre, perché egli è in verità il pastore di suo fratello e il ricercatore dei figli smarriti. E la mia giustizia calerà sopra di loro con grandissima vendetta e furiosissimo sdegno su coloro che si proveranno ad ammorbare e infine a distruggere i miei fratelli. E tu saprai che il mio nome è quello del Signore quando farò calare la mia vendetta sopra di te", hasUppercase=True)
    # text = "1 2 3 4 5 6 A B C D E F G H I L M N O P Q R S T U X a b c d e f g h i l m n o p q r s t u x , . Ca Ga ce con curl curly_dash de eg epo ex fa fi gl nt pa per pro prop que qui rum semicolon si ue us"
    # text = "que bus qui Ce"
    char2Images, orederedComps = query(ix, text)
    print('\n\n')
    pprint(char2Images)
    print(orederedComps)


if __name__ == '__main__':
    minaceLecter()

