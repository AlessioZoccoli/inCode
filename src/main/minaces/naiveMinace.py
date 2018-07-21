from src.utils.whooshUtils.indices import *

"""
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


def filterNonLatin(text):
    alphabet = {'d', 'n', 'p', 't', 'b', 'c', 'x', 'l', 'm', 's', 'i', 'a', 'u', 'o', 'q', 'g', 'h', 'f', 'e', 'r', ' '}
    return ''.join([char if char in alphabet else ' ' for char in text.lower()])


def minaceLecter():
    ix = getIndex(indexName='minacesIndex')
    fillIndex(ix)
    query(ix, "Il cammino dell uomo timorato e minacciato da ogni parte dalle iniquita degli esseri egoisti e dalla tirannia degli uomini malvagi")


if __name__ == '__main__':
    minaceLecter()
