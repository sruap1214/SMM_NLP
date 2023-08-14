#!/usr/bin/python3
"""
Created on Thu Nov  4 12:57:28 2021

@author: Andres
"""

import numpy as np
import pickle

print('Reading Vectors...')
print('This task may take few minutes...')

with open('./models/newgloveWords.pkl', 'rb') as handle:
    palabrasUnicas = pickle.load(handle)

with open('./models/newGloveVectors.pkl', 'rb') as handle:
    newGloveEmbeding = pickle.load(handle)

print('Ready')

# Keep some interesting statistics
NodeCount = 0
WordCount = 0

# The Trie data structure keeps a set of words, organized with one node for
# each letter. Each node has a branch for each letter that may follow it in the
# set of words.
class TrieNode:
    def __init__(self):
        self.word = None
        self.children = {}

        global NodeCount
        NodeCount += 1

    def insert( self, word ):
        node = self
        for letter in word:
            if letter not in node.children: 
                node.children[letter] = TrieNode()

            node = node.children[letter]

        node.word = word

# read dictionary file into a trie
        
print('Reading dictionary file into a trie...')
print('This task may take few minutes...')

trie = TrieNode()
for word in palabrasUnicas:
    WordCount += 1
    trie.insert( word )
    
print('Ready')


# The search function returns a list of all words that are less than the given
# maximum distance from the target word
def search( word, maxCost ):

    # build first row
    currentRow = range( len(word) + 1 )

    results = []

    # recursively search each branch of the trie
    for letter in trie.children:
        searchRecursive( trie.children[letter], letter, word, currentRow, 
            results, maxCost )

    return results

# This recursive helper is used by the search function above. It assumes that
# the previousRow has been filled in already.
def searchRecursive( node, letter, word, previousRow, results, maxCost ):

    columns = len( word ) + 1
    currentRow = [ previousRow[0] + 1 ]

    # Build one row for the letter, with a column for each letter in the target
    # word, plus one for the empty string at column 0
    for column in range( 1, columns ):

        insertCost = currentRow[column - 1] + 1
        deleteCost = previousRow[column] + 1

        if word[column - 1] != letter:
            replaceCost = previousRow[ column - 1 ] + 1
        else:                
            replaceCost = previousRow[ column - 1 ]

        currentRow.append( min( insertCost, deleteCost, replaceCost ) )

    # if the last entry in the row indicates the optimal cost is less than the
    # maximum cost, and there is a word in this trie node, then add it.
    if currentRow[-1] <= maxCost and node.word != None:
        results.append( (node.word, currentRow[-1] ) )

    # if any entries in the row are less than the maximum cost, then 
    # recursively search each branch of the trie
    if min( currentRow ) <= maxCost:
        for letter in node.children:
            searchRecursive( node.children[letter], letter, word, currentRow, 
                results, maxCost )


def searchWord(word):
    result=search(word,0)  
    if not result:
        result=search(word,1) 
        if not result:
            result=search(word,2)
            if not result:
                result=[('notFound',3)]
    minword=[]
    mindist=4
    for i in range(len(result)) :
        distancia=result[i][1]
        if distancia < mindist:
            mindist=distancia
            minword=result[i][0]
    return minword
    

def embeddingMGlove(palabra):
    wordGlove=searchWord(palabra)
    if wordGlove != 'notFound':
        vector=newGloveEmbeding[palabrasUnicas.index(wordGlove),:]
    else:
        vector=np.zeros([300,])
    return vector

