# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:41:22 2021

@author: María Camila 
"""
import pickle
import numpy as np

modelo= pickle.load(open('./models/EmbeddingCBOW.pkl', 'rb'))
Vectores=modelo['Vectors']
Palabras=modelo['Words']


def EmbeddingCBOW(word):
    try:
        Vector=Vectores[Palabras.index(word),:]
    except:
        Vector=np.zeros([300,])
    return Vector


def AverageEmbedding(ListOfWords):
    wordVectorAverage = np.zeros(300)
    countWords = 0
    for word in ListOfWords:
        wordVector = EmbeddingCBOW(word, Vectores, Palabras)
        if ~np.all(wordVector==0):
            wordVectorAverage+= wordVector
            countWords+=1
    if countWords != 0 :
        wordVectorAverage/=countWords
    return wordVectorAverage
