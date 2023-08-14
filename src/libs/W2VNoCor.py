#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:48:50 2022

@author: sruap
"""
import numpy as np
import pickle

print('Reading W2V Vectors...')
print('This task may take few minutes...')

with open('./models/W2VVectors.pkl', 'rb') as handle:
    embebimientoW2V = pickle.load(handle)
    
with open('./models/W2VWords.pkl', 'rb') as handle:
    palabrasW2V = pickle.load(handle)
    
print('Done...')   
#Función de embebimiento
def embeddingW2V(palabra):
    try:
        vector=embebimientoW2V[palabrasW2V.index(palabra),:]
    except:
        vector=np.zeros([300,])
    return vector
