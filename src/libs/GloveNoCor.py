#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:48:50 2022

@author: sruap
"""
import numpy as np
import pickle

print('Reading GloVe Vectors...')
print('This task may take few minutes...')

with open('./models/GloveVectors.pkl', 'rb') as handle:    
    embebimientoGlove = pickle.load(handle)
    
with open('./models/GloveWords.pkl', 'rb') as handle:
    palabrasGlove = pickle.load(handle)
    
print('Done...')   
#Función de embebimiento
def embeddingGlove(palabra):
    try:
        vector=embebimientoGlove[palabrasGlove.index(palabra),:]
    except:
        vector=np.zeros([300,])
    return vector
