#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:14:02 2022

@author: Santiago Rua Perez

This code is used to print all the results in the development set
"""
import pickle

dataFolder = './data/Dataset/Models/ml_model/'

classifiers = ['LogReg_','MLP_','SVC_','RF_','KNN_']
wordEmbNames = ['FastText_','GloveNoCor_','W2VNoCor_','GloveCor_','W2VCor_','W2VTrained_']
features = ['Average_','Clustering_','PCA_']

# Print results of CountVectorizer (not Embedding model)
for classifier in classifiers:
    model = classifier + 'CountVectorizer_'
    filename = dataFolder + model + 'SubF1_model.sav'
    loadedGridSearch_result = pickle.load(open(filename, 'rb'))
    best_mean_ = loadedGridSearch_result.cv_results_['mean_test_score'][loadedGridSearch_result.best_index_]
    best_std_ = loadedGridSearch_result.cv_results_['std_test_score'][loadedGridSearch_result.best_index_]
    print('Model: {:30} Mean: {:.2f}  and  Std: {:.2f}'.format(model,best_mean_*100,best_std_*100))

# Print results of Embedding models
for feature in features:
    for wordEmbName in wordEmbNames:
        for classifier in classifiers:
            model = classifier + wordEmbName + feature
            filename = dataFolder + model + 'SubF1_model.sav'
            loadedGridSearch_result = pickle.load(open(filename, 'rb'))
            best_mean_ = loadedGridSearch_result.cv_results_['mean_test_score'][loadedGridSearch_result.best_index_]
            best_std_ = loadedGridSearch_result.cv_results_['std_test_score'][loadedGridSearch_result.best_index_]
            print('Model: {:30} Mean: {:.2f}  and  Std: {:.2f}'.format(model,best_mean_*100,best_std_*100))

