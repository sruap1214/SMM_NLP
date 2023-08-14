#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 12:01:37 2022

@author: Santiago Rúa Pérez
@objective: Obtain a dataframe to apply a A/B testing regardless the length of a note.

"""

import pandas as pd
import ast
import pickle
from libraries.GloveNoCor import embeddingGlove
from libraries.W2VNoCor import embeddingW2V
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import fligner
from scipy.stats import ks_2samp


def ImportDataModels(dataFolder, fileFeatures):
    """
    Parameters
    ----------
    dataFolder : file path to X and y test
    fileFeaturesName : file path for the features of the corresponding algorithm

    Returns
    -------
    X : test dataset
    y : true value output for the test dataset
    features : features input for the machine learning algorithm

    """
    # Loading X_test, y_test and features for the classifiers
    X =  pd.read_csv( dataFolder + '/X_test.txt', header=None).stack().tolist()
    X = [ast.literal_eval(listString) for listString in X]
    y = pd.read_csv( dataFolder + '/y_test.txt', header=None).stack().tolist()

    Features = pickle.load(open(fileFeatures, 'rb'))
    return X, y, Features

def ErrorAnalysisDataframe (X, y, features, model_file_path, embeddingFunction):
    """
    Parameters
    ----------
    X : test dataset 
    y : test values for the output dataset
    features : input vector to the classifier. If empty, then a preprocessing 
               must be done with X in order to obtain features
    model_file_path : file path to the machine learning model

    Returns
    -------
    Dataframe : a Dataframe with information with respect to the predictions.

    """
    
    # Model prediction
    print('Resultado para la prediccion del modelo')
    loadedPipe_Model = pickle.load(open(model_file_path, 'rb'))
    y_pred = loadedPipe_Model.best_estimator_.predict(features)
    F1_model = f1_score(y, y_pred, average = 'macro') 
    print('F1_score: {:.2f}'.format(F1_model*100))

    # Length of each clinical note
    LenNotesXTest = [len(note) for note in X]

    # Obtaining how many word are out of the vocabulary
    OutOfVocabularyWords = []
    for note in tqdm(X):
        count = 0
        for word in note:
            wordVector = embeddingFunction(word)
            if np.all((wordVector == 0)):
                count = count + 1
        OutOfVocabularyWords.append(count)
    
    # Generating DataFrame
    df_TestAnalysis = pd.DataFrame({'X_test' : X,
                                    'y_true' : y,
                                    'y_pred' : y_pred,
                                    'len_note' : LenNotesXTest,
                                    'oov_words' : OutOfVocabularyWords})

    df_TestAnalysis['pred_ok'] = df_TestAnalysis['y_true'] == df_TestAnalysis['y_pred']
    return df_TestAnalysis

def ErrorAnalysisPlots (df_plot):
    """
    Parameters
    ----------
    X : dataframe with the data to be plot for the error analysis 

    Returns
    -------

    """
    fig = plt.subplots(figsize=(1, 4))
    ax = sns.violinplot(x = "pred_ok", 
                        y = "len_note", 
                        data = df_plot,
                        inner = None,
                        order=[True, False])
    ax2 = sns.boxplot(x = "pred_ok", 
                      y = "len_note", 
                      data = df_plot,
                      #color = 'white', 
                      width = 0.2,
                      linewidth=1,
                      boxprops = {'zorder': 2, 'facecolor':'white'}, 
                      flierprops={'marker': 'o', 'markersize': 2},
                      showfliers = True,
                      ax = ax)
    ax.set_ylabel(ylabel='Note length',fontsize=8, fontname = 'Times New Roman')
    ax.set_xlabel(xlabel=None,rotation=0, ha='center',fontsize=8, fontname = 'Times New Roman')
    ax.set_xticklabels(['Incorrect','Correct'],rotation=45, ha='center',fontsize=8, fontname = 'Times New Roman')
    plt.yticks(fontsize=8, fontname = 'Times New Roman')
            
    fig = plt.gcf()
    fig.savefig('./imag/fig_ErrorAnalysis_' + nameModel + '_NL.pdf', 
                format="pdf", 
                bbox_inches="tight")        
            
    fig2 = plt.subplots(figsize=(1, 4))       
    ax3 = sns.violinplot(x = "pred_ok", 
                        y = "word_ratio", 
                        data = df_plot,
                        inner = None,
                        order=[True, False])
    
    ax4 = sns.boxplot(x = "pred_ok", 
                      y = "word_ratio", 
                      data = df_plot,
                      #color = 'white', 
                      width = 0.2,
                      linewidth=1,
                      boxprops = {'zorder': 2, 'facecolor':'white'}, 
                      flierprops={'marker': 'o', 'markersize': 2},
                      showfliers = True,
                      ax = ax3)
    ax3.set_ylabel(ylabel='Dictionary word ratio',fontsize=8, fontname = 'Times New Roman')
    ax3.set_xlabel(xlabel=None,rotation=0, ha='center',fontsize=8, fontname = 'Times New Roman')
    ax3.set_xticklabels(['Incorrect','Correct'],rotation=45, ha='center',fontsize=8, fontname = 'Times New Roman')
    plt.yticks(fontsize=8,fontname = 'Times New Roman')
            
    fig = plt.gcf()        
    fig.savefig('./imag/fig_ErrorAnalysis_' + nameModel + '_WR.pdf', 
                format="pdf", 
                bbox_inches="tight")






# Parser input
parser = argparse.ArgumentParser()
parser.add_argument("-algo", "--algorithm",type=str, help="Select algorithm representation \
                    for the text (Average, Clustering, PCA)")
parser.add_argument("-word_emb", "--word_embedding",type=str, help="Select word embedding representation \
                    for the text (GloveNoCor, W2VNoCor, GloveCor, W2VCor)")
parser.add_argument("-ml_model", "--ml_model",type=str, help="Select machine learning model to plot \
                    (LogReg, MLP, SVC, RF, KNN)")
args = parser.parse_args()

# Identifying dictionary
if args.word_embedding == 'GloveNoCor' or args.word_embedding == 'GloveCor':
    embeddingFunction = embeddingGlove
elif args.word_embedding == 'W2VNoCor' or args.word_embedding == 'W2VCor':
    embeddingFunction = embeddingW2V
else:
    raise NameError('Selecction of word embedding incorrect')



# Name of the save models
dataFolder = './data/Dataset/'
modelFolder = dataFolder + 'Models/ml_model/'
featuresFolder = dataFolder + 'Models/features/'
nameModel = args.ml_model + '_' + args.word_embedding + '_' + args.algorithm 
filenameModel = modelFolder + nameModel + '_SubF1_model.sav'
fileFeatures = featuresFolder + args.algorithm + '_' + args.word_embedding + '_Sub_test.pkl'

# Error analysis file
ErrorAnalysis_file = dataFolder + 'ErrorAnalysis/ErrorAnalysis_' + nameModel + '_SubF1_model.csv'

if os.path.exists(ErrorAnalysis_file):
    df_TestAnalysis = pd.read_csv(ErrorAnalysis_file, header = 0)
    df_TestAnalysis['word_ratio'] = 1 - df_TestAnalysis['oov_words']/df_TestAnalysis['len_note']
    df_error = df_TestAnalysis[df_TestAnalysis['len_note']<2500]
    ErrorAnalysisPlots(df_error)
    
    # Fligner-Killeen test for validating homoscedasticity
    stat, pValue_FlignerKillen_lennote = fligner(df_error.loc[df_error['pred_ok']==True, 'len_note'], df_error.loc[df_error['pred_ok']==False, 'len_note'])
    stat, pValue_FlignerKillen_wordratio = fligner(df_error.loc[df_error['pred_ok']==True, 'word_ratio'], df_error.loc[df_error['pred_ok']==False, 'word_ratio'])

    # Kolmogorov-Smirnov test for validating homoscedasticity
    stat, pValue_Kolmogorov_lennote = ks_2samp(df_error.loc[df_error['pred_ok']==True, 'len_note'], df_error.loc[df_error['pred_ok']==False, 'len_note'])
    stat, pValue_Kolmogorov_wordratio = ks_2samp(df_error.loc[df_error['pred_ok']==True, 'word_ratio'], df_error.loc[df_error['pred_ok']==False, 'word_ratio'])

    print('El valor p en el test de Fligner-Killen para la longitud de las notas fue: {}'.format(pValue_FlignerKillen_lennote))
    print('El valor p en el test de Fligner-Killen para la word ratio fue: {}'.format(pValue_FlignerKillen_wordratio))
    print('El valor p en el test de Kolmogorov-Smirno para la longitud de las notas fue: {}'.format(pValue_Kolmogorov_lennote))
    print('El valor p en el test de Kolmogorov-Smirno para la longitud de las notas fue: {}'.format(pValue_Kolmogorov_wordratio))

else:
    # Importing X_test, y_test and features from the machine learning model
    X_test_, y_test_, Features_test = ImportDataModels(dataFolder, fileFeatures)

    # Error analysis dataframe
    df_TestAnalysis = ErrorAnalysisDataframe(X_test_, y_test_, Features_test, filenameModel, embeddingFunction)
    df_TestAnalysis.to_csv(ErrorAnalysis_file, index=False)



