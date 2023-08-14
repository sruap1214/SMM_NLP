# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:47:46 2022

@author: Santiago Rua Perez
"""
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import ast
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-algo", "--algorithm",type=str, help="Select algorithm representation \
                    for the text (Average, Clustering, PCA)")
parser.add_argument("-word_emb", "--word_embedding",type=str, help="Select word embedding representation \
                    for the text (FastText, GloveNoCor, W2VNoCor, GloveCor, W2VCor, W2VTrained)")
parser.add_argument("-ml_model", "--ml_model",type=str, help="Select machine learning model to plot \
                    (LogReg, MLP, SVC, RF, KNN)")

args = parser.parse_args()


dataFolder = './data/Dataset/'
nameModel = args.ml_model + '_' + args.word_embedding + '_' + args.algorithm + '_SubF1_model.sav'

y_test_ = pd.read_csv(dataFolder + '/y_test.txt', header=None).stack().tolist()


# Opening features inputs for the model
fileFeatures_test = dataFolder + '/Models/features/' + args.algorithm + '_' + args.word_embedding + '_Sub_test.pkl'
Features_test = pickle.load(open(fileFeatures_test, 'rb'))

labelsOutput = ['Otra causa',
                'Sepsis de origen no obstetrico',
                'Enf. Preexistente que se complica', 
                'Sepsis de origen obstetrico', 
                'Complicaciones Hemorragicas', 
                'Sin MME', 
                'Complicaciones de aborto', 
                'Trastornos Hipertensivos'
                ]
dic_labelsOutput2 = {'Otra causa' : 'OC', 
                    'Sepsis de origen no obstetrico' : 'NS', 
                    'Enf. Preexistente que se complica' : 'PI', 
                    'Sepsis de origen obstetrico' : 'OS', 
                    'Complicaciones Hemorragicas' : 'HC', 
                    'Sin MME' : 'WS', 
                    'Complicaciones de aborto' : 'MC', 
                    'Trastornos Hipertensivos' : 'HD'}
list_labelsOutput2 = [dic_labelsOutput2[item] for item in labelsOutput]

# ---------    Machine Learning Model  -------------------------
#
#------------------------------------------------------------
# Opening Train and Test datasets
root_MachineLearningModel = dataFolder + '/Models/ml_model/' + nameModel

print('Resultado para el modelo de Machine Learning')
loadedPipe_MLModel = pickle.load(open(root_MachineLearningModel, 'rb'))
predictions = loadedPipe_MLModel.best_estimator_.predict(Features_test)
accuracy_model = accuracy_score(y_test_, predictions)
print('accuracy_score: {}'.format(accuracy_model))
F1_model = f1_score(y_test_, predictions, average = 'macro') 
print('F1_score: {}'.format(F1_model))
print(confusion_matrix(y_test_, predictions))

disp = ConfusionMatrixDisplay.from_predictions(
        y_test_,
        predictions,
        labels = labelsOutput,
        normalize = 'true',
        cmap = plt.cm.Blues,
        values_format = ".3f",
        colorbar = False
    )

disp.ax_.set_ylabel(ylabel='True label',rotation=90, ha='right',fontsize=12, fontname = 'Times New Roman')
disp.ax_.set_xlabel(xlabel='Predicted label',rotation=0, ha='right',fontsize=12, fontname = 'Times New Roman')
disp.ax_.set_yticklabels(list_labelsOutput2,rotation=0, ha='right',fontsize=12, fontname = 'Times New Roman')
disp.ax_.set_xticklabels(list_labelsOutput2,rotation=45, ha='right',fontsize=12, fontname = 'Times New Roman')
plt.subplots_adjust(left=0, bottom=0.433, right=0.820, top=0.88, wspace=0.2, hspace=0)
fig = plt.gcf()
fig.set_size_inches(9, 7)

for labels in disp.text_.ravel():
    labels.set_fontsize(10)
    labels.set_fontname('Times New Roman')

fig.savefig('./imag/fig_ConfMatrixNormalized_' + args.ml_model + '_' + args.word_embedding + '_' + args.algorithm + '.pdf', format="pdf", bbox_inches="tight")