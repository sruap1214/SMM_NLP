#!/home/mariadurango/bin/python
"""
Created on Thu Dec  9 16:41:22 2021

@author: Santiago Rúa Pérez
"""
import pickle
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import argparse
import fasttext
import fasttext.util

# Parser input
parser = argparse.ArgumentParser()
parser.add_argument("-wemb", "--word_embedding",type=str, help="Select word embedding representation \
                    for the text (FastText, GloveNoCor, GloveCor, W2VNoCor,  W2VCor, W2VTrained)")
parser.add_argument("-c", "--cluster",type=int, help="Select root directory depend on running: 1 cluster, 0 local")
parser.add_argument("-cf", "--cluster_folder",type=str, help="Name of the folder where the data is located")
args = parser.parse_args()

if args.cluster:
    rootFolder = args.cluster_folder
else:
    rootFolder = '.'

if args.word_embedding == 'FastText':
    print("Cargando el modelo de FastText")
    ft = fasttext.load_model(rootFolder + '/models/cc.es.300.bin')
    embedding_function = ft.get_word_vector
elif args.word_embedding == 'GloveNoCor':
    print("Cargando el modelo de GloVe sin Correccion")
    from libs.GloveNoCor import embeddingGlove
    embedding_function = embeddingGlove
elif args.word_embedding == 'GloveCor':
    print("Cargando el modelo de GloVe Correccion")
    from libs.GloveCor import embeddingGlove
    embedding_function = embeddingGlove
elif args.word_embedding == 'W2VNoCor':
    print("Cargando el modelo de W2V sin Correccion")
    from libs.W2VNoCor import embeddingW2V
    embedding_function = embeddingW2V
elif args.word_embedding == 'W2VCor':
    print("Cargando el modelo de W2V Correccion")
    from libs.W2VCor import embeddingW2V
    embedding_function = embeddingW2V
elif args.word_embedding == 'W2VTrained':
    print("Cargando el modelo de W2V pre-entrenado")
    from libs.ModeloCBOW import embeddingCBOW
    embedding_function = embeddingCBOW


# Obteniendo conjuntos de entrenamiento y test
print("Obteniendo los conjuntos de entrenamiento y test")
DatasetFolder = '/data/Dataset'
X_train = pd.read_csv(rootFolder +  DatasetFolder + '/X_train.txt', header=None).stack().tolist()
X_train = [ast.literal_eval(listString) for listString in X_train]
X_test =  pd.read_csv(rootFolder +  DatasetFolder + '/X_test.txt', header=None).stack().tolist()
X_test = [ast.literal_eval(listString) for listString in X_test]
y_train = pd.read_csv(rootFolder +  DatasetFolder + '/y_train.txt', header=None).stack().tolist()
y_test = pd.read_csv(rootFolder +  DatasetFolder + '/y_test.txt', header=None).stack().tolist()

# check if the pipeline has any empty episodes and pop these episodes
emptyList_X_train = [idx for idx, val in enumerate(X_train) if len(val) == 0 ]
emptyList_X_test = [idx for idx, val in enumerate(X_test) if len(val) == 0 ]

for index in sorted(emptyList_X_train, reverse=True):
    del X_train[index]
    del y_train[index]

for index in sorted(emptyList_X_test, reverse=True):
    del X_test[index]
    del y_test[index]


def AverageEmbedding(ListOfWords, get_word_vector):
    wordVectorAverage = np.zeros(300)
    countWords = 0
    for word in ListOfWords:
        #wordVector = ft.get_word_vector(word)
        wordVector = get_word_vector(word)
        if ~np.all(wordVector==0):
            wordVectorAverage+= wordVector
            countWords+=1
    if countWords != 0 :
        wordVectorAverage/=countWords
    return wordVectorAverage

print('Obteniendo el vector de entradas (train) mediante promedio de los vectores de las palabras')
AverageEmbedding_train = [AverageEmbedding(lista_Palabras, embedding_function) for lista_Palabras in X_train if len(lista_Palabras)!=0]
AverageEmbedding_train = np.stack( AverageEmbedding_train, axis=0 )
filename = rootFolder + DatasetFolder + '/Models/features/Average_' + args.word_embedding + '_Sub_train.pkl' 
pickle.dump(AverageEmbedding_train, open(filename, 'wb'))

print('Obteniendo el vector de entradas (test)')
AverageEmbedding_test = [AverageEmbedding(lista_Palabras, embedding_function) for lista_Palabras in X_test if len(lista_Palabras)!=0]
AverageEmbedding_test = np.stack( AverageEmbedding_test, axis=0 )
filename = rootFolder + DatasetFolder + '/Models/features/Average_' + args.word_embedding + '_Sub_test.pkl' 
pickle.dump(AverageEmbedding_test, open(filename, 'wb'))



### ------- Logistic Regression    ---------####
###
###
###-----------------------------------------####
print('Sintonizando el primer clasificador - Logistic Regression')
parameters = {'solver':['newton-cg', 'lbfgs', 'sag','saga'],
              'penalty':['l2'],
              'C':[100, 10, 1.0, 0.01],
              'max_iter':[2000,4000]}

f1_score_macro = make_scorer(f1_score, average='macro')
model=GridSearchCV(estimator=LogisticRegression(), param_grid= parameters, scoring = f1_score_macro, cv=5, n_jobs=-1, verbose=10)
model.fit(AverageEmbedding_train,y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))


filename = rootFolder + DatasetFolder + '/Models/ml_model/LogReg_' + args.word_embedding + '_Average_SubF1_model.sav' 
pickle.dump(model, open(filename, 'wb'))

### -------     MLP Classifier     ---------####
###
###
###-----------------------------------------####
print('Sintonizando el segundo clasificador - MLP')
parameters = {'hidden_layer_sizes': [(100,),(200,100,),(150,50,),(250,150,50,)],
              'activation': ['relu'],
              'solver': ['adam'],
              'alpha': [0.0001, 0.05],
              'learning_rate': ['constant','adaptive','invscaling'],
              'max_iter' : [2000]}

f1_score_macro = make_scorer(f1_score, average='macro')
model=GridSearchCV(estimator= MLPClassifier(), param_grid= parameters,scoring = f1_score_macro, cv=5, n_jobs=-1, verbose=10)
model.fit(AverageEmbedding_train,y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))

filename = rootFolder + DatasetFolder + '/Models/ml_model/MLP_' + args.word_embedding + '_Average_SubF1_model.sav' 
pickle.dump(model, open(filename, 'wb'))

### ------- Support Vector Machine ---------####
###
###
###-----------------------------------------####
print('Sintonizando el tercer clasificador - SVM')
parameters = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': ['scale'],
              'kernel': ['poly', 'rbf', 'sigmoid'],
              'cache_size':[2000]}


f1_score_macro = make_scorer(f1_score, average='macro')
model=GridSearchCV(estimator= SVC(), param_grid= parameters,scoring = f1_score_macro, cv=5, n_jobs=-1, verbose=10)
model.fit(AverageEmbedding_train,y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))



filename = rootFolder + DatasetFolder + '/Models/ml_model/SVC_' + args.word_embedding + '_Average_SubF1_model.sav' 
pickle.dump(model, open(filename, 'wb'))

### -------     Random Forest      ---------####
###
###
###-----------------------------------------####
print('Sintonizando el cuarto clasificador - Random Forest')
parameters = {'max_depth': [10, None],
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 10, 20],
              'n_estimators': [100, 200, 300]}


f1_score_macro = make_scorer(f1_score, average='macro')
model=GridSearchCV(estimator= RandomForestClassifier(), param_grid= parameters,scoring = f1_score_macro, cv=5, n_jobs=-1, verbose=10)
model.fit(AverageEmbedding_train,y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))


filename = rootFolder + DatasetFolder + '/Models/ml_model/RF_' + args.word_embedding + '_Average_SubF1_model.sav' 
pickle.dump(model, open(filename, 'wb'))


### -------     KNN Classifier      ---------####
###
###
###-----------------------------------------####
print('Sintonizando el quinto clasificador - KNN')
parameters = {'n_neighbors': [1,3,5,7,9]}

f1_score_macro = make_scorer(f1_score, average='macro')
model=GridSearchCV(estimator= KNeighborsClassifier(), param_grid= parameters,scoring = f1_score_macro, cv=5, n_jobs=-1, verbose=10)
model.fit(AverageEmbedding_train,y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))


filename = rootFolder + DatasetFolder + '/Models/ml_model/KNN_' + args.word_embedding + '_Average_SubF1_model.sav' 
pickle.dump(model, open(filename, 'wb'))

