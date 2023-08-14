#!/home/mariadurango/bin/python
"""
Created on Thu Dec  9 16:41:22 2021

@author: Santiago Rua
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
from sklearn.decomposition import PCA
import argparse
from GloveCor import embeddingGlove

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pipeline",type=int, help="select pipe to execute")
args = parser.parse_args()

print("Obteniendo los conjuntos de entrenamiento y test")

X_train = pd.read_csv(r'/home/mariadurango/EmbedingAO/data/Red_Medico/Pipe'+str(args.pipeline)+'_Sub/X_train.txt', header=None).stack().tolist()
X_train = [ast.literal_eval(listString) for listString in X_train]
X_test =  pd.read_csv(r'/home/mariadurango/EmbedingAO/data/Red_Medico/Pipe'+str(args.pipeline)+'_Sub/X_test.txt', header=None).stack().tolist()
X_test = [ast.literal_eval(listString) for listString in X_test]
y_train = pd.read_csv(r'/home/mariadurango/EmbedingAO/data/Red_Medico/Pipe'+str(args.pipeline)+'_Sub/y_train.txt', header=None).stack().tolist()
y_test = pd.read_csv(r'/home/mariadurango/EmbedingAO/data/Red_Medico/Pipe'+str(args.pipeline)+'_Sub/y_test.txt', header=None).stack().tolist()

# check if the pipeline has any empty episodes and pop these episodes
emptyList_X_train = [idx for idx, val in enumerate(X_train) if len(val) == 0 ]
emptyList_X_test = [idx for idx, val in enumerate(X_test) if len(val) == 0 ]

for index in sorted(emptyList_X_train, reverse=True):
    del X_train[index]
    del y_train[index]

for index in sorted(emptyList_X_test, reverse=True):
    del X_test[index]
    del y_test[index]

def PCAEmbedding(ListOfWords):
    wordVectorAverage = np.zeros((1,300))
    countWords = 0
    if len(ListOfWords) < 2:
        for word in ListOfWords:
            #wordVector = ft.get_word_vector(word)
            wordVector = embeddingGlove(word)
            if ~np.all(wordVector==0):
                wordVectorAverage+= wordVector
                countWords+=1
        if countWords != 0 :
            wordVectorAverage/=countWords
        return wordVectorAverage
    else:
        X = np.zeros([len(ListOfWords),300])
        for idx, word in enumerate(ListOfWords):
            X[idx] = embeddingGlove(word)
        pca = PCA(n_components=1).fit(X)
        PCAVector = pca.components_
        return PCAVector


print('Obteniendo el vector de entradas mediante PCA de palabras')
PCA_Embedding_train = [PCAEmbedding(lista_Palabras) for lista_Palabras in X_train if len(lista_Palabras)!=0]
PCA_Embedding_train = np.squeeze(np.stack( PCA_Embedding_train, axis=0 ))
filename = '/home/mariadurango/EmbedingAO/data/Red_Medico/Pipe'+str(args.pipeline)+'_Sub/Models/PCA_EmbGloveCor_Sub_train.pkl'
pickle.dump(PCA_Embedding_train, open(filename, 'wb'))
print(f'El tamano del train es {PCA_Embedding_train.shape}')

PCA_Embedding_test = [PCAEmbedding(lista_Palabras) for lista_Palabras in X_test if len(lista_Palabras)!=0]
PCA_Embedding_test = np.squeeze(np.stack( PCA_Embedding_test, axis=0 ))
filename = '/home/mariadurango/EmbedingAO/data/Red_Medico/Pipe'+str(args.pipeline)+'_Sub/Models/PCA_EmbGloveCor_Sub_test.pkl'
pickle.dump(PCA_Embedding_test, open(filename, 'wb'))
print(f'El tamano del test es {PCA_Embedding_test.shape}')


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
model=GridSearchCV(estimator=LogisticRegression(), param_grid= parameters, scoring = f1_score_macro, cv=5, n_jobs=-1, verbose=True)
model.fit(PCA_Embedding_train,y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))


#predictions=model.predict(CountingClusterEmbedding_test)
#print('f1_macro_score :')
#accuracy= f1_score(y_test, predictions, 'macro')
#print(accuracy)

filename = '/home/mariadurango/EmbedingAO/data/Red_Medico/Pipe'+str(args.pipeline)+'_Sub/Models/LogReg_GloveCor_PCA_SubF1_model.sav'
pickle.dump(model, open(filename, 'wb'))
#loaded_LogReg = pickle.load(open(filename, 'rb'))

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
model=GridSearchCV(estimator= MLPClassifier(), param_grid= parameters,scoring = f1_score_macro, cv=5, n_jobs=-1, verbose=True)
model.fit(PCA_Embedding_train,y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))

#predictions=model.predict(CountingClusterEmbedding_test)
#print('f1_macro_score :')
#accuracy= f1_score(y_test, predictions, 'macro')
#print(accuracy)


filename = '/home/mariadurango/EmbedingAO/data/Red_Medico/Pipe'+str(args.pipeline)+'_Sub/Models/MLP_GloveCor_PCA_SubF1_model.sav'
pickle.dump(model, open(filename, 'wb'))
#loaded_MLP = pickle.load(open(filename, 'rb'))



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
model=GridSearchCV(estimator= SVC(), param_grid= parameters,scoring = f1_score_macro, cv=5, n_jobs=-1, verbose=True)
model.fit(PCA_Embedding_train,y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))


filename = '/home/mariadurango/EmbedingAO/data/Red_Medico/Pipe'+str(args.pipeline)+'_Sub/Models/SVC_GloveCor_PCA_SubF1_model.sav'
pickle.dump(model, open(filename, 'wb'))
#loaded_SVM = pickle.load(open(filename, 'rb'))


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
model=GridSearchCV(estimator= RandomForestClassifier(), param_grid= parameters,scoring = f1_score_macro, cv=5, n_jobs=-1, verbose=True)
model.fit(PCA_Embedding_train,y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))


filename = '/home/mariadurango/EmbedingAO/data/Red_Medico/Pipe'+str(args.pipeline)+'_Sub/Models/RF_GloveCor_PCA_SubF1_model.sav'
pickle.dump(model, open(filename, 'wb'))
#loaded_SVM = pickle.load(open(filename, 'rb'))


### -------     KNN Classifier      ---------####
###
###
###-----------------------------------------####
print('Sintonizando el quinto clasificador - KNN')
parameters = {'n_neighbors': [1,3,5,7,9]}


f1_score_macro = make_scorer(f1_score, average='macro')
model=GridSearchCV(estimator= KNeighborsClassifier(), param_grid= parameters,scoring = f1_score_macro, cv=5, n_jobs=-1, verbose=True)
model.fit(PCA_Embedding_train,y_train)
print("Best: %f using %s" % (model.best_score_, model.best_params_))


filename = '/home/mariadurango/EmbedingAO/data/Red_Medico/Pipe'+str(args.pipeline)+'_Sub/Models/KNN_GloveCor_PCA_SubF1_model.sav'
pickle.dump(model, open(filename, 'wb'))
#loaded_SVM = pickle.load(open(filename, 'rb'))

