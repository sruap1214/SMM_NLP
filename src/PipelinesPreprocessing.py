# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 19:40:42 2021

@author: Santiago Rua
"""
import pickle
import pandas as pd
import sys

from libs.Preprocessing_OOP import CleanText
from sklearn.model_selection import train_test_split

def data_preprocessing (ehr_df: pd.DataFrame, token_med: str)-> pd.DataFrame:
    """
    Preprocessing of the dataframe. Erase all EHR that have less than 20 
    characters and undersampling class "Sin MME"

    Parameters
    ----------
    ehr_df : pd.DataFrame
        Dataframe with the EHR
    token_med : str
        Name of the column with the EHR
        
    Returns
    -------
    df3 : pd.DataFrame
        Dataframe with the EHR preprocessed
    """


    # Erase all EHR that have less than 20 characters
    minCaracteres = 1
    ehr_df.drop(ehr_df[ehr_df[token_med].str.len() < minCaracteres].index, inplace=True)

    # Undersampling Sin MME
    df1 = ehr_df[ehr_df['mme_agrupada last']!='Sin MME']
    df2 = ehr_df[ehr_df['mme_agrupada last']=='Sin MME'].sample(n=2000,random_state=42)
    df3 = pd.concat([df1, df2])

    return df3


def pipeline_execution(
    ehr_df: pd.DataFrame, 
    readmeFile: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    path_to_save: str,
    numbers: bool = True,
    lowercase: bool = True,
    punctuation: bool = True,
    accent_mark: bool = False,
    stop_words: bool = False,
    stemming: bool = False,
    lemmatization: bool = False
    ) -> None:
    """
    Process all EHR with the pipeline and save the results in a txt file

    Parameters
    ----------
    ehr_df : pd.DataFrame
        Dataframe with the EHR
    readmeFile : str
        Description of the pipeline
    train : pd.DataFrame
        Dataframe with the EHR for training
    test : pd.DataFrame
        Dataframe with the EHR for test
    y_train : pd.DataFrame
        Dataframe with the labels for training
    y_test : pd.DataFrame
        Dataframe with the labels for test
    path_to_save : str
        Path to save the results
    numbers : bool, optional
        If True, remove numbers. The default is True.
    lowercase : bool, optional
        If True, convert all characters to lowercase. The default is True.
    punctuation : bool, optional
        If True, remove punctuation. The default is True.
    accent_mark : bool, optional
        If True, remove accent mark. The default is False.
    stop_words : bool, optional
        If True, remove stop words. The default is False.
    stemming : bool, optional
        If True, stemming words. The default is False.
    lemmatization : bool, optional
        If True, lemmatization words. The default is False.

    Returns
    -------
    None.
    """
    Pipe_train = pd.Series([CleanText(texto).
                            Preprocessing(numbers=numbers,
                                        lowercase=lowercase,
                                        punctuation=punctuation,
                                        accent_mark=accent_mark,
                                        stop_words=stop_words,
                                        stemming=stemming,
                                        lemmatization=lemmatization) for texto in train ])
    Pipe_test = pd.Series([CleanText(texto).
                            Preprocessing(numbers=numbers,
                                        lowercase=lowercase,
                                        punctuation=punctuation,
                                        accent_mark=accent_mark,
                                        stop_words=stop_words,
                                        stemming=stemming,
                                        lemmatization=lemmatization) for texto in test ])
    assert train.shape[0] == len(Pipe_train)
    assert test.shape[0] == len(Pipe_test)

    Pipe_train.to_csv(f'{path_to_save}/X_train.txt', header = None,index = None)
    y_train.to_csv(f'{path_to_save}/y_train.txt', header = None, index = None)
    Pipe_test.to_csv(f'{path_to_save}/X_test.txt', header = None,index = None)
    y_test.to_csv(f'{path_to_save}/y_test.txt', header = None, index = None)
    with open(f"{path_to_save}/readme.txt", "w") as text_file:
        text_file.write(readmeFile)
    

def main():
    # Load data into a DataFrame
    df = pickle.load(open('./data/Tabla_TokensMedico/tabla_tokens_medicos.pkl', 'rb'))

    # Preprocess data
    dataC_name = 'tokens_medico'
    df_pos = data_preprocessing(df, dataC_name)

    # Se divide el conjunto entre entrenamiento (80%) y test (20%)
    outC_name = 'mme_agrupada last'
    train, test , y_train, y_test = train_test_split(df_pos[dataC_name],df_pos[outC_name],
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    shuffle=True,
                                                    stratify=df_pos[outC_name])
    
    # Pipeline
    # Characteristics: tokenizado, numbers, punctuation, lowercase
    print('Processing first pipeline')
    readmeFile = 'Note: numbers, punctuation, lowercase'
    path_to_save = './data/Red_Medico/Pipe1/'
    pipeline_execution(
        train,
        test,
        y_train,
        y_test,
        readmeFile,
        path_to_save,
        numbers=True,
        lowercase=True,
        punctuation=True,
        accent_mark=False,
        stop_words=False,
        stemming=False,
        lemmatization=False
    )

    

if __name__ == '__main__':
    main()