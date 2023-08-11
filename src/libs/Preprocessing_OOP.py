# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:02:37 2021

@author: Maria Camila
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, SnowballStemmer
import re
import string 
import unicodedata
from nltk.tag import StanfordPOSTagger
import os
import nltk
nltk.download("wordnet")


class CleanText():
    def __init__(self,texto):
        self.text=texto
        
    def __Tokenizer(self):
        """
        Tokenize the text
        """
        self.words=word_tokenize(self.text)
        return self
    
    def __Remove_numbers(self):
        """
        Remove any numbers from list of  words
        """
        new_words=[]
        for word in self.words:
            new_word=re.sub('[-+]?[0-9]+', '', word)
            if len(new_word)>0:
                new_words.append(new_word)
        self.words=new_words
        return self
    
    def __Lowercase(self):
       """ list of normalized words"""
       new_words = [word.lower() for word in self.words]
       self.words = new_words
       return self
       
    
    def __Remove_punctuation(self):
        """Remove any punctuation of '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' from list of  words"""
        new_words=[]
        for word in self.words:
         new_word= re.sub('[%s]' % re.escape(string.punctuation),'',word)
         if len(new_word)>0:
             new_words.append(new_word)
        self.words=new_words
        return self
    
    def __Remove_accentmark(self):
        """Remove stop words from list of tokenized words"""
        new_words = [unicodedata.normalize("NFD", word).encode("ascii", "ignore").decode("utf-8") for word in self.words]
        self.words=new_words
        return self
    
    def __Remove_stopwords(self):
        """Remove stop words from list of tokenized words"""
        stop=stopwords.words('spanish')
        new_words = [word for word in self.words if word not in stop]
        self.words = new_words
        return self
    
    def __Stem_words(self):
       """Stem words in list of tokenized words, this try reducing a word to a stem or base from """
       stemmer = SnowballStemmer("spanish")
       stems = [stemmer.stem(word) for word in self.words]
       self.words = stems
       return self
   
    def __Lemmatize_verbs(self):
       """Lemmatize verbs in list of tokenized words into a word's lemma"""
       lemmatizer = WordNetLemmatizer()
       lemmas = [lemmatizer.lemmatize(word, pos='v') for word in self.words]
       self.words = lemmas
       return self
   
    def Preprocessing(
            self,
            numbers=False,
            lowercase=False,
            punctuation=False,
            accent_mark=False, 
            stop_words=False,
            stemming=False,
            lemmatization=False):
        """
        Preprocessing the text
        
        Parameters
        ----------
        numbers : bool, optional
            Remove numbers. The default is False.
        lowercase : bool, optional
            Convert to lowercase. The default is False.
        punctuation : bool, optional
            Remove punctuation. The default is False.
        accent_mark : bool, optional
            Remove accent mark. The default is False.
        stop_words : bool, optional
            Remove stop words. The default is False.
        stemming : bool, optional
            Stemming words. The default is False.
        lemmatization : bool, optional
            Lemmatization words. The default is False.

        Returns
        -------
        list
            list of words preprocessed.
        """
        self.__Tokenizer()
        if numbers==True:
            self.__Remove_numbers()
        if lowercase==True:
            self.__Lowercase()
        if punctuation==True:
            self.__Remove_punctuation()
        if accent_mark==True:
            self.__Remove_accentmark()
        if stop_words==True:
            self.__Remove_stopwords()
        if stemming==True:
            self.__Stem_words()
        if lemmatization==True:
            self.__Lemmatize_verbs()
        return self.words
    
           







    

     
    
            
    