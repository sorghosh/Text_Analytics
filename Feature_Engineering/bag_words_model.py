# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 06:53:04 2018

@author: saughosh
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]

new_doc = ['loving this blue sky today']


#bag of words model
def bow_extractors(corpus, ngram_range = (1,1)):
    vectorizer = CountVectorizer(min_df = 1 , ngram_range = ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer,features

def bow_extractors_binary(corpus, ngram_range = (1,1)):
    vectorizer = CountVectorizer(min_df = 1 , ngram_range = ngram_range, binary = True)
    features = vectorizer.fit_transform(corpus)
    return vectorizer,features


def display(features,features_name):
    df = pd.DataFrame(features)
    df.columns  = features_name
    print (df)

print (" ")
print("Continious Bag of Words >>>>>>>>>>>>>>>>>>>>>>>>>")
print (" ")

###get continious bag of words
bow_vectorizer , bow_features = bow_extractors(CORPUS)
featues = bow_features.todense()
print(featues)

#extract new features from new document using built model
new_doc_features = bow_vectorizer.transform(new_doc)
new_doc_features =  new_doc_features.todense()
print(new_doc_features)

#get feature names 
feature_names = bow_vectorizer.get_feature_names()
#display 
display(featues,feature_names)
display(new_doc_features,feature_names)


print (" ")
print("Binary Bag of Words >>>>>>>>>>>>>>>>>>>>>>>>>")
print (" ")

###get binary bag of words
bow_vectorizer , bow_features = bow_extractors_binary(CORPUS)
featues = bow_features.todense()
print(featues)

#extract new features from new document using built model
new_doc_features = bow_vectorizer.transform(new_doc)
new_doc_features =  new_doc_features.todense()
print(new_doc_features)

#get feature names 
feature_names = bow_vectorizer.get_feature_names()
#display 
display(featues,feature_names)
display(new_doc_features,feature_names)
