# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 07:23:35 2018

@author: saughosh
"""
''''
1.Why tf-idf & Intution behind it :
    The bag of word model is based absolute raw frequency which tends to create a baisness when building features. Key potential problem :
        words which tend to occure a lot across the documents(however,if ,else, might etc.) might overshadow the words which may not occure not frequently but might be more intresting & effective to identify specific categories

2.tf- idf
  The equation of tf- idf is term fequency(tf) * inverse document frequency(idf)
   where :
       1.tf
           a. It is similar to bag of words methodogy which implies to find raw frequency count of words in a document.
           b. Mathematical it is represented as tf(w,D) = f(wd) , f(wd) denotes frequency for word w in document D which becomes term frequency(tf)
           c. There are various other representations & computation for term frequency such as converting to binary features where 1 means the term 
              has occured and 0 meands it has not.
           d. Sometimes you can also normalize the absolute raw frequency using log or averaging frequencies.
        2.idf 
            a The equation of idf is 1 + log(c/1+df(t))
            b. c represents count of the total number of documents in a given corpus
            c. df(t) represents the fequency of the number of documents in which the term t is present
        3.  In the last step we add l2 norm of matrix also known as the euclidean norm which is the sum of the square of each term in tfid weight
            
        4.  tfid = tfid / ||tfidf|| where ||tfidf|| is the euclidean l2 norm for tfidf matrix which is square root of sum of squares of each term's tfidf weight
'''''

#let do some code with using in-built python libary

CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]

new_doc = ['loving this blue sky today']

#####################################################################################################
'''
building the model without sklearn module
'''
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

####code for bag of words
def bow_extractors(corpus):
    vectorizer  = CountVectorizer(min_df =1)
    features    = vectorizer.fit_transform(corpus)
    return vectorizer , features

def display(features,features_name):
    df = pd.DataFrame(features)
    df.columns  = features_name
    print (df)
    
    
bow_vectorizer , bow_features = bow_extractors(CORPUS)
tf     = bow_features.todense()
tf     = np.array(tf , dtype ='float64')
feature_name = bow_vectorizer.get_feature_names()
display(tf,feature_name)

##applying on the new document
new_bow_vectorizer = bow_vectorizer.transform(new_doc)
new_bow_vectorizer = new_bow_vectorizer.todense()
display(new_bow_vectorizer,feature_name)

###code for document frequencies
'''
will add 1 to document frequency to smoothen the idf values later and prevent division by zero error by assuming we have a imaginary document that has the terms once.
In this section of code we get the total count for a given document
'''
from numpy.linalg import norm
import scipy.sparse as sp

df = np.diff(sp.csc_matrix(bow_features,copy = True).indptr)
df = 1 + df 
display([df],feature_name)

##calculate idf
total_docs = 1+len(CORPUS)
idf = 1 + np.log(float(total_docs)/df) 

#now converting the idf into diagonal matrix so that we can multiple with tf matrix
total_features = bow_features.shape[1]
idf_diag = sp.spdiags(idf,diags =0 , m = total_features , n = total_features)
idf = idf_diag.todense()
#print(idf)

##compute tfidf
tfidf = tf*idf
display(np.round(tfidf,2),feature_name)

#compute l2 norm to the tfidf output
norms = norm(tfidf,axis =1)

'''
norm ouput : array([2.50494598, 4.35010407, 3.49707286, 2.88865719])

lets take one small example of how l2 nom is calculated and see if it matching with norm output
    and  beautiful  blue  cheese    is  love   sky    so   the
0  0.00       0.00   1.0    0.00  1.22  0.00  1.22  0.00  1.51
1  1.92       1.51   1.0    0.00  2.45  0.00  2.45  0.00  0.00
2  0.00       1.51   1.0    0.00  1.22  0.00  1.22  1.92  1.51
3  0.00       0.00   1.0    1.92  0.00  1.92  0.00  0.00  0.00

for the first document which is at index 0 , take the square of each elements :
    1.0^2 + 0^2 + 1^2 + 0^2 + 1.22^2 + 0^2 + 1.22 ^2 + 0^2 + 1.51^2 
    2.then do a sum total of it which will come to : 6.25 and take square root of it that will be equal to 2.501
    3.hence if we see it matches with first index value of 2.50
'''
#############################
#final tfidf output
tfidf = tfidf / norms[:,None]
print ("---------final tfidf output---------------")
display(np.round(tfidf,2),feature_name) 

#####applying on the new doc the tfidf model
nd_tf =  new_bow_vectorizer
nd_tf =  np.array(nd_tf ,dtype = 'float64')
nd_tfidf = nd_tf * idf
nd_norms = norm(nd_tfidf,axis = 1)
norm_nd_tfidf = nd_tfidf/nd_norms[:,None]
print('------Appying in the new document ------------')
print(display(np.round(norm_nd_tfidf,2),feature_name))

#######################################################################################################
'''
Now lets implement tfidf model using sklearn 
'''
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_extractor(corpus,ngram_range = (1,1)):
    vectorizer = TfidfVectorizer(min_df =1 , 
                                 norm = 'l2',
                                 smooth_idf = True,
                                 use_idf = True,
                                 ngram_range = ngram_range)
    
    features = vectorizer.fit_transform(corpus)
    
    return vectorizer ,features

tfidf_vectorizer , tfidf_features =  tfidf_extractor(CORPUS)
features = tfidf_features.todense()
feature_names = tfidf_vectorizer.get_feature_names()
print('------Using Sklearn------------')
print(display(np.round(features,2),feature_names))

#applying to new document
new_tfidf_vectorizer = tfidf_vectorizer.transform(new_doc)
print('------Using Sklearn on new doc------------')
print(display(np.round(new_tfidf_vectorizer.todense(),2),feature_names))



