# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 10:54:33 2019

@author: saughosh
"""

from nltk import ngrams

sentence = 'this is a good bar sentences and i want to ngramize it'

n = 3
sixgrams = ngrams(sentence.split(), n)

for grams in sixgrams:
  print (grams)