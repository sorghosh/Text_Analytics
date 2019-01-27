# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 20:45:10 2016

@author: Saurav Ghosh
"""

from contractions import CONTRACTION_MAP
import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
from html.parser import HTMLParser
import unicodedata
from nltk.corpus import wordnet
import numpy as np

stopword_list = nltk.corpus.stopwords.words('english')
##infra
#adhoc_stopword_list = ["ok","na","n","need","ditto","everything","infra","good","comment","approach","well","improve","poor","improvement","high","must","bad","aspect","."
#                       "availability","level","per","infrastructure","availability","require","cf"]

adhoc_stopword_list = ["ok","na","n","need","ditto","everything","infra","good","comment","approach","well","improve","poor","improvement","high","must","bad","aspect","."
                       "availability","level","per","infrastructure","availability","require","cf","proper","take","india","ensure","flow","due","handle","except","lack","reduction","reduce","also","suggest","congestion"
                       ]

for a in adhoc_stopword_list :
    stopword_list.append(a)

wnl = WordNetLemmatizer()
html_parser = HTMLParser()
lemmatizer = WordNetLemmatizer()

def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

def expand_contractions(text, contraction_mapping):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
    

# Annotate text tokens with POS tags
def pos_tag_text(text):
    
#    def penn_to_wn_tags(pos_tag):
#        if pos_tag.startswith('J'):
#            return wn.ADJ
#        elif pos_tag.startswith('V'):
#            return wn.VERB
#        elif pos_tag.startswith('N'):
#            return wn.NOUN
#        elif pos_tag.startswith('R'):
#            return wn.ADV
#        else:
#            return None
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)


    tagged_text = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]
    tagged_lower_text = [(word.lower()) for word in tagged_text]
    return tagged_lower_text
    
# lemmatize text based on POS tags    
def lemmatize_text(text):
    
    pos_tagged_text = pos_tag_text(text)
    lemmatized_text = ' '.join(pos_tagged_text)
    return lemmatized_text
    
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
      
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def keep_text_characters(text):
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def unescape_html(parser, text):
    
    return parser.unescape(text)

def normalize_accented_characters(text):
    text = unicodedata.normalize('NFKD', text.decode('utf-8')).encode('ascii', 'ignore')
    return text

#from HTMLParser import HTMLParser
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)
        
def strip_html(text):
    html_stripper = MLStripper()
    html_stripper.feed(text)
    return html_stripper.get_data()

	
def normalize_corpus(corpus, lemmatize=True, tokenize=False ,only_text_chars = False):
    normalized_corpus = []  
    for text in corpus:
        try:
            text = html_parser.unescape(text)
            text = expand_contractions(text, CONTRACTION_MAP)
            if lemmatize:
                text = lemmatize_text(text)
            text = text.lower()
            text = remove_special_characters(text)
            text = remove_stopwords(text)

            if only_text_chars:
                text = keep_text_characters(text)
        
            if tokenize:
                text = tokenize_text(text)
                normalized_corpus.append(text)
            
            normalized_corpus.append(text)
        except :
             normalized_corpus.append(np.NaN)
             pass
    return normalized_corpus


#corpus = ["The brown fox wasn't that quick and he couldn't win the race",
#          "Hey that's a great deal! I just bought a phone for $199",
#          "@@You'll (learn) a **lot** in the book. Python is an amazing language!@@"]
#
#corpus = ["The brown fox wasn't that quick and he couldn't win the race",
#          "Hey that's a great deal! I just bought a phone for $199",
#          None]
#
#
#n_corpus = normalize_corpus(corpus)
#print(n_corpus)



