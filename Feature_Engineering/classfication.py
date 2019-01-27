# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 11:30:16 2019

@author: saughosh
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from normalization import normalize_corpus
from feature_extractors import bow_extractor, tfidf_extractor
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings("ignore")


def get_data():
    data = fetch_20newsgroups(subset='all',
                              shuffle=True,
                              remove=('headers', 'footers', 'quotes'))
    return data
    
def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels, 
                                                        test_size=0.33, random_state=42)
    return train_X, test_X, train_Y, test_Y

def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels
    
    
dataset = get_data()

print (dataset.target_names)

corpus, labels = dataset.data, dataset.target
corpus, labels = remove_empty_docs(corpus, labels)

print ('Sample document:', corpus[10])
print ('Class label:',labels[10])
print ('Actual class label:', dataset.target_names[labels[10]])

train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus[:50],
                                                                        labels[:50],
                                                                        test_data_proportion=0.3)


norm_train_corpus = normalize_corpus(train_corpus)
norm_test_corpus = normalize_corpus(test_corpus) 

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)  
bow_test_features = bow_vectorizer.transform(norm_test_corpus) 

# tfidf features
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)  
tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)    


def get_metrics(true_labels, predicted_labels):
    
    print (metrics.classification_report(true_labels,predicted_labels))
#    print ('Accuracy:', np.round(
#                        metrics.accuracy_score(true_labels, 
#                                               predicted_labels),
#                        2))
#    print ('Precision:', np.round(
#                        metrics.precision_score(true_labels, 
#                                               predicted_labels,
#                                               average='weighted'),
#                        2))
#    print ('Recall:', np.round(
#                        metrics.recall_score(true_labels, 
#                                               predicted_labels,
#                                               average='weighted'),
#                        2))
#    print ('F1 Score:', np.round(
#                        metrics.f1_score(true_labels, 
#                                               predicted_labels,
#                                               average='weighted'),
#                        2))
                        

def train_predict_evaluate_model(classifier, 
                                 train_features, train_labels, 
                                 test_features, test_labels):
    # build model    
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features) 
    # evaluate model prediction performance   
    get_metrics(true_labels=test_labels, 
                predicted_labels=predictions)
    return predictions  




mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge', n_iter=100)

# Multinomial Naive Bayes with bag of words features
mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                           train_features=bow_train_features,
                                           train_labels=train_labels,
                                           test_features=bow_test_features,
                                           test_labels=test_labels)

# Support Vector Machine with bag of words features
svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=bow_train_features,
                                           train_labels=train_labels,
                                           test_features=bow_test_features,
                                           test_labels=test_labels)

cm = metrics.confusion_matrix(test_labels, mnb_bow_predictions)
  


# Multinomial Naive Bayes with tfidf features                                           
mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                           train_features=tfidf_train_features,
                                           train_labels=train_labels,
                                           test_features=tfidf_test_features,
                                           test_labels=test_labels)

# Support Vector Machine with tfidf features
svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=tfidf_train_features,
                                           train_labels=train_labels,
                                           test_features=tfidf_test_features,
                                           test_labels=test_labels)

cm = metrics.confusion_matrix(test_labels, svm_tfidf_predictions)


