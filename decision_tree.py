#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 20:55:27 2018

@author: siddartharevur
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

NUM_OF_CLASSES = [4, 9, 14, 19, 24]

COLUMN_NAMES = [
    'budget',
    'popularity',
    'runtime',
    'vote_average',
    'vote_count'
]

COLUMNS_TO_KEEP = COLUMN_NAMES

COLUMNS_TO_POP = [
    'genres',
    'homepage',
    'keywords',
    'id',
    'keywords',
    'original_language',
    'original_title',
    'overview',
    'production_countries',
    'production_companies',
    'release_date',
    'spoken_languages',
    'status',
    'tagline',
    'title'
]

all_movie_data = 0

def load_data():
    global all_movie_data, raw_data 
    all_movie_data = pd.read_csv('tmdb_5000_movies.csv', na_values=[])
    raw_data = pd.read_csv('tmdb_5000_movies.csv', na_values=[])

def filter_data(classes):
    global all_movie_data
    load_data()
    for key in all_movie_data.columns:
        if key not in COLUMNS_TO_KEEP and key != 'revenue':
            all_movie_data.pop(key)

    all_movie_data = all_movie_data[~(all_movie_data.isnull().any(axis=1))]

    #REMOVE MOVIES WITH REVENUE == 0 (0 means Not available)
    zero_revenue_indices = all_movie_data.revenue != 0
    all_movie_data = all_movie_data[zero_revenue_indices]
    
    #bin the revenue data
    all_movie_data['revenue'] = pd.cut(all_movie_data['revenue'], len(range(classes)), labels=range(classes)).cat.codes.astype(np.int64)
    train_data = all_movie_data.sample(int(0.8 * len(all_movie_data)))
    train_data_labels = train_data.pop('revenue')
    test_data = all_movie_data.drop(train_data.index)
    test_data_labels = test_data.pop('revenue')
    return (train_data, train_data_labels), (test_data, test_data_labels)

#Decision Tree Classifier with Gini Index
def main():
    global train, train_labels, test, test_labels, test_predict, test_labels_array, class_accuracies
    class_accuracies = []
    for classes in NUM_OF_CLASSES:
        (train, train_labels), (test, test_labels) = filter_data(classes)
        clf_gini = DecisionTreeClassifier(criterion = "gini", min_samples_leaf=5)
        clf_gini.fit(train, train_labels)
        test_predict = clf_gini.predict(test)
        test_labels_array = test_labels.values
        accuracy = (accuracy_score(test_labels_array, test_predict))
        class_accuracies.append(accuracy)
        print('Accuracy for Decision Tree -' + ' Num Classes = ' + str(classes) + ' = ' + str(accuracy))
    print

#Decision Tree Classifier with Information Gain
def main2():
    global train, train_labels, test, test_labels, test_predict, test_labels_array, class_accuracies
    class_accuracies = []
    for classes in NUM_OF_CLASSES:
        (train, train_labels), (test, test_labels) = filter_data(classes)
        clf_entropy = DecisionTreeClassifier(criterion = "entropy", min_samples_leaf=5)
        clf_entropy.fit(train, train_labels)
        test_predict = clf_gini.predict(test)
        test_labels_array = test_labels.values
        accuracy = (accuracy_score(test_labels_array, test_predict))
        class_accuracies.append(accuracy)
        print('Accuracy for Decision Tree -' + ' Num Classes = ' + str(classes) + ' = ' + str(accuracy))
    print
	
main2()
