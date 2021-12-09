#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:02:10 2021

@author: stefan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from Preprocessing import *

"""
def adjust_to_sentiment_range(model_output):
    
    return 2 * model_output - 1

def adjust_to_model_range(model_input):
    
    return (model_input + 1) / 2
"""

def train_model(model_class, train_x, train_y):
    #adjust_to_model_range(train_y)
    model_class.fit(train_x, train_y)
    return model_class

def results_model(comparison_table, col_names):
    conf_mat = confusion_matrix(comparison_table[col_names['our']], comparison_table[col_names['gold']])
    return conf_mat

def make_gold_file(gold_model, model_input, entities, gold_name):
    gold_table = pd.DataFrame()
    gold_table['Entity'] = entities
    gold_table[gold_name] = gold_model.predict(model_input)
    return gold_table

def make_comparison_file(model, model_input, gold_table, file_path, our_name):
    comparison_table = gold_table.copy()
    comparison_table[our_name] = model.predict(model_input)
    comparison_table.to_csv(file_path, sep='\t')
    return comparison_table


def main():
    sents = read_sentences('data/datasetSentences.txt')
    #print(sents.head())
    embedding_model = train_word_2_vec(list(sents['sentence']))
    preprocess(sents, 'sentence', 'data/preprocessed_sentences.csv')
    encode(sents, 'sentence', embedding_model)
    model1 = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
    model2 = AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)
    
    print(sents.head())
    # create train-validate-test split
    #make_gold_file()
    # our model
    """
    train_model()
    make_comparison_file()
    results_model()
    """
    return


if __name__ == '__main__':
    main()