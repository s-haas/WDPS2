#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:02:10 2021

@author: stefan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from Preprocessing import *


def make_comparison_file(model, data, file_path, our_name):
    data[our_name] = model.predict(data['sentence_encoded'])
    data.to_csv(file_path, sep='|')
    return data

def preprocessing_and_embedding():
    sents = read_sentences('data/datasetSentences.txt')
    embedding_model = train_word_2_vec(list(sents['sentence']))
    preprocess(sents, 'sentence', 'data/preprocessed_sentences.csv')
    encode(sents, 'sentence', embedding_model)
    return sents

def read_encoding_and_sentiments():
    input_sents = pd.read_csv('data/encoded_sentences.csv', sep='|')
    sentiments = pd.read_csv('data/BERT_output.csv', sep='|')
    input_sents = input_sents.merge(sentiments)
    input_sents.to_csv('data/model_input.csv', sep='|')
    return input_sents

def main():
    #input_sents = preprocessing_and_embedding()
    #input_sents = pd.read_csv('data/model_input.csv', sep='|')
    input_sents = read_encoding_and_sentiments()
    
    model1 = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
    model2 = AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)
    
    train_data, test_data = train_test_split(input_sents, train_size=0.7, random_state = 42)
    model1.fit(train_data['sentence_encoded'], train_data['score'])
    make_comparison_file(model1, test_data, 'data/comp.csv', 'predicted_sentiment')
    
    
    return


if __name__ == '__main__':
    main()