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
from sklearn.linear_model import LogisticRegression
from preprocessing import *


def make_comparison_file(model, data, file_path, our_name, proba = True):
    """

    Parameters
    ----------
    model : sentiment prediction/detection model
    data : dataframe with encoded/embedded test sentences
    file_path : path to file the dataframe with predictions will be written to
    our_name : name the column with predicted sentiments is going to have
    proba: boolean to indicate whether to call the function predict_proba if true and predict if false

    Returns
    -------
    data : dataframe with a column added that contains predicted sentiments

    """
    if proba:
        data[our_name] = model.predict_proba(np.array(data['sentence_encoded']).reshape(-1,1))[:,1]
    else:
        data[our_name] = model.predict(np.array(data['sentence_encoded']).reshape(-1,1))
    #data.to_csv(file_path, sep='|')
    return data

def preprocessing_and_embedding():
    """
    reads the file with sentences, preprocesses them and encodes/embeds them
    does the same for the sentences scraped from rottentomatoes, but does
    not concatenate the resulting dataframes, instead only returning the one 
    with the training data and writing both to different files
    Returns
    -------
    sents : a dataframe containing the sentence_ids, preprocessed sentences,
            sentence encodings/embeddings

    """
    sents = read_sentences('data/datasetSentences.txt')
    final_sents = pd.read_json('data/critic_reviews.json')
    embed_sents = list(sents['sentence'])
    embed_sents.extend(list(final_sents['content']))
    embedding_model = train_word_2_vec(embed_sents)
    preprocess(sents, 'sentence', 'data/preprocessed_sentences.csv')
    preprocess(final_sents, 'content', 'data/preprocessed_final_sentences.csv')
    encode(sents, 'sentence', embedding_model)
    encode(final_sents, 'content', embedding_model, 'data/encoded_final_sentences.csv')
    final_sents.to_csv('data/encoded_final_sentences.csv', sep='|')
    return sents

def read_encoding_and_sentiments(input_sents=None):
    """

    Parameters
    ----------
    input_sents : dataframe containing the input sentences and their encodings/embeddings
                  The default is None.
                  if set to None, reads from default file

    Returns
    -------
    input_sents : dataframe containing the input sentences, their encoding/embedding and
                  their predicted sentiments according to the BERT model

    """
    if input_sents is None:
        input_sents = pd.read_csv('data/encoded_sentences.csv', sep='|')
    sentiments = read_sentences('data/predictions.tsv')
    sentiments.drop('sentence', axis=1, inplace=True)
    input_sents = pd.merge(input_sents, sentiments, how='left', on='sentence_index')
    #input_sents.to_csv('data/model_input.csv', sep='|')
    return input_sents

def mae(model_col, label_col):
    """

    Parameters
    ----------
    model_col : column of the dataframe containing the sentiments predicted by our model
    label_col : column of the dataframe containing the sentiments predicted by BERT

    Returns
    -------
    The mean absolute error of the models' predictions compared to those of BERT

    """
    diff = abs(model_col - label_col)
    return np.average(diff.values)

def main():
    """

    Returns
    -------
    the predictions our model makes for new data

    """
    #prevent SettingWithCopyWarning message from appearing
    pd.options.mode.chained_assignment = None
    
    input_sents = preprocessing_and_embedding()
    input_sents = read_encoding_and_sentiments(input_sents)
    model1 = GradientBoostingRegressor(n_estimators=100)
    model2 = AdaBoostRegressor(n_estimators=100)
    model3 = LogisticRegression()
    train_data, test_data = train_test_split(input_sents, train_size=0.7, random_state = 42)
    
    model = model2
    model.fit(np.array(train_data['sentence_encoded']).reshape(-1,1), train_data['label'])
    comp = make_comparison_file(model, test_data, 'data/comp.csv', 'predicted_sentiment', False)
    
    print(mae(comp['predicted_sentiment'], comp['num']))
    
    final_sents = pd.read_csv('data/encoded_final_sentences.csv', sep='|')
    final_sents.drop('Unnamed: 0', axis=1, inplace=True)
    # final_sents['pred'] = model.predict_proba(np.array(final_sents['content_encoded']).reshape(-1,1))[:,1]
    final_sents['pred'] = model.predict(np.array(final_sents['content_encoded']).reshape(-1,1))
    # final_sents.to_csv('data/critic_preds.csv', sep='|')
    return final_sents['pred']


if __name__ == '__main__':
    main()