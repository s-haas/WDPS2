#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:02:10 2021

@author: stefan
"""

import string

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler

"""
def read_csv(fpath, delimiter='|'):
    raw_data = pd.read_csv(fpath, delimiter=delimiter, header=None, index_col=1)
    raw_data.rename(columns={0:'Phrase'}, inplace=True)
    return raw_data
"""
def read_sentences(sentence_file):
    """

    Parameters
    ----------
    sentence_file : path to tsv file containing sentences and sentence ids

    Returns
    -------
    sentences : pandas dataframe made from the sentence file

    """
    sentences = pd.read_csv(sentence_file, sep='\t', index_col='sentence_index')
    return sentences

def preprocess(raw_df, column_name, fpath):
    """

    Parameters
    ----------
    raw_df : dataframe containing 'raw' sentences
    column_name : name of the column in the dataframe containing 'raw' sentences
    fpath : path to file where the adjusted dataframe is written to

    Returns
    -------
    preprocesses the sentences in the given column name and replaces them with
    the preprocessed sentences

    """
    raw_df[column_name] = raw_df[column_name].apply(remove_punct)
    raw_df[column_name] = raw_df[column_name].apply(lambda x: x.lower())
    
    #raw_df.to_csv(fpath, sep='|')
    return

def encode_sentence(sentence, model):
    """

    Parameters
    ----------
    sentence : sentence to be encoded/embedded
    model : model used for encoding/embedding
            in this case word2vec

    Returns
    -------
    sentence encoded/embedded into a number

    """
    scaler = MinMaxScaler(feature_range = (0,1))

    sent_mat = [model.wv[token] for token in word_tokenize(sentence)]
    scaler.fit(sent_mat)
    sent_mat = scaler.transform(sent_mat)
    return sum(np.add.reduce(sent_mat))

def encode(df, column_name, model, fpath='data/encoded_sentences.csv'):
    """

    Parameters
    ----------
    df : dataframe containing the sentence information
    column_name : name of the column containing the preprocessed sentences
    model : model used for encoding/embedding
            in this case word2vec
    fpath : path to file where the adjusted dataframe is written to
            The default is 'data/encoded_sentences.csv'.

    Returns
    -------
    encodes/embeds all sentences to be used for the model

    """
    df[column_name + '_encoded'] = df[column_name].apply(encode_sentence, model=model)
    #df.to_csv(fpath, sep='|')
    return

def remove_punct(txt):
    """

    Parameters
    ----------
    txt : sentence

    Returns
    -------
    sentence with punctuation characters removed

    """
    return "".join([w for w in txt if w not in string.punctuation])

def train_word_2_vec(input_sentences):
    """

    Parameters
    ----------
    input_sentences : array of sentences to be encoded/embedded

    Returns
    -------
    model : word2vec model trained with all sentences

    """
    lower = [sent.lower() for sent in input_sentences]
    without_punct = [remove_punct(sent) for sent in lower]
    dataset = [word_tokenize(sent) for sent in without_punct]
    data = dataset
    
    model = Word2Vec(
        sentences = data,
        vector_size = 50,
        window = 10,
        epochs = 20,
        min_count = 0
    )

    return model