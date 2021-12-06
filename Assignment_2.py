#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:02:10 2021

@author: stefan
"""

import pandas as pd
from sklearn.metrics import train_test_split
from sklearn.metrics import confusion_matrix


def read_sentences(sentence_file):
    sentences = pd.read_csv(sentence_file, sep='\t', index_col='sentence_index')
    return sentences

def extract_entity_sentence_pairs():
    
    return

def adjust_to_sentiment_range(model_output):
    
    return 2 * model_output - 1

def adjust_to_model_range(model_input):
    
    return (model_input + 1) / 2

def generate_model_input():
    
    return

def train_model(model_class, model_input):
    adjust_to_model_range(model_input)
    model_class.fit(model_input)
    return model_class

def results_model(comparison_table, col_names):
    conf_mat = confusion_matrix(comparison_table[col_names['our']], comparison_table[col_names['gold']])
    return conf_mat

def setup_gold_model():
    
    return

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
    read_sentences()
    extract_entity_sentence_pairs()
    generate_model_input()
    # create train-validate-test split
    train_test_split()
    # getting test data
    setup_gold_model()
    make_gold_file()
    # our model
    train_model()
    make_comparison_file()
    results_model()
    return


if __name__ == '__main__':
    main()