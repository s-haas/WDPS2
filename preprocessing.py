import nltk
import string

import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import brown

def read_csv(fpath, delimiter='|'):
    raw_data = pd.read_csv(fpath, delimiter=delimiter, header=None, index_col=1)
    raw_data.rename(columns={0:'Phrase'}, inplace=True)
    return raw_data

def read_sentences(sentence_file):
    sentences = pd.read_csv(sentence_file, sep='\t', index_col='sentence_index')
    return sentences

def preprocess(raw_df, column_name, fpath):
    raw_df[column_name] = raw_df[column_name].apply(remove_punct)
    raw_df[column_name] = raw_df[column_name].apply(lambda x: x.lower())
    
    raw_df.to_csv(fpath, delimiter='|')

def encode_sentence(sentence, model):
    # data is tokenised text
    
    tokens = [token for token in word_tokenize(sentence)]
    sent_mat = [model.wv[token] for token in word_tokenize(sentence)]
    #model = Word2Vec(tokens, min_count = 1, size = 100, window = 5)
    return

def encode(df, column_name, model):
    df[column_name + '_encoded'] = df[column_name].apply(encode_sentence, model=model)
    return

def extract_entity_sentence_pairs(): 
    return

def remove_punct(txt):
    return "".join([w for w in txt if w not in string.punctuation])

def train_word_2_vec():
    # copied
    nltk.download("brown")
    data = []
    for sent in brown.sents():
    new_sent = []
    for word in sent:
        new_word = word.lower()
        if new_word[0] not in string.punctuation:
        new_sent.append(new_word)
    if len(new_sent) > 0:
        data.append(new_sent)
    
    model = Word2Vec(
        sentences = data,
        size = 50,
        window = 10,
        iter = 20,
    )

    return model

