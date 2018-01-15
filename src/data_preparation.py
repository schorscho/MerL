import os
import pandas as pd
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from sklearn.model_selection import StratifiedShuffleSplit
from mercari_config import MercariConfig


def load_data(file_name, head=None):
    data = pd.read_csv(filepath_or_buffer=os.path.join(MercariConfig.DATASETS_DIR, file_name), 
                            header=0, index_col=['train_id'])
    
    if head != None:
        data = data.head(head)
        
    return data


def save_data(data, file_name):
    data.to_csv(path_or_buf=os.path.join(MercariConfig.DATASETS_DIR, file_name))


def split_data(X, y, size1):
    split = StratifiedShuffleSplit(n_splits=1, train_size=size1, random_state=None)
 
    for index1, index2 in split.split(X, y):
        if isinstance(X, pd.DataFrame):
            X1 = X.iloc[index1]
            y1 = y[index1]
            X2 = X.iloc[index2]
            y2 = y[index2]
        else:
            X1 = X[index1]
            y1 = y[index1]
            X2 = X[index2]
            y2 = y[index2]

    return X1, y1, X2, y2

        
def load_word2index(file_name, max_words_from_index=None):
    word2index = pd.read_csv(filepath_or_buffer=os.path.join(MercariConfig.DATASETS_DIR, file_name), 
                        header=0, index_col=['word'])

    word2index_h = word2index.sort_values(by='count', ascending=False).head(max_words_from_index)

    for index in word2index[word2index['word_id'] < MercariConfig.WORD_I].index:
        word2index_h.loc[index] = word2index.loc[index]

    return word2index_h


def save_word2index(word2index, file_name):
    word2index.to_csv(path_or_buf=os.path.join(MercariConfig.DATASETS_DIR, file_name))   