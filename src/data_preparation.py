import os
import operator

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


def split_data(X, y, train_size, val_size):
    split = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=val_size, random_state=None)
 
    for train_i, val_i in split.split(X, y):
        if isinstance(X, pd.DataFrame):
            X_t = X.iloc[train_i]
            y_t = y[train_i]
            X_v = X.iloc[val_i]
            y_v = y[val_i]
        else:
            X_t = X[train_i]
            y_t = y[train_i]
            X_v = X[val_i]
            y_v = y[val_i]

    return X_t, y_t, X_v, y_v

        
def load_word2index(file_name, max_words_from_index=None):
    word2index = pd.read_csv(filepath_or_buffer=os.path.join(MercariConfig.DATASETS_DIR, file_name), 
                        header=0, index_col=['word'])

    word2index_h = word2index.sort_values(by='count', ascending=False).head(max_words_from_index)

    for index in word2index[word2index['word_id'] < MercariConfig.WORD_I].index:
        word2index_h.loc[index] = word2index.loc[index]

    return word2index_h


def save_word2index(word2index, file_name):
    word2index.to_csv(path_or_buf=os.path.join(MercariConfig.DATASETS_DIR, file_name))   
    

def walk_tokens(doc, word, start, end):
    tok_len = 0
    
    for i in range(start, end):
        tok = doc[i]

        if not tok.text in word:
            word[tok.text] = [1, MercariConfig.NON_ENTITY_TYPE]
        else:
            word[tok.text][0] += 1

        tok_len += 1
        
        #print('Token:', tok.text, tok.i)
        
    return tok_len


def walk_items(items, nlp, word):
    progress = 0
    set_len = len(items)

    max_item_len = 0

    for item in items:
        doc = nlp(item)
        tok_cnt = len(doc)
        ent_cnt = len(doc.ents)    
        item_len = 0
        tok_i = 0

        #print (doc.ents)
        #print (doc)

        for ent in doc.ents:
            item_len += walk_tokens(doc, word, tok_i, ent.start)

            tok_i = ent.end

            if not ent.text in word:
                word[ent.text] = [1, ent.label_]
            else:
                word[ent.text][0] += 1

            item_len += 1
            
            #print('Entity:', ent.text, ent.start, ent.end, ent.label_)

        item_len += walk_tokens(doc, word, tok_i, tok_cnt)

        max_item_len = item_len if max_item_len < item_len else max_item_len

        progress += 1

        if not progress % 1000:
            print("Progress: %3.2f" % (progress * 100.0 / set_len))
    
    return max_item_len


def init_word_dict():
    word = {}
    init = [0, MercariConfig.NON_ENTITY_TYPE]

    word[MercariConfig.PAD] = list(init)
    word[MercariConfig.START] = list(init)
    word[MercariConfig.OOV] = list(init)
    word[MercariConfig.REMOVED_PRICE] = list(init)
    word[MercariConfig.EMPTY_NAME] = list(init)
    word[MercariConfig.EMPTY_CAT] = list(init)
    word[MercariConfig.EMPTY_BRAND] = list(init)
    word[MercariConfig.EMPTY_DESC] = list(init)
    
    return word


def build_word2index(word):
    word2index = pd.Series(word)
    word2index = word2index.reset_index()

    word2index['word_id'] = [i for i in range(MercariConfig.WORD_I, len(word) + MercariConfig.WORD_I)]

    word2index.columns = ['word', 'comp', 'word_id']

    word2index.set_index(['word'], inplace=True)

    word2index = word2index[['word_id', 'comp']]

    word2index.at[MercariConfig.PAD, 'word_id'] = MercariConfig.PAD_I
    word2index.at[MercariConfig.START, 'word_id'] = MercariConfig.START_I
    word2index.at[MercariConfig.OOV, 'word_id'] = MercariConfig.OOV_I
    word2index.at[MercariConfig.REMOVED_PRICE, 'word_id'] = MercariConfig.REMOVED_PRICE_I
    word2index.at[MercariConfig.EMPTY_NAME, 'word_id'] = MercariConfig.EMPTY_NAME_I
    word2index.at[MercariConfig.EMPTY_CAT, 'word_id'] = MercariConfig.EMPTY_CAT_I
    word2index.at[MercariConfig.EMPTY_BRAND, 'word_id'] = MercariConfig.EMPTY_BRAND_I
    word2index.at[MercariConfig.EMPTY_DESC, 'word_id'] = MercariConfig.EMPTY_DESC_I

    word2index['count'] = word2index['comp'].map(operator.itemgetter(0))
    word2index['entity_type'] = word2index['comp'].map(operator.itemgetter(1))

    word2index = word2index[['word_id', 'entity_type', 'count']].sort_values(by='word_id')    

    return word2index