import sys
import os
import operator
import logging
import logging.config
from time import time
from datetime import datetime

import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

import spacy
from spacy.symbols import ORTH

import keras
import keras.models as km
import keras.layers as kl
import keras.constraints as kc
from keras import losses
from keras.models import load_model
from keras.preprocessing import sequence
from keras import backend as K


DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': { 
        'standard': { 
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': { 
        'default': { 
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': { 
        'root': { 
            'handlers': ['default'],
            'level': 'INFO'
        },
        'MerL': { 
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        },
    } 
}


logging.config.dictConfig(DEFAULT_LOGGING)


logger = logging.getLogger('MerL.data_preparation')


class MercariConfig:
    PROJECT_ROOT_DIR = '/home/ubuntu'
    INPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
    TF_LOG_DIR = os.path.join(OUTPUT_DIR, "tf_logs")
    TRAINING_SET_FILE = "train.tsv"
    TRAINING_SET_PREP_FILE = "mercari_train_prep.csv"
    VALIDATION_SET_PREP_FILE = "mercari_val_prep.csv"
    TEST_SET_FILE = "test.tsv"
    TEST_SET_PREP_FILE = "mercari_test_prep.csv"
    WORD_2_INDEX_4_ITEM_DESC_FILE = "mercari_word_2_index_4_item_desc.csv"
    WORD_2_INDEX_4_NAME_FILE = "mercari_word_2_index_4_name.csv"
    TRAINING_NAME_INDEX_FILE = "mercari_train_name_index.csv"
    TRAINING_ITEM_DESC_INDEX_FILE = "mercari_train_item_desc_index.csv"
    VALIDATION_NAME_INDEX_FILE = "mercari_val_name_index.csv"
    VALIDATION_ITEM_DESC_INDEX_FILE = "mercari_val_item_desc_index.csv"
    TEST_NAME_INDEX_FILE = "mercari_test_name_index.csv"
    TEST_ITEM_DESC_INDEX_FILE = "mercari_test_item_desc_index.csv"

    PAD = '___PAD___'
    START = '___START___'
    OOV = '___OOV___'
    REMOVED_PRICE = '[rm]'
    EMPTY_NAME = '___VERY_EMPTY_NAME___'
    EMPTY_CAT = '___VERY_EMPTY_CATEGORY___'
    EMPTY_BRAND = '___VERY_EMPTY_BRAND___'
    EMPTY_DESC = '___VERY_EMPTY_DESCRIPTION___'

    PAD_I = 0
    START_I = 1
    OOV_I = 2
    REMOVED_PRICE_I = 3
    EMPTY_NAME_I = 4
    EMPTY_CAT_I = 5
    EMPTY_BRAND_I = 6
    EMPTY_DESC_I = 7
    WORD_I = 8

    NON_ENTITY_TYPE = '___NONE_ENTITY___'
    
    SPACY_MODEL ='en_core_web_md'
    INCLUDE_NER = True

    MAX_WORDS_FROM_INDEX_4_ITEM_DESC = 40000
    MAX_WORDS_FROM_INDEX_4_NAME = 31000
    MAX_WORDS_IN_ITEM_DESC = 300
    MAX_WORDS_IN_NAME = 20
    
    TRAIN_SIZE = 0.16
    VAL_SIZE = 0.02
    
    DP = 'DP01R00'
    WP = 'WP01R00'
    MV = 'MV01R00'
    OV = 'OV01R01'

    @staticmethod
    def get_new_tf_log_dir():
        now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        log_dir = "{}/run-{}/".format(MercariConfig.TF_LOG_DIR, now)

        return log_dir
    
    
def time_it(start, end):
    h, r = divmod(end - start, 3600)
    m, s = divmod(r, 60)
    
    return "{:0>2}:{:0>2}:{:06.3f}".format(int(h), int(m), s)


def load_data(file_name, sep):
    data = pd.read_csv(filepath_or_buffer=os.path.join(MercariConfig.INPUT_DIR, file_name), 
                       sep=sep, header=0, index_col=None)
    
    id_column = None
    
    if 'train_id' in data.columns:
        id_column = 'train_id'
    elif 'test_id' in data.columns:
        id_column = 'test_id'
    
    if id_column is not None:
        data['item_id'] = data[id_column]
    
        data.drop(columns=[id_column], inplace=True)
                  
    data.set_index(['item_id'], inplace=True)
    
    data.sort_index(inplace=True)
    
    return data


def save_data(data, file_name):
    data.to_csv(path_or_buf=os.path.join(MercariConfig.OUTPUT_DIR, file_name))


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


def load_all_data(train_set, val_set, test_set, initialization):
    train_data = None
    val_data = None
    test_data = None
    
    sep = '\t' if initialization else ','
    
    if train_set:
        if initialization:
            logger.info("Loading initial training data ...")
        
            train_data = load_data(file_name=MercariConfig.TRAINING_SET_FILE, sep=sep)

            logger.info("Loading initial training data done.")
        else:
            logger.info("Loading prepared training data ...")
        
            train_data = load_data(file_name=MercariConfig.TRAINING_SET_PREP_FILE, sep=sep)
        
            logger.info("Loading prepared training data done.")

    if val_set:
        logger.info("Loading prepared validation data ...")

        val_data = load_data(file_name=MercariConfig.VALIDATION_SET_PREP_FILE, sep=sep)

        logger.info("Loading prepared validation data done.")

    if test_set:
        if initialization:
            logger.info("Loading initial test data ...")
            
            test_data = load_data(MercariConfig.TEST_SET_FILE, sep=sep)

            logger.info("Loading initial test data done.")
        else:
            logger.info("Loading prepared test data ...")
        
            test_data = load_data(MercariConfig.TEST_SET_PREP_FILE, sep=sep)

            logger.info("Loading prepared test data done.")

    return train_data, val_data, test_data


def save_all_prepared_data(train_data, val_data, test_data):
    if train_data is not None:
        logger.info("Saving prepared training data ...")

        save_data(train_data, MercariConfig.TRAINING_SET_PREP_FILE)

        logger.info("Saving prepared training data done.")

    if val_data is not None:
        logger.info("Saving prepared validation data ...")
    
        save_data(val_data, MercariConfig.VALIDATION_SET_PREP_FILE)

        logger.info("Saving prepared validation data done.")

    if test_data is not None:
        logger.info("Saving prepared test data ...")
    
        save_data(test_data, MercariConfig.TEST_SET_PREP_FILE)

        logger.info("Saving prepared test data done.")
    

def prepare_data(df):
    logger.info("Filling missing values ...")

    df['category_name'].fillna(value=MercariConfig.EMPTY_CAT, inplace=True)

    assert(len(df[df.category_name.isnull()]) == 0)

    df['brand_name'].fillna(value=MercariConfig.EMPTY_BRAND, inplace=True)

    assert(len(df[df.brand_name.isnull()]) == 0)

    df['item_description'].fillna(value=MercariConfig.EMPTY_DESC, inplace=True)

    assert(len(df[df.item_description.isnull()]) == 0)

    df['name'].fillna(value=MercariConfig.EMPTY_NAME, inplace=True)

    assert(len(df[df.name.isnull()]) == 0)

    logger.info("Filling missing values done.")


def execute_full_data_initialization(train_data, test_data):
    logger.info("Starting initial data preparation ...")
    
    if train_data is not None:
        logger.info("Preparing training data ...")

        prepare_data(train_data)
        
        logger.info("Preparing training data done.")

        logger.info("Splitting prepared training data ...")

        bins = np.linspace(train_data.price.min(), 1000, 10)

        train_data,_,val_data,_ = split_data(X=train_data, y=np.digitize(train_data.price, bins),
                                             train_size=MercariConfig.TRAIN_SIZE, val_size=MercariConfig.VAL_SIZE)

        logger.info("Splitting prepared training data done.")

    if test_data is not None:
        logger.info("Preparing test data ...")

        prepare_data(test_data)

        logger.info("Preparing test data done.")

    logger.info("Initial data preparation done.")    
    
    return train_data, val_data, test_data
    

def build_category2index(data, category):
    category2index = data[[category + '_name', 'name']]
    category2index.columns = [category, category + '_cnt']
    category2index = category2index.groupby(category).count()

    category2index[category + '_id'] = [i for i in range(MercariConfig.WORD_I, len(category2index) + MercariConfig.WORD_I)]

    category2index.at[MercariConfig.PAD, category + '_id'] = MercariConfig.PAD_I
    category2index.at[MercariConfig.START, category + '_id'] = MercariConfig.START_I
    category2index.at[MercariConfig.OOV, category + '_id'] = MercariConfig.OOV_I
    category2index.at[MercariConfig.REMOVED_PRICE, category + '_id'] = MercariConfig.REMOVED_PRICE_I
    category2index.at[MercariConfig.EMPTY_NAME, category + '_id'] = MercariConfig.EMPTY_NAME_I
    category2index.at[MercariConfig.EMPTY_CAT, category + '_id'] = MercariConfig.EMPTY_CAT_I
    category2index.at[MercariConfig.EMPTY_BRAND, category + '_id'] = MercariConfig.EMPTY_BRAND_I
    category2index.at[MercariConfig.EMPTY_DESC, category + '_id'] = MercariConfig.EMPTY_DESC_I

    category2index[category + '_cnt'].fillna(value=0, inplace=True)
    category2index.sort_values(by=category + '_id', inplace=True)
    category2index = category2index.astype('int32')

    return category2index


def categorize_column(data, category, category2index):
    data[category + '_id'] = 0

    data_len = len(data)

    progress = 0

    row_iterator = data.iterrows()

    for index, row in row_iterator:
        if not progress % 10000:
            logger.info("Progress: %3.2f", (progress * 100.0 / data_len))

        cat_nm = row[category + '_name']

        if cat_nm in category2index.index:
            data.at[index, category + '_id'] = category2index.at[cat_nm, category + '_id']
        else:
            data.at[index, category + '_id'] = MercariConfig.EMPTY_CAT_I

        progress += 1

                        
def execute_full_categorization(train_data, val_data, test_data, category, brand):
    logger.info("Starting categorization ...")

    train_prop = None
    val_prop = None
    test_prop = None
    
    cat_prop = None
    brand_prop = None

    if train_data is not None:
        train_prop = {'data': train_data, 'file': MercariConfig.TRAINING_SET_PREP_FILE}

    if val_data is not None:
        val_prop = {'data': val_data, 'file': MercariConfig.VALIDATION_SET_PREP_FILE}

    if test_data is not None:
        test_prop = {'data': test_data, 'file': MercariConfig.TEST_SET_PREP_FILE}

    if category:
        logger.info("Building category2index for category ...")
        
        cat2i = build_category2index(data=train_data, category='category')
        cat_prop = {'cat2i': cat2i, 'category': 'category'}

        logger.info("Building category2index for category done.")

    if brand:
        logger.info("Building category2index for brand ...")
        
        cat2i = build_category2index(data=train_data, category='brand')
        brand_prop = {'cat2i': cat2i, 'category': 'brand'}

        logger.info("Building category2index for brand done.")

    for data_prop in [train_prop, val_prop, test_prop]:
        if data_prop is not None:
            for cat2i_prop in [cat_prop, brand_prop]:
                if cat2i_prop is not None:
                    logger.info("Performing categorization for file %s and category %s ...", 
                                data_prop['file'], cat2i_prop['category'])
                    
                    categorize_column(data=data_prop['data'], 
                                      category2index=cat2i_prop['cat2i'], 
                                      category=cat2i_prop['category'])

                    logger.info("Performing categorization for file %s and category %s done.", 
                                data_prop['file'], cat2i_prop['category'])

    logger.info("Done with categorization.")    

    return train_data, val_data, test_data


def load_language_model(model_nm, include_ner):
    logger.info("Loading language model ...")
    
    disabled = ['parser', 'tagger']
    
    if (not include_ner):
        disabled.append('ner')
        
    nlp = spacy.load(model_nm, disable=disabled)  
    nlp.tokenizer.add_special_case('[rm]', [{ORTH: '[rm]'}])
    
    logger.info("Loading language model done.")

    return nlp


def walk_tokens_4_word2i(item_id, doc, pos_i, words, start, end):
    for i in range(start, end):
        tok = doc[i]

        words.append([item_id, pos_i, tok.text, MercariConfig.NON_ENTITY_TYPE])

        pos_i += 1
        
        logger.debug('Token: %s %i', tok.text, tok.i)
        
    return pos_i


def walk_items_4_word2i(items, nlp, max_words):
    then = time()
    words = []
    progress = 0
    set_len = len(items)

    if max_words is None:
        max_words = 10000
        
    max_words_len = 0

    logger.info("Walkthrough for word2i for %s ...", items.name)     

    for item_id in items.index:
        if not progress % 1000:
            logger.info("Walkthrough for word2i for %s: %3.2f %%", items.name, (progress * 100.0 / set_len))

        doc = nlp(items[item_id])
        tok_cnt = len(doc)
        pos_i = 0
        tok_i = 0

        logger.debug('Entities: %s', doc.ents)
        logger.debug('Tokens: %s', doc)

        words.append([item_id, pos_i, MercariConfig.START, MercariConfig.NON_ENTITY_TYPE]) # <START>

        pos_i = 1
        
        for ent in doc.ents:
            if pos_i <= max_words:
                pos_i = walk_tokens_4_word2i(item_id=item_id, doc=doc, pos_i=pos_i, 
                                             words=words, 
                                             start=tok_i, end=min(ent.start, tok_i + max_words - pos_i + 1))

                tok_i = ent.end

                if pos_i <= max_words:
                    logger.debug('Entity: %s %i %i %s', ent.text, ent.start, ent.end, ent.label_)

                    words.append([item_id, pos_i, ent.text, ent.label_])
                
                pos_i += 1
            else:
                break

        pos_i = walk_tokens_4_word2i(item_id=item_id, doc=doc, pos_i=pos_i, words=words, 
                                     start=tok_i, end=min(tok_cnt, tok_i + max_words - pos_i + 1))

        max_words_len = pos_i if max_words_len < pos_i else max_words_len

        words.extend(
            [[item_id, pos_i, MercariConfig.PAD , MercariConfig.NON_ENTITY_TYPE] for pos_i in range(pos_i, max_words + 1)])        

        progress += 1
        
    logger.info("Walkthrough for word2i for %s done in %s.", items.name, time_it(then, time()))
    
    return words, max_words_len


def build_word2i_4_items(items, nlp, max_words):
    words, max_words_len = walk_items_4_word2i(items=items, nlp=nlp, max_words=max_words)

    word2i = build_word2i_from_sequence_list(words=words)

    return word2i, words, max_words_len


def build_word2i_from_sequence_list(words):
    word2i = pd.DataFrame(words)

    word2i.columns = ['item_id', 'pos', 'word', 'entity_type']

    word2i = word2i.groupby(['word']).count()

    word2i.drop(columns=['pos', 'entity_type'], inplace=True)

    word2i.columns = ['word_cnt']

    word2i['word_id'] = [i for i in range(MercariConfig.WORD_I,
                                          len(word2i) + MercariConfig.WORD_I)]

    word2i.at[MercariConfig.PAD, 'word_id'] = MercariConfig.PAD_I
    word2i.at[MercariConfig.START, 'word_id'] = MercariConfig.START_I
    word2i.at[MercariConfig.OOV, 'word_id'] = MercariConfig.OOV_I
    word2i.at[MercariConfig.REMOVED_PRICE, 'word_id'] = MercariConfig.REMOVED_PRICE_I
    word2i.at[MercariConfig.EMPTY_NAME, 'word_id'] = MercariConfig.EMPTY_NAME_I
    word2i.at[MercariConfig.EMPTY_CAT, 'word_id'] = MercariConfig.EMPTY_CAT_I
    word2i.at[MercariConfig.EMPTY_BRAND, 'word_id'] = MercariConfig.EMPTY_BRAND_I
    word2i.at[MercariConfig.EMPTY_DESC, 'word_id'] = MercariConfig.EMPTY_DESC_I

    word2i['word_cnt'].fillna(value=0, inplace=True)

    word2i = word2i.astype(dtype='int32')

    word2i.sort_values('word_id', inplace=True)
    
    return word2i


def load_word2i(file_name, word2i, max_words_from_index):
    if word2i is None:
        logger.info("Loading word2index from %s ...", file_name)

        word2i = pd.read_csv(
            filepath_or_buffer=os.path.join(MercariConfig.INPUT_DIR, file_name),
            header=0, index_col=['word'])

        logger.info("Loading word2index from %s done.", file_name)

    word2i_s = word2i.sort_values(by='word_cnt', ascending=False).head(max_words_from_index)

    for index in word2i[word2i['word_id'] < MercariConfig.WORD_I].index:
        word2i_s.loc[index] = word2i.loc[index]
    
    word2i_s.sort_values('word_id', inplace=True)
    
    return word2i_s


def save_word2i(word2i, file_name):
    word2i.to_csv(path_or_buf=os.path.join(MercariConfig.OUTPUT_DIR, file_name))   


def execute_full_word2i(name, item_description, train_data, nlp):
    then = time()
    columns = []
    word2i_name = None
    word2i_id = None
    words_raw_name = None
    words_raw_id = None
    context = "["
    
    if name:
        columns.append(['name', MercariConfig.MAX_WORDS_IN_NAME])
        context += 'name'

    if item_description:
        columns.append(['item_description', MercariConfig.MAX_WORDS_IN_ITEM_DESC])
        context += 'item_description'
    
    context += ']'
    
    logger.info("Creating word2index for %s ...", context)

    for column in columns:
        column_nm = column[0]
        max_words = column[1]
                            
        logger.info("Building word2index for %s ...", column_nm)

        word2i, words_raw, max_words = build_word2i_4_items(items=train_data[column_nm], nlp=nlp, max_words=max_words)

        logger.info("Building word2index for %s done.", column_nm)

        logger.info("Saving word2index for %s ...", column_nm)

        if column_nm == 'item_description':
            word2i_file_nm = MercariConfig.WORD_2_INDEX_4_ITEM_DESC_FILE
            word2i_id = word2i
            words_raw_id = words_raw
        elif column_nm == 'name':
            word2i_file_nm = MercariConfig.WORD_2_INDEX_4_NAME_FILE
            word2i_name = word2i
            words_raw_name = words_raw

        save_word2i(word2i=word2i, file_name=word2i_file_nm)

        logger.info("Saving word2index for %s done.", column_nm)

    logger.info("Creating word2index for %s done in %s.", context, time_it(then, time()))     

    return word2i_name, word2i_id, words_raw_name, words_raw_id


def build_index_sequence(word2i, words, max_words):
    words_raw = pd.DataFrame(words)

    words_raw.columns = ['item_id', 'pos', 'word', 'entity_type']

    words_raw.drop(columns=['entity_type'], inplace=True)
    
    if max_words is not None:
        words_raw = words_raw[words_raw.pos <= max_words]

    words_raw.set_index(['item_id'], inplace=True)
    
    index_sequence_data = pd.merge(words_raw, word2i, left_on='word', right_index=True, how='left')
    
    index_sequence_data.drop(columns=['word', 'word_cnt'], inplace=True)
    
    index_sequence_data['word_id'].fillna(value=MercariConfig.OOV_I, inplace=True)
    
    index_sequence_data['word_id'] = index_sequence_data['word_id'].astype(dtype='int32')
    
    index_sequence_data.reset_index(inplace=True)
    
    index_sequence_data = index_sequence_data.pivot(index='item_id', columns='pos', values='word_id')
    
    index_sequence_data.sort_index(inplace=True)

    return index_sequence_data


def load_index_sequence(file_name, max_len):
    logger.info("Loading index_sequence_data from %s ...", file_name)

    index_sequence_data = pd.read_csv(
        filepath_or_buffer=os.path.join(MercariConfig.INPUT_DIR, file_name),
        header=0, index_col=['item_id'])

    index_sequence_data.sort_index(inplace=True)
    
    if max_len is not None:
        cols = [str(i) for i in range(max_len + 1, index_sequence_data.shape[1])]
        
        index_sequence_data.drop(columns=cols, inplace=True)
    
    logger.info("Loading index_sequence_data from %s done.", file_name)

    return index_sequence_data


def save_index_sequence_data(index_sequence_data, file_name):
    index_sequence_data.to_csv(path_or_buf=os.path.join(MercariConfig.OUTPUT_DIR, file_name))   
                              
                                
def execute_full_nl_indexation(train_data, val_data, test_data, name, item_desc, 
                               nlp, word2i_name, word2i_id, words_raw_name, words_raw_id):

    logger.info("Starting indexation ...")

    name_prop = None
    item_desc_prop = None
    train_prop = None
    val_prop = None
    test_prop = None

    if train_data is not None:
        train_prop = {'data': train_data,
                      'name_file': MercariConfig.TRAINING_NAME_INDEX_FILE,
                      'id_file': MercariConfig.TRAINING_ITEM_DESC_INDEX_FILE,
                      'words_raw_name': words_raw_name,
                      'words_raw_id': words_raw_id}

    if val_data is not None:
        val_prop = {'data': val_data,
                    'name_file': MercariConfig.VALIDATION_NAME_INDEX_FILE,
                    'id_file': MercariConfig.VALIDATION_ITEM_DESC_INDEX_FILE,
                    'words_raw_name': None,
                    'words_raw_id': None}

    if test_data is not None:
        test_prop = {'data': test_data,
                     'name_file': MercariConfig.TEST_NAME_INDEX_FILE,
                     'id_file': MercariConfig.TEST_ITEM_DESC_INDEX_FILE,
                     'words_raw_name': None,
                     'words_raw_id': None}

    if name:
        logger.info("Picking up word2index for name ...")

        word2i_name = load_word2i(file_name=MercariConfig.WORD_2_INDEX_4_NAME_FILE, word2i=word2i_name,
                                  max_words_from_index=MercariConfig.MAX_WORDS_FROM_INDEX_4_NAME)

        logger.info("Picking up word2index for name done.")

        name_prop = {'word2i': word2i_name, 'col_name': 'name',
                     'max_words': MercariConfig.MAX_WORDS_IN_NAME}

    if item_desc:
        logger.info("Picking up word2index for item_description ...")

        word2i_id = load_word2i(file_name=MercariConfig.WORD_2_INDEX_4_ITEM_DESC_FILE, word2i=word2i_id,
                                max_words_from_index=MercariConfig.MAX_WORDS_FROM_INDEX_4_ITEM_DESC)

        logger.info("Picking up word2index for item_description done.")

        item_desc_prop = {'word2i': word2i_id, 'col_name': 'item_description',
                          'max_words': MercariConfig.MAX_WORDS_IN_ITEM_DESC}
   
    for data_prop in [train_prop, val_prop, test_prop]:
        if data_prop is not None:
            for word2i_prop in [name_prop, item_desc_prop]:
                if word2i_prop is not None:
                    if word2i_prop['col_name'] == 'name':
                        file_name = data_prop['name_file']
                        words = data_prop['words_raw_name']
                    else:    
                        file_name = data_prop['id_file']
                        words = data_prop['words_raw_id']
                        
                    data = data_prop['data']
                    col_name = word2i_prop['col_name']
                    max_words = word2i_prop['max_words']
                    word2i = word2i_prop['word2i']

                    logger.info("Walking items for file %s ...", file_name)
                    
                    if words is None:
                        words, _ = walk_items_4_word2i(items=data[col_name], nlp=nlp, max_words=max_words)

                    logger.info("Walking items for file %s done.", file_name)

                    logger.info("Building index sequence for file %s ...", file_name)

                    index_sequence_data = build_index_sequence(word2i=word2i, words=words, max_words=max_words)
                    
                    logger.info("Building index sequence for file %s done.", file_name)

                    logger.info("Saving index sequence file %s ...", file_name)

                    save_index_sequence_data(index_sequence_data=index_sequence_data, file_name=file_name)

                    logger.info("Saving index sequence file %s done.", file_name)

    logger.info("Done with indexation.")   
    
    return train_data, val_data, test_data


def get_data_for_training(data_file, name_index_file, item_desc_file, max_words_in_name, max_words_in_item_desc):
    data = load_data(data_file, sep=',')
    name_seq = load_index_sequence(name_index_file, max_words_in_name)
    item_desc_seq = load_index_sequence(item_desc_file, max_words_in_item_desc)

    X_name_seq = name_seq.as_matrix()
    X_item_desc_seq = item_desc_seq.as_matrix()
    x_cat = data['category_id'].as_matrix()
    x_brand = data['brand_id'].as_matrix()
    x_f = data[['item_condition_id', 'shipping']].as_matrix()
    y = data['price'].as_matrix()
    
    return X_name_seq, X_item_desc_seq, x_cat, x_brand, x_f, y


def pad_sequences(X_name_seq, X_item_desc_seq, max_seq_len_name, max_seq_len_item_desc):
    X_name_seq = sequence.pad_sequences(
        X_name_seq, maxlen=max_seq_len_name, padding='post', truncating='post')
    X_item_desc_seq = sequence.pad_sequences(
        X_item_desc_seq, maxlen=max_seq_len_item_desc, padding='post', truncating='post')

    return X_name_seq, X_item_desc_seq


def root_mean_squared_logarithmic_error(y_true, y_pred):
    ret = losses.mean_squared_logarithmic_error(y_true, y_pred)
    return K.sqrt(ret)


def root_mean_squared_error(y_true, y_pred):
    ret = losses.mean_squared_error(y_true, y_pred)
    return K.sqrt(ret)


def load_keras_model(model_file):
    model = load_model(os.path.join(MercariConfig.INPUT_DIR, model_file), 
                       custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                       'root_mean_squared_logarithmic_error': root_mean_squared_logarithmic_error})
    
    return model
    

def save_keras_model(model, model_file):
    model.save(os.path.join(MercariConfig.OUTPUT_DIR, model_file))


def build_keras_model(word_embedding_dims, 
                      num_words_name, max_seq_len_name, 
                      num_words_item_desc, max_seq_len_item_desc,
                      cat_embedding_dims,
                      num_categories, num_brands):
    
    feature_input = kl.Input(shape=(2,), name='feature_input')
    category_input = kl.Input(shape=(1,), name='category_input')
    brand_input = kl.Input(shape=(1,), name='brand_input')
    item_desc_input = kl.Input(shape=(max_seq_len_item_desc,), name='item_desc_input')
    name_input = kl.Input(shape=(max_seq_len_name,), name='name_input')

    item_desc_embedding = kl.Embedding(num_words_item_desc, word_embedding_dims, name='item_desc_embedding')
    item_desc_embedding_dropout = kl.SpatialDropout1D(0.5, name='item_desc_embedding_dropout')
    item_desc_lstm_1 = kl.CuDNNLSTM(units=200, name='item_desc_lstm_1', return_sequences=True)
    item_desc_lstm_2 = kl.CuDNNLSTM(units=200, name='item_desc_lstm_2')
    item_desc_lstm_dropout = kl.Dropout(0.5, name='item_desc_lstm_dropout')

    name_embedding = kl.Embedding(num_words_name, word_embedding_dims, name='name_embedding')
    name_embedding_dropout = kl.SpatialDropout1D(0.5, name='name_embedding_dropout')
    name_lstm_1 = kl.CuDNNLSTM(units=100, name='name_lstm_1', return_sequences=True)
    name_lstm_2 = kl.CuDNNLSTM(units=100, name='name_lstm_2')
    name_lstm_dropout = kl.Dropout(0.5, name='name_lstm_dropout')

    category_embedding = kl.Embedding(num_categories, cat_embedding_dims, name='category_embedding')
    category_reshape = kl.Reshape(target_shape=(cat_embedding_dims,), name='category_reshape')

    brand_embedding = kl.Embedding(num_brands, cat_embedding_dims, name='brand_embedding')
    brand_reshape = kl.Reshape(target_shape=(cat_embedding_dims,), name='brand_reshape')

    input_fusion = kl.Concatenate(axis=1, name='input_fusion')
    fusion_dense_1 = kl.Dense(400, activation='relu', name='fusion_dense_1')
    fusion_dense_2 = kl.Dense(200, activation='relu', name='fusion_dense_2')
    fusion_dense_3 = kl.Dense(1, activation='relu', name='fusion_dense_3')

    item_desc_output = item_desc_embedding(item_desc_input)
    item_desc_output = item_desc_embedding_dropout(item_desc_output)
    item_desc_output = item_desc_lstm_1(item_desc_output)
    item_desc_output = item_desc_lstm_2(item_desc_output)
    item_desc_output = item_desc_lstm_dropout(item_desc_output)

    name_output = name_embedding(name_input)
    name_output = name_embedding_dropout(name_output)
    name_output = name_lstm_1(name_output)
    name_output = name_lstm_2(name_output)
    name_output = name_lstm_dropout(name_output)

    category_output = category_embedding(category_input)
    category_output = category_reshape(category_output)

    brand_output = brand_embedding(brand_input)
    brand_output = brand_reshape(brand_output)

    output = input_fusion([name_output, item_desc_output, category_output, brand_output, feature_input])
    output = fusion_dense_1(output)
    output = fusion_dense_2(output)
    prediction = fusion_dense_3(output)

    model = km.Model(inputs=[feature_input, category_input, brand_input, name_input, item_desc_input], outputs=prediction)

    return model    


def compile_keras_model(model):
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.00, clipvalue=0.5) #epsilon=None (doesn't work)
    
    model.compile(optimizer=adam, loss=root_mean_squared_logarithmic_error, metrics=[root_mean_squared_error])

    return model


def run_training(start_epoch, end_epoch, load_model_as, save_model_as):
    X_name_seq_train, X_item_desc_seq_train, x_cat_train, x_brand_train, x_f_train, y_train = get_data_for_training(
        MercariConfig.TRAINING_SET_PREP_FILE, 
        MercariConfig.TRAINING_NAME_INDEX_FILE,
        MercariConfig.TRAINING_ITEM_DESC_INDEX_FILE,
        MercariConfig.MAX_WORDS_IN_NAME,
        MercariConfig.MAX_WORDS_IN_ITEM_DESC)
    
    X_name_seq_val, X_item_desc_seq_val, x_cat_val, x_brand_val, x_f_val, y_val = get_data_for_training(
        MercariConfig.VALIDATION_SET_PREP_FILE, 
        MercariConfig.VALIDATION_NAME_INDEX_FILE,
        MercariConfig.VALIDATION_ITEM_DESC_INDEX_FILE,
        MercariConfig.MAX_WORDS_IN_NAME,
        MercariConfig.MAX_WORDS_IN_ITEM_DESC)

    num_words_item_desc = MercariConfig.MAX_WORDS_FROM_INDEX_4_ITEM_DESC + MercariConfig.WORD_I
    max_seq_len_item_desc = MercariConfig.MAX_WORDS_IN_ITEM_DESC + 1 # Remember: first word is always <START>

    num_words_name = MercariConfig.MAX_WORDS_FROM_INDEX_4_NAME + MercariConfig.WORD_I
    max_seq_len_name = MercariConfig.MAX_WORDS_IN_NAME + 1 # Remember: first word is always <START>

    X_name_seq_train, X_item_desc_seq_train = pad_sequences(X_name_seq=X_name_seq_train, 
                                                            X_item_desc_seq=X_item_desc_seq_train, 
                                                            max_seq_len_name=max_seq_len_name,
                                                            max_seq_len_item_desc=max_seq_len_item_desc)

    X_name_seq_val, X_item_desc_seq_val = pad_sequences(X_name_seq=X_name_seq_val, 
                                                            X_item_desc_seq=X_item_desc_seq_val, 
                                                            max_seq_len_name=max_seq_len_name,
                                                            max_seq_len_item_desc=max_seq_len_item_desc)    
    if load_model_as is None:
        word_embedding_dims = 32
        cat_embedding_dims = 10

        num_categories = 1098
        num_brands = 2767

        model = build_keras_model(word_embedding_dims=word_embedding_dims, 
                              num_words_name=num_words_name, max_seq_len_name=max_seq_len_name, 
                              num_words_item_desc=num_words_item_desc, max_seq_len_item_desc=max_seq_len_item_desc,
                              cat_embedding_dims=cat_embedding_dims,
                              num_categories=num_categories, num_brands=num_brands)
        
#        model = compile_keras_model(model)
    else:
        model = load_keras_model(model_file=load_model_as)

    model = compile_keras_model(model)

    tf_log_dir = MercariConfig.get_new_tf_log_dir()

    batch_size = 200
    
    callbacks = []

    tb_callback = keras.callbacks.TensorBoard(log_dir=tf_log_dir, histogram_freq=0, batch_size=batch_size, 
                                write_graph=True, write_grads=False, write_images=False, 
                                embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    
    nan_callback = keras.callbacks.TerminateOnNaN()
    
    callbacks.append(nan_callback)

#    lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, mode='min', min_lr=0)
    
#    callbacks.append(lr_callback)

    if save_model_as is not None:
        file = save_model_as + '_' + MercariConfig.MV + '_' + MercariConfig.OV + '_' + MercariConfig.WP + '_' + MercariConfig.DP
        file += '_SE{0:02d}_EE{0:02d}'.format(start_epoch, end_epoch)
        file += '_EP{epoch:02d}-{val_loss:.4f}_'
        file += datetime.utcnow().strftime("%Y%m%d-%H%M%S") + '.hdf5'

        mc_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(MercariConfig.OUTPUT_DIR, file),
                                                      monitor='val_loss',verbose=0,save_best_only=False, 
                                                      save_weights_only=False, mode='min', period=1)   
        
        callbacks.append(mc_callback)

    history_simple = model.fit(
        [x_f_train, x_cat_train, x_brand_train, X_name_seq_train, X_item_desc_seq_train], y_train,
        batch_size=batch_size,
        epochs=end_epoch,
        verbose=1,
        callbacks=[nan_callback, mc_callback],
        shuffle=True,
        initial_epoch=start_epoch,
        steps_per_epoch=None,
        validation_data=[[x_f_val, x_cat_val, x_brand_val, X_name_seq_val, X_item_desc_seq_val], y_val])
    

def main():
    then = time()
    
    initialization = False
    categorization = False
    word2i = False
    indexation = False
    training = False
    
    train_set = False
    val_set = False
    test_set = False
    
    category = False
    brand = False
    name = False
    item_desc = False
    
    word2i_name = None
    word2i_id = None
    words_raw_name = None
    words_raw_id = None
    
    for arg in sys.argv[1:]:
        if arg == 'initialization':
            initialization = True
        elif arg == 'categorization':
            categorization = True
        elif arg == 'word2i':
            word2i = True
        elif arg == 'indexation':
            indexation = True
        elif arg == 'training':
            training = True
        elif arg == 'train_set':
            train_set = True
        elif arg == 'val_set':
            val_set = True
        elif arg == 'test_set':
            test_set = True
        if arg == 'category':
            category = True
        elif arg == 'brand':
            brand = True
        elif arg == 'name':
            name = True
        elif arg == 'item_desc':
            item_desc = True
    
    #initialization = True
    #categorization = True
    #word2i = True
    #indexation = True
    #training = True

    #train_set = True
    #val_set = True
    #test_set = True

    #category = True
    #brand = True
    #name = True
    #item_desc = True

    if (not initialization and not categorization and not word2i and not indexation):
        initialization = True
        categorization = True
        word2i = True
        indexation = True
    
    if (not train_set and not test_set and not val_set):
        train_set = True
        val_set = True
        test_set = True
        
    if (not brand and not category):
        category = True
        brand = True
        
    if (not name and not item_desc and (word2i or indexation)):
        name = True
        item_desc = True

    word2i_only = word2i and not initialization and not categorization and not indexation
    
    if not training:
        logger.info("Data preparation started ...")     

        train_data, val_data, test_data = load_all_data(
            train_set=train_set or categorization or word2i,
            val_set=val_set and not initialization and not word2i_only,
            test_set=test_set and not word2i_only,
            initialization=initialization)

        if initialization:
            train_data, val_data, test_data = execute_full_data_initialization(
                train_data=train_data if train_set else None, 
                test_data=test_data if test_set else None)

        if categorization:
            train_data, val_data, test_data = execute_full_categorization(
                train_data=train_data if train_set else None, 
                val_data=val_data if val_set else None,
                test_data=test_data if test_set else None,
                category=category, 
                brand=brand)

        if initialization or categorization:
            save_all_prepared_data(
                train_data=train_data if train_set else None, 
                val_data=val_data if val_set else None, 
                test_data=test_data if test_set else None)

        if word2i or indexation:
            nlp = load_language_model(model_nm=MercariConfig.SPACY_MODEL, include_ner=MercariConfig.INCLUDE_NER)

        if word2i:
            word2i_name, word2i_id, words_raw_name, words_raw_id = execute_full_word2i(
                name, item_desc, train_data, nlp)

        if indexation:
            execute_full_nl_indexation(
                train_data=train_data if train_set else None, 
                val_data=val_data if val_set else None, 
                test_data=test_data if test_set else None, 
                name=name, item_desc=item_desc,
                nlp=nlp, word2i_name=word2i_name, word2i_id=word2i_id,
                words_raw_name=words_raw_name if train_set else None,
                words_raw_id=words_raw_id if train_set else None)

        logger.info("Data preparation done in %s.", time_it(then, time()))  
    elif training:
        start_epoch = 0
        end_epoch = 10
        load_model_as = 'merl_model_MV01R00_OV01R00_WP01R00_DP01R00_0_10_10-0.4120_20180130-132801.hdf5'
        save_model_as = 'TR002'
        
        run_training(start_epoch=start_epoch, end_epoch=end_epoch,
                    load_model_as=load_model_as, save_model_as=save_model_as)


if __name__ == "__main__":
    main()
    