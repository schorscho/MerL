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

import keras
import keras.models as km
import keras.layers as kl
import keras.constraints as kc
from keras import losses
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import backend as K

import tensorflow as tf


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


class MerLConfig:
    KAGGLE_MODE = False
    INPUT_DIR = '../input' if KAGGLE_MODE else '/home/ubuntu/data/MerL/input'
    OUTPUT_DIR = '.' if KAGGLE_MODE else '/home/ubuntu/data/MerL'
    TRAIN_FILE = "mercari-price-suggestion-challenge/train.tsv"
    TRAIN_PREP_FILE = "mercari_train_prep"
    TRAIN_NAME_INDEX_FILE = "mercari_train_name_index"
    TRAIN_ITEM_DESC_INDEX_FILE = "mercari_train_item_desc_index"
    VAL_PREP_FILE = "mercari_val_prep"
    VAL_NAME_INDEX_FILE = "mercari_val_name_index"
    VAL_ITEM_DESC_INDEX_FILE = "mercari_val_item_desc_index"
    TEST_PREP_FILE = "mercari_test_prep"
    TEST_NAME_INDEX_FILE = "mercari_test_name_index"
    TEST_ITEM_DESC_INDEX_FILE = "mercari_test_item_desc_index"
    SUB_FILE = "mercari-price-suggestion-challenge/test.tsv"
    SUB_PREP_FILE = "mercari_sub_prep"
    SUB_NAME_INDEX_FILE = "mercari_sub_name_index"
    SUB_ITEM_DESC_INDEX_FILE = "mercari_sub_item_desc_index"

    EMB_ITEM_DESC_FILE = "mercari_emb_item_desc"
    EMB_NAME_FILE = "mercari_emb_name"
    
    CAT_CAT2I_FILE = 'mercari_cat_cat2i'
    BRAND_CAT2I_FILE = 'mercari_brand_cat2i'

    EMPTY = '___VERY_EMPTY___'

    TRAIN_SIZE = 0.4
    VAL_SIZE = 0.02
    TEST_SIZE = 0.02
    DP = 'DP05R00'
    
    EMBED_FILE ='glove-global-vectors-for-word-representation/glove.6B.100d.txt'

    MAX_WORDS_FROM_INDEX_ITEM_DESC = 40000
    MAX_WORDS_FROM_INDEX_NAME = 40000
    MAX_WORDS_IN_ITEM_DESC = 100
    MAX_WORDS_IN_NAME = 20
    WP = 'WP09R00'

    WORD_EMBED_DIMS = 100
    CAT_EMBED_DIMS = 20
    MV = 'MV10R00'
    
    LEARNING_RATE = 0.001
    OV = 'OV01R00'

    START_EP = 0
    END_EP = 40
    BATCH_SIZE = 192
    LOAD_MODEL = None
    SAVE_MODEL = 'TR063'
    
    SUBMISSION_MODE = False
    GPU = True
    
    if KAGGLE_MODE:
        SUBMISSION_MODE = True
        GPU = False


def time_it(start, end):
    h, r = divmod(end - start, 3600)
    m, s = divmod(r, 60)
    
    return "{:0>2}:{:0>2}:{:06.3f}".format(int(h), int(m), s)


def load_data(file_name, sep):
    logger.info("Loading data from %s ...", file_name)

    if sep == ',':
        file_name += '_' + MerLConfig.DP + '.csv'
        file_name = os.path.join(MerLConfig.OUTPUT_DIR, file_name)
    else:
        file_name = os.path.join(MerLConfig.INPUT_DIR, file_name)
    
    data = pd.read_csv(filepath_or_buffer=file_name, 
                       sep=sep, header=0, index_col=None)
    
    id_column = None
    
    if 'train_id' in data.columns:
        id_column = 'train_id'
    elif 'test_id' in data.columns:
        id_column = 'test_id'
    
    if id_column is not None:
        data['item_id'] = data[id_column]
    
        data.drop(id_column, axis=1, inplace=True)
                          
    data.set_index(['item_id'], inplace=True)
    
    data.sort_index(inplace=True)
    
    logger.info("Loading data from %s done.", file_name)

    return data


def save_data(data, file_name):
    file_name += '_' + MerLConfig.DP + '.csv'
    data.to_csv(path_or_buf=os.path.join(MerLConfig.OUTPUT_DIR, file_name))


def split_data(X, y, train_size, val_size, test_size):
    X_tr, y_tr, X_v, y_v = split_train_test(X, y, train_size, val_size+test_size)
    
    X_v, y_v, X_te, y_te = split_train_test(X_v, y_v, val_size/(val_size+test_size), test_size/(val_size+test_size))

    return X_tr, y_tr, X_v, y_v, X_te, y_te


def split_train_test(X, y, train_size, test_size):
    split = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=None)
 
    for train_i, test_i in split.split(X, y):
        if isinstance(X, pd.DataFrame):
            X_tr = X.iloc[train_i]
            y_tr = y[train_i]
            X_te = X.iloc[test_i]
            y_te = y[test_i]
        else:
            X_tr = X[train_i]
            y_tr = y[train_i]
            X_te = X[test_i]
            y_te = y[test_i]
    
    return X_tr, y_tr, X_te, y_te


def load_all_data(train_set, val_set, test_set, sub_set, initialization):
    train_data = None
    val_data = None
    test_data = None
    sub_data = None
    
    sep = '\t' if initialization else ','
    
    if train_set:
        if initialization:
            logger.info("Loading initial training data ...")
        
            train_data = load_data(file_name=MerLConfig.TRAIN_FILE, sep=sep)

            logger.info("Loading initial training data done.")
        else:
            logger.info("Loading prepared training data ...")
        
            train_data = load_data(file_name=MerLConfig.TRAIN_PREP_FILE, sep=sep)
        
            logger.info("Loading prepared training data done.")

    if val_set:
        logger.info("Loading prepared validation data ...")

        val_data = load_data(file_name=MerLConfig.VAL_PREP_FILE, sep=sep)

        logger.info("Loading prepared validation data done.")

    if test_set:
        if initialization:
            logger.info("Loading initial test data ...")
            
            test_data = load_data(MerLConfig.TEST_FILE, sep=sep)

            logger.info("Loading initial test data done.")
        else:
            logger.info("Loading prepared test data ...")
        
            test_data = load_data(MerLConfig.TEST_PREP_FILE, sep=sep)

            logger.info("Loading prepared test data done.")

    if sub_set:
        if initialization:
            logger.info("Loading initial submission data ...")
        
            sub_data = load_data(file_name=MerLConfig.SUB_FILE, sep=sep)

            logger.info("Loading initial submission data done.")
        else:
            logger.info("Loading prepared submission data ...")
        
            sub_data = load_data(file_name=MerLConfig.SUB_PREP_FILE, sep=sep)
        
            logger.info("Loading prepared submission data done.")

    return train_data, val_data, test_data, sub_data


def save_all_prepared_data(train_data, val_data, test_data, sub_data):
    if train_data is not None:
        logger.info("Saving prepared training data ...")

        save_data(train_data, MerLConfig.TRAIN_PREP_FILE)

        logger.info("Saving prepared training data done.")

    if val_data is not None:
        logger.info("Saving prepared validation data ...")
    
        save_data(val_data, MerLConfig.VAL_PREP_FILE)

        logger.info("Saving prepared validation data done.")

    if test_data is not None:
        logger.info("Saving prepared test data ...")
    
        save_data(test_data, MerLConfig.TEST_PREP_FILE)

        logger.info("Saving prepared test data done.")
    
    if sub_data is not None:
        logger.info("Saving prepared submission data ...")
    
        save_data(sub_data, MerLConfig.SUB_PREP_FILE)

        logger.info("Saving prepared submission data done.")


def prepare_data(df):
    logger.info("Filling missing values ...")

    df['category_name'].fillna(value=MerLConfig.EMPTY, inplace=True)

    assert(len(df[df.category_name.isnull()]) == 0)

    df['brand_name'].fillna(value=MerLConfig.EMPTY, inplace=True)

    assert(len(df[df.brand_name.isnull()]) == 0)

    df['item_description'].fillna(value=MerLConfig.EMPTY, inplace=True)

    assert(len(df[df.item_description.isnull()]) == 0)

    df['name'].fillna(value=MerLConfig.EMPTY, inplace=True)

    assert(len(df[df.name.isnull()]) == 0)

    logger.info("Filling missing values done.")


def execute_data_initialization(train_data, sub_data):
    logger.info("Starting initial data preparation ...")
    
    logger.info("Preparing training data ...")

    prepare_data(train_data)

    logger.info("Preparing training data done.")

    logger.info("Splitting prepared training data ...")

    bins = np.linspace(train_data.price.min(), 1000, 10)

    train_data, _, val_data, _, test_data, _ = split_data(X=train_data, y=np.digitize(train_data.price, bins),
                                         train_size=MerLConfig.TRAIN_SIZE, val_size=MerLConfig.VAL_SIZE, 
                                         test_size=MerLConfig.TEST_SIZE)

    logger.info("Splitting prepared training data done.")

    if sub_data is not None:
        logger.info("Preparing submission data ...")

        prepare_data(sub_data)

        logger.info("Preparing submission data done.")

    return train_data, val_data, test_data, sub_data
    

def save_cat2i(cat2i, file_name):
    file_name += '_' + MerLConfig.DP + '.csv'
    cat2i.to_csv(path_or_buf=os.path.join(MerLConfig.OUTPUT_DIR, file_name))


def load_cat2i(file_name):
    file_name += '_' + MerLConfig.DP + '.csv'
    cat2i = pd.read_csv(filepath_or_buffer=os.path.join(MerLConfig.OUTPUT_DIR, file_name), header=0, index_col=None)

    return cat2i


def build_category2index(data, category):
    category2index = data[[category + '_name', 'name']]

    category2index.columns = [category, category + '_cnt']

    category2index = category2index.groupby(category).count()

    category2index[category + '_id'] = [i for i in range(1, len(category2index) + 1)]

    category2index.at['___OOV___', category + '_id'] = 0
    category2index.at['___OOV___', category + '_cnt'] = 0

    category2index.sort_values(by=category + '_id', inplace=True)
    category2index = category2index.astype('int32')

    return category2index


def categorize_column(data, category, cat2i):
    if (category + '_id') in data.columns:
        data.drop(category + '_id', axis=1, inplace=True)

    data = pd.merge(data, cat2i, left_on=category + '_name', right_index=True, how='left')

    data[category + '_id'].fillna(value=0, inplace=True)
    
    data[category + '_id'] = data[category + '_id'].astype(dtype='int32')    

    data.drop(category + '_cnt', axis=1, inplace=True)
    
    return data

                        
def execute_categorization(train_data, val_data, test_data, sub_data, category, brand):
    logger.info("Starting categorization ...")

    cat_prop = None
    brand_prop = None

    train_prop = {'data': train_data, 'file': MerLConfig.TRAIN_PREP_FILE}
    val_prop = {'data': val_data, 'file': MerLConfig.VAL_PREP_FILE}
    test_prop = {'data': test_data, 'file': MerLConfig.TEST_PREP_FILE}
    sub_prop = {'data': sub_data, 'file': MerLConfig.SUB_PREP_FILE}

    if category:
        logger.info("Building category2index for category ...")
        
        cat2i = build_category2index(data=train_data, category='category')
        
        save_cat2i(cat2i, MerLConfig.CAT_CAT2I_FILE)
        
        cat_prop = {'cat2i': cat2i, 'category': 'category'}

        logger.info("Building category2index for category done.")

    if brand:
        logger.info("Building category2index for brand ...")
        
        cat2i = build_category2index(data=train_data, category='brand')
        
        save_cat2i(cat2i, MerLConfig.BRAND_CAT2I_FILE)
        
        brand_prop = {'cat2i': cat2i, 'category': 'brand'}

        logger.info("Building category2index for brand done.")

    for data_prop in [train_prop, val_prop, test_prop, sub_prop]:
        if data_prop['data'] is not None:
            for cat2i_prop in [cat_prop, brand_prop]:
                if cat2i_prop is not None:
                    logger.info("Performing categorization for file %s and category %s ...", 
                                data_prop['file'], cat2i_prop['category'])
                    
                    data_prop['data'] = categorize_column(data=data_prop['data'], 
                                                          category=cat2i_prop['category'],
                                                          cat2i=cat2i_prop['cat2i'])

                    logger.info("Performing categorization for file %s and category %s done.", 
                                data_prop['file'], cat2i_prop['category'])

    logger.info("Done with categorization.")    

    return train_prop['data'], val_prop['data'], test_prop['data'], sub_prop['data']


def build_tokenizer(items, max_words_from_index):
    tokenizer = Tokenizer(num_words=max_words_from_index)    
    
    tokenizer.fit_on_texts(items.values)    
    
    return tokenizer


def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')


def build_emb_matrix(tokenizer, max_words_from_index):
    embeddings_index = dict(get_coefs(*o.strip().split())
                            for o in open(os.path.join(MerLConfig.INPUT_DIR, MerLConfig.EMBED_FILE)))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean = all_embs.mean()
    emb_std = all_embs.std()

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words_from_index, MerLConfig.WORD_EMBED_DIMS))
    
    word_index = tokenizer.word_index
    
    nb_words = min(max_words_from_index, len(word_index))
    
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, MerLConfig.WORD_EMBED_DIMS))
    
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        
        embedding_vector = embeddings_index.get(word)
        
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector

    embedding_matrix = pd.DataFrame(embedding_matrix)

    return embedding_matrix


def load_emb_matrix(file_name):
    logger.info("Loading embedding matrix from %s ...", file_name)

    file_name += '_' + MerLConfig.WP + '_' + MerLConfig.DP + '.csv'

    embedding_matrix = pd.read_csv(
        filepath_or_buffer=os.path.join(MerLConfig.OUTPUT_DIR, file_name),
        header=0, index_col=None)

    columns = [str(i) for i in range(MerLConfig.WORD_EMBED_DIMS)]
    
    columns.insert(0, 'word_id')
    
    embedding_matrix.columns = columns
    
    embedding_matrix.set_index(['word_id'], inplace=True)
    
    embedding_matrix.sort_index(inplace=True)
    
    logger.info("Loading embedding matrix from %s done.", file_name)

    return embedding_matrix


def save_emb_matrix(embedding_matrix, file_name):
    file_name += '_' + MerLConfig.WP + '_' + MerLConfig.DP + '.csv'

    embedding_matrix.to_csv(path_or_buf=os.path.join(MerLConfig.OUTPUT_DIR, file_name))   


def execute_word2i(name, item_desc, embedding, train_data):
    tokenizer_name = None
    tokenizer_item_desc = None

    if name:
        then = time()

        logger.info("Executing w2i for name ...")

        logger.info("Building tokenizer for name ...")

        tokenizer_name = build_tokenizer(items=train_data['name'],
                                         max_words_from_index=MerLConfig.MAX_WORDS_FROM_INDEX_NAME)

        logger.info("Building tokenizer for name done.")

        if embedding:
            logger.info("Building embedding matrix for name ...")

            emb_matrix_name = build_emb_matrix(tokenizer=tokenizer_name, 
                                               max_words_from_index=MerLConfig.MAX_WORDS_FROM_INDEX_NAME)

            logger.info("Building embedding matrix for name done.")

            logger.info("Saving embedding matrix for name ...")

            save_emb_matrix(emb_matrix_name, MerLConfig.EMB_NAME_FILE)

            logger.info("Saving embedding matrix for name done.")
        
        logger.info("Executing w2i for name done in %s.", time_it(then, time()))     

    if item_desc:
        then = time()

        logger.info("Executing w2i for item_description ...")

        logger.info("Building tokenizer for item_description ...")

        tokenizer_item_desc = build_tokenizer(items=train_data['item_description'],
                                              max_words_from_index=MerLConfig.MAX_WORDS_FROM_INDEX_ITEM_DESC)

        logger.info("Building tokenizer for item_description done.")

        if embedding:
            logger.info("Building embedding matrix for item_description ...")

            emb_matrix_item_desc = build_emb_matrix(tokenizer=tokenizer_item_desc, 
                                                    max_words_from_index=MerLConfig.MAX_WORDS_FROM_INDEX_ITEM_DESC)

            logger.info("Building embedding matrix for item_description done.")

            logger.info("Saving embedding matrix for item_description ...")

            save_emb_matrix(emb_matrix_name, MerLConfig.EMB_ITEM_DESC_FILE)

            logger.info("Saving embedding matrix for item_description done.")

        logger.info("Executing w2i for item_description done in %s.", time_it(then, time()))     

    return tokenizer_name, tokenizer_item_desc


def build_index_data(data, tokenizer, col_name, max_words):
    list_sentences = data[col_name].values                    
    list_tokenized = tokenizer.texts_to_sequences(list_sentences)
    index_data = sequence.pad_sequences(list_tokenized, maxlen=max_words, padding='post', truncating='post')
    index_data = pd.DataFrame(index_data)
    index_data['item_id'] = data.index.values
    
    index_data.set_index(['item_id'], inplace=True)
    
    index_data.sort_index(inplace=True)

    return index_data


def load_index_data(file_name):
    logger.info("Loading index data from %s ...", file_name)

    file_name += '_' + MerLConfig.WP + '_' + MerLConfig.DP + '.csv'

    index_data = pd.read_csv(
        filepath_or_buffer=os.path.join(MerLConfig.OUTPUT_DIR, file_name),
        header=0, index_col=['item_id'])

    index_data.sort_index(inplace=True)
        
    logger.info("Loading index data from %s done.", file_name)

    return index_data


def save_index_data(index_data, file_name):
    file_name += '_' + MerLConfig.WP + '_' + MerLConfig.DP + '.csv'

    index_data.to_csv(path_or_buf=os.path.join(MerLConfig.OUTPUT_DIR, file_name))   
                              
                                
def execute_indexation(train_data, val_data, test_data, sub_data, tokenizer_name, tokenizer_item_desc):
    logger.info("Starting indexation ...")

    then = time()

    name_prop = None
    item_desc_prop = None
    train_prop = None
    val_prop = None
    test_prop = None
    sub_prop = None

    if train_data is not None:
        train_prop = {'data': train_data,
                      'name_file': MerLConfig.TRAIN_NAME_INDEX_FILE,
                      'id_file': MerLConfig.TRAIN_ITEM_DESC_INDEX_FILE}

    if val_data is not None:
        val_prop = {'data': val_data,
                    'name_file': MerLConfig.VAL_NAME_INDEX_FILE,
                    'id_file': MerLConfig.VAL_ITEM_DESC_INDEX_FILE}

    if test_data is not None:
        test_prop = {'data': test_data,
                     'name_file': MerLConfig.TEST_NAME_INDEX_FILE,
                     'id_file': MerLConfig.TEST_ITEM_DESC_INDEX_FILE}

    if sub_data is not None:
        sub_prop = {'data': sub_data,
                    'name_file': MerLConfig.SUB_NAME_INDEX_FILE,
                    'id_file': MerLConfig.SUB_ITEM_DESC_INDEX_FILE}

    if tokenizer_name:
        name_prop = {'tokenizer': tokenizer_name, 'col_name': 'name',
                     'max_words': MerLConfig.MAX_WORDS_IN_NAME}

    if tokenizer_item_desc:
        item_desc_prop = {'tokenizer': tokenizer_item_desc, 'col_name': 'item_description',
                          'max_words': MerLConfig.MAX_WORDS_IN_ITEM_DESC}
   
    for data_prop in [train_prop, val_prop, test_prop, sub_prop]:
        if data_prop is not None:
            for w2i_prop in [name_prop, item_desc_prop]:
                if w2i_prop is not None:
                    if w2i_prop['col_name'] == 'name':
                        file_name = data_prop['name_file']
                    else:    
                        file_name = data_prop['id_file']
                        
                    data = data_prop['data']
                    col_name = w2i_prop['col_name']
                    max_words = w2i_prop['max_words']
                    tokenizer = w2i_prop['tokenizer']

                    logger.info("Indexation for %s ...", file_name)

                    index_data = build_index_data(data=data, tokenizer=tokenizer, col_name=col_name, max_words=max_words)

                    logger.info("Indexation for %s done.", file_name)

                    logger.info("Saving index file %s ...", file_name)

                    save_index_data(index_data=index_data, file_name=file_name)

                    logger.info("Saving index file %s done.", file_name)

    logger.info("Done with indexation in %s.", time_it(then, time()))    


def get_data_packages(data_file, name_index_file, item_desc_index_file, submission):
    data = load_data(data_file, sep=',')
    name_seq = load_index_data(name_index_file)
    item_desc_seq = load_index_data(item_desc_index_file)

    x_name_seq = name_seq.as_matrix()
    x_item_desc_seq = item_desc_seq.as_matrix()
    x_cat = data['category_id'].as_matrix()
    x_brand = data['brand_id'].as_matrix()
    x_cond = data['item_condition_id'].as_matrix()    
    x_ship = data['shipping'].as_matrix()
    
    if not submission:
        y = data['price'].as_matrix()
    else:
        y = None
    
    item_id = data.index.values
        
    return x_name_seq, x_item_desc_seq, x_cat, x_brand, x_cond, x_ship, y, item_id


def old_kpi(y_true, y_pred):
    return K.sqrt(mean_squared_logarithmic_error(y_true, y_pred))


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def mean_squared_error(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred)


def load_keras_model(model_file):
    model = load_model(os.path.join(MerLConfig.OUTPUT_DIR, model_file), 
                       custom_objects={'mean_squared_error': mean_squared_error,
                                       'old_kpi': old_kpi,
                                       'mean_squared_logarithmic_error': mean_squared_logarithmic_error})
    
    return model
    

def save_keras_model(model, model_file):
    model.save(os.path.join(MerLConfig.OUTPUT_DIR, model_file))


def create_LSTM(units, name):
    if MerLConfig.GPU:
        #return kl.CuDNNLSTM(units=units, 
         #                  kernel_initializer='glorot_uniform', 
          #                 recurrent_initializer='orthogonal', 
           #                bias_initializer='zeros', 
            #               unit_forget_bias=True, 
             #              kernel_regularizer=None, 
              #             recurrent_regularizer=None, 
               #            bias_regularizer=None, 
                #           activity_regularizer=None, 
                 #          kernel_constraint=None, 
                  #         recurrent_constraint=None, 
                   #        bias_constraint=None, 
                    #       return_sequences=False, 
                     #      return_state=False, 
                      #     stateful=False,
                       #    name=name)
    #else:
     #   return kl.LSTM(units=units, 
      #                 activation='tanh', recurrent_activation='hard_sigmoid', 
       #                #activation='relu', recurrent_activation='hard_sigmoid', 
        #               use_bias=True, 
         #              kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
          #             bias_initializer='zeros', unit_forget_bias=True, 
           #            kernel_regularizer=None, recurrent_regularizer=None, 
            #           bias_regularizer=None, activity_regularizer=None, 
             #          kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
              #         dropout=0.0, recurrent_dropout=0.0, implementation=1, 
               #        return_sequences=False, return_state=False, 
                #       go_backwards=False, stateful=False, unroll=False, name=name)
    
        return kl.CuDNNGRU(units=units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                           bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None,
                           bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                           recurrent_constraint=None, bias_constraint=None, return_sequences=False, 
                           return_state=False, stateful=False, name=name)
    
    return kl.GRU(units=units, 
                  activation='tanh', recurrent_activation='hard_sigmoid', 
                  use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                  bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                  activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                  dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False,
                  go_backwards=False, stateful=False, unroll=False, name=name)


def build_keras_model(word_embedding_dims, 
                      num_words_name, emb_matrix_name, max_seq_len_name, 
                      num_words_item_desc, emb_matrix_item_desc, max_seq_len_item_desc,
                      cat_embedding_dims,
                      num_categories, num_brands):
    
    cond_input = kl.Input(shape=(1,), name='cond_input')
    ship_input = kl.Input(shape=(1,), name='ship_input')
    category_input = kl.Input(shape=(1,), name='category_input')
    brand_input = kl.Input(shape=(1,), name='brand_input')
    item_desc_input = kl.Input(shape=(max_seq_len_item_desc,), name='item_desc_input')
    name_input = kl.Input(shape=(max_seq_len_name,), name='name_input')

    item_desc_embedding = kl.Embedding(num_words_item_desc, word_embedding_dims, 
                 weights=[emb_matrix_item_desc], trainable=True, name='item_desc_embedding')
    item_desc_embedding_dropout = kl.SpatialDropout1D(0.5, name='item_desc_embedding_dropout')
    #item_desc_lstm_1 = kl.CuDNNLSTM(units=200, name='item_desc_lstm_1', return_sequences=True)
    item_desc_lstm_2 = create_LSTM(units=300, name='item_desc_lstm_2')
    item_desc_lstm_dropout = kl.Dropout(0.5, name='item_desc_lstm_dropout')

    name_embedding = kl.Embedding(num_words_name, word_embedding_dims, 
                                  weights=[emb_matrix_name], trainable=True, name='name_embedding')
    name_embedding_dropout = kl.SpatialDropout1D(0.5, name='name_embedding_dropout')
    #name_lstm_1 = kl.CuDNNLSTM(units=100, name='name_lstm_1', return_sequences=True)
    name_lstm_2 = create_LSTM(units=150, name='name_lstm_2')
    name_lstm_dropout = kl.Dropout(0.5, name='name_lstm_dropout')


    category_embedding = kl.Embedding(num_categories + 1, cat_embedding_dims, name='category_embedding')
    category_embedding_dropout = kl.Dropout(0.5, name='category_embedding_dropout')
    category_reshape = kl.Reshape(target_shape=(cat_embedding_dims,), name='category_reshape')

    brand_embedding = kl.Embedding(num_brands + 1, cat_embedding_dims, name='brand_embedding')
    brand_embedding_dropout = kl.Dropout(0.5, name='brand_embedding_dropout')
    brand_reshape = kl.Reshape(target_shape=(cat_embedding_dims,), name='brand_reshape')

    input_fusion = kl.Concatenate(axis=1, name='input_fusion')
    fusion_dense_1 = kl.Dense(400, activation='relu', name='fusion_dense_1')
    fusion_dropout_1 = kl.Dropout(0.5, name='fusion_dropout_1')
    fusion_dense_2 = kl.Dense(300, activation='relu', name='fusion_dense_2')
    fusion_dropout_2 = kl.Dropout(0.5, name='fusion_dropout_2')
    fusion_dense_3 = kl.Dense(200, activation='relu', name='fusion_dense_3')
    fusion_dropout_3 = kl.Dropout(0.5, name='fusion_dropout_3')
    fusion_dense_4 = kl.Dense(100, activation='relu', name='fusion_dense_4')
    fusion_dropout_4 = kl.Dropout(0.5, name='fusion_dropout_4')
    fusion_dense_5 = kl.Dense(1, activation='relu', name='fusion_dense_5')

    item_desc_output = item_desc_embedding(item_desc_input)
    item_desc_output = item_desc_embedding_dropout(item_desc_output)
    #item_desc_output = item_desc_lstm_1(item_desc_output)
    item_desc_output = item_desc_lstm_2(item_desc_output)
    item_desc_output = item_desc_lstm_dropout(item_desc_output)

    name_output = name_embedding(name_input)
    name_output = name_embedding_dropout(name_output)
    #name_output = name_lstm_1(name_output)
    name_output = name_lstm_2(name_output)
    name_output = name_lstm_dropout(name_output)

    category_output = category_embedding(category_input)
    category_output = category_embedding_dropout(category_output)
    category_output = category_reshape(category_output)

    brand_output = brand_embedding(brand_input)
    brand_output = brand_embedding_dropout(brand_output)
    brand_output = brand_reshape(brand_output)

    output = input_fusion([cond_input, ship_input, name_output, item_desc_output, 
                           category_output, brand_output])
    output = fusion_dense_1(output)
    output = fusion_dropout_1(output)
    output = fusion_dense_2(output)
    output = fusion_dropout_2(output)
    output = fusion_dense_3(output)
    output = fusion_dropout_3(output)
    output = fusion_dense_4(output)
    output = fusion_dropout_4(output)
    prediction = fusion_dense_5(output)

    model = km.Model(inputs=[cond_input, ship_input, category_input, brand_input, name_input, item_desc_input],
                     outputs=prediction)

    return model    


def compile_keras_model(model):
    adam = keras.optimizers.Adam(lr=MerLConfig.LEARNING_RATE, beta_1=0.9, beta_2=0.999, 
                                 decay=0.00, clipvalue=0.5) #epsilon=None (doesn't work)
    
    model.compile(optimizer=adam, loss=mean_squared_logarithmic_error, 
                  metrics=[old_kpi, mean_squared_error])

    return model


def lr_schedule(ep):
    #lr = MerLConfig.LEARNING_RATE / (ep)
    if ep < 10:
        lr = 0.001
    elif ep < 20:
        lr = 0.00055
    elif ep < 30:
        lr = 0.0001
    else:
        lr = 0.000055
    
    logger.info('New learning rate for epoch %i: %01.10f', ep, lr)
    
    return lr


def execute_training(start_epoch, end_epoch, build_on_model, save_model_as, submission):
    x_name_seq_t, x_item_desc_seq_t, x_cat_t, x_brand_t, x_cond_t, x_ship_t, y_t, _ = get_data_packages(
        MerLConfig.TRAIN_PREP_FILE, 
        MerLConfig.TRAIN_NAME_INDEX_FILE,
        MerLConfig.TRAIN_ITEM_DESC_INDEX_FILE, 
        submission=False)
    
    x_name_seq_v, x_item_desc_seq_v, x_cat_v, x_brand_v, x_cond_v, x_ship_v, y_v, _ = get_data_packages(
        MerLConfig.VAL_PREP_FILE, 
        MerLConfig.VAL_NAME_INDEX_FILE,
        MerLConfig.VAL_ITEM_DESC_INDEX_FILE, 
        submission=False)
        
    val_data = [[x_cond_v, x_ship_v, x_cat_v, x_brand_v, x_name_seq_v, x_item_desc_seq_v], y_v]

    if build_on_model is None:
        emb_matrix_item_desc = load_emb_matrix(MerLConfig.EMB_ITEM_DESC_FILE)
        max_words_item_desc = MerLConfig.MAX_WORDS_FROM_INDEX_ITEM_DESC
        max_seq_len_item_desc = MerLConfig.MAX_WORDS_IN_ITEM_DESC

        emb_matrix_name = load_emb_matrix(MerLConfig.EMB_NAME_FILE)
        max_words_name = MerLConfig.MAX_WORDS_FROM_INDEX_NAME
        max_seq_len_name = MerLConfig.MAX_WORDS_IN_NAME
        
        num_categories = len(load_cat2i(MerLConfig.CAT_CAT2I_FILE))
        num_brands = len(load_cat2i(MerLConfig.BRAND_CAT2I_FILE))
        
        logger.info('Categories: %s, Brands: %s', num_categories, num_brands)

        model = build_keras_model(word_embedding_dims=MerLConfig.WORD_EMBED_DIMS,
                                  num_words_name=max_words_name, 
                                  emb_matrix_name=emb_matrix_name, 
                                  max_seq_len_name=max_seq_len_name, 
                                  num_words_item_desc=max_words_item_desc, 
                                  emb_matrix_item_desc=emb_matrix_item_desc, 
                                  max_seq_len_item_desc=max_seq_len_item_desc,
                                  cat_embedding_dims=MerLConfig.CAT_EMBED_DIMS,
                                  num_categories=num_categories, 
                                  num_brands=num_brands)
    else:
        model = build_on_model

    model = compile_keras_model(model)

    callbacks = []
    
    lrs_callback = keras.callbacks.LearningRateScheduler(lr_schedule)

    callbacks.append(lrs_callback)

    if save_model_as is not None and not submission:
        file = save_model_as + '_{0}_{1}_{2}_{3}'.format(MerLConfig.MV, MerLConfig.OV, MerLConfig.WP, MerLConfig.DP)
        file += '_SE{0:02d}_EE{1:02d}'.format(start_epoch, end_epoch)
        file += '_EP{epoch:02d}-{val_loss:.4f}_'
        file += datetime.utcnow().strftime("%Y%m%d-%H%M%S") + '.hdf5'

        mc_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(MerLConfig.OUTPUT_DIR, file),
                                                      monitor='val_loss',verbose=0,save_best_only=False, 
                                                      save_weights_only=False, mode='min', period=1)   
        
        callbacks.append(mc_callback)

    history_simple = model.fit(
        x=[x_cond_t, x_ship_t, x_cat_t, x_brand_t, x_name_seq_t, x_item_desc_seq_t], y=y_t,
        batch_size=MerLConfig.BATCH_SIZE,
        epochs=end_epoch,
        verbose=1 if not submission else 2,
        callbacks=callbacks,
        shuffle=True,
        initial_epoch=start_epoch,
        steps_per_epoch=None,
        validation_data=val_data if not submission else None)
    
    if submission:
        then = time()
        
        logger.info("Validation ...")    

        result = model.evaluate(x=[x_cond_v, x_ship_v, x_cat_v, x_brand_v, x_name_seq_v, x_item_desc_seq_v],
                       y=y_v, batch_size=None, verbose=1, sample_weight=None, steps=None)

        logger.info("Validation done in %s. Result: %01.4f", time_it(then, time()), result)    
    
    return model


def execute_test(model):
    x_name_seq, x_item_desc_seq, x_cat, x_brand, x_cond, x_ship, y, _ = get_data_packages(
        MerLConfig.TEST_PREP_FILE, 
        MerLConfig.TEST_NAME_INDEX_FILE,
        MerLConfig.TEST_ITEM_DESC_INDEX_FILE, 
        submission=False)
    
    result = model.evaluate(x=[x_cond, x_ship, x_cat, x_brand, x_name_seq, x_item_desc_seq],
                   y=y, batch_size=None, verbose=1, sample_weight=None, steps=None)
    
    return result


def execute_submission(model):
    x_name_seq, x_item_desc_seq, x_cat, x_brand, x_cond, x_ship, y, item_id = get_data_packages(
        MerLConfig.SUB_PREP_FILE, 
        MerLConfig.SUB_NAME_INDEX_FILE,
        MerLConfig.SUB_ITEM_DESC_INDEX_FILE, 
        submission=True)
    
    y_pred = model.predict(x=[x_cond, x_ship, x_cat, x_brand, x_name_seq, x_item_desc_seq], 
                  batch_size=None, verbose=1, steps=None)

    sub_results_data = pd.DataFrame(item_id)
    sub_results_data.columns = ['test_id']
    sub_results_data['price'] = y_pred
    
    sub_results_data.set_index(['test_id'], inplace=True)
    
    sub_results_data.sort_index(inplace=True)

    sub_results_data.to_csv(path_or_buf=os.path.join(MerLConfig.OUTPUT_DIR, 'sample_submission.csv'))   
    
    
def main():
    overall = time()

    logger.info("Main script started ...")     
    
    initialization = False
    categorization = False
    embedding = False
    indexation = False
    training = False
    test = False
    submission = False
    
    train_set = False
    val_set = False
    test_set = False
    sub_set = False
    
    category = False
    brand = False
    name = False
    item_desc = False
    
    tokenizer_name = None
    tokenizer_item_desc = None

    model = None
    
    for arg in sys.argv[1:]:
        if arg == 'initialization':
            initialization = True
        elif arg == 'categorization':
            categorization = True
        elif arg == 'embedding':
            embedding = True
        elif arg == 'indexation':
            indexation = True
        elif arg == 'training':
            training = True
        elif arg == 'test':
            test = True
        elif arg == 'submission':
            submission = True
        elif arg == 'train_set':
            train_set = True
        elif arg == 'val_set':
            val_set = True
        elif arg == 'test_set':
            test_set = True
        elif arg == 'sub_set':
            sub_set = True
        elif arg == 'category':
            category = True
        elif arg == 'brand':
            brand = True
        elif arg == 'name':
            name = True
        elif arg == 'item_desc':
            item_desc = True
    
    if MerLConfig.SUBMISSION_MODE:
        initialization = True
        categorization = True
        embedding = True
        indexation = True
        training = True
        submission = True

        train_set = True
        val_set = True
        test_set = False
        sub_set = True

        category = True
        brand = True
        name = True
        item_desc = True

    if (not initialization and not categorization and not embedding and not indexation
        and not training and not test and not submission):
        initialization = True
        categorization = True
        embedding = True
        indexation = True
    
    if (not train_set and not test_set and not val_set and not sub_set):
        train_set = True
        val_set = True
        test_set = True
        sub_set = True
        
    if (not brand and not category):
        category = True
        brand = True
        
    if (not name and not item_desc and (embedding or indexation)):
        name = True
        item_desc = True

    embedding_only = embedding and not initialization and not categorization and not indexation

    if initialization or categorization or embedding or indexation:
        then = time()

        logger.info("Data preparation started ...")     

        train_data, val_data, test_data, sub_data = load_all_data(
            train_set=train_set or categorization or embedding or indexation,
            val_set=val_set and not initialization and not embedding_only,
            test_set=test_set and not initialization and not embedding_only,
            sub_set=sub_set and not embedding_only,
            initialization=initialization)

        if initialization:
            train_data, val_data, test_data, sub_data = execute_data_initialization(
                train_data=train_data, sub_data=sub_data)

        if categorization:
            train_data, val_data, test_data, sub_data = execute_categorization(
                train_data=train_data, 
                val_data=val_data if val_set else None,
                test_data=test_data if test_set else None,
                sub_data=sub_data if sub_set else None,
                category=category, 
                brand=brand)

        if initialization or categorization:
            save_all_prepared_data(
                train_data=train_data if train_set else None, 
                val_data=val_data if val_set else None, 
                test_data=test_data if test_set else None,
                sub_data=sub_data if sub_set else None)

        if embedding or indexation:
            tokenizer_name, tokenizer_item_desc = execute_word2i(name, item_desc, embedding, train_data)

        if indexation:
            execute_indexation(
                train_data=train_data if train_set else None, 
                val_data=val_data if val_set else None, 
                test_data=test_data if test_set else None, 
                sub_data=sub_data if sub_set else None, 
                tokenizer_name=tokenizer_name if name else None, 
                tokenizer_item_desc=tokenizer_item_desc if item_desc else None)

        logger.info("Data preparation done in %s.", time_it(then, time()))  

    if (not MerLConfig.SUBMISSION_MODE) and (training or test or submission) and (MerLConfig.LOAD_MODEL is not None):
        model = load_keras_model(model_file=MerLConfig.LOAD_MODEL)

    if training:
        logger.info("Executing training ...")    

        then = time()

        model = execute_training(start_epoch=MerLConfig.START_EP, end_epoch=MerLConfig.END_EP,
                                 build_on_model=model, save_model_as=MerLConfig.SAVE_MODEL, submission=submission)
    
        logger.info("Done executing training in %s.", time_it(then, time()))    

    if test:
        logger.info("Executing test ...")    

        then = time()

        result = execute_test(model)
    
        logger.info("Done executing test in %s. Result: %01.4f", time_it(then, time()), result)    
        
    if submission:
        logger.info("Executing submission ...")    

        then = time()

        execute_submission(model)

        logger.info("Done executing submission in %s.", time_it(then, time()))    

        logger.info("Main script finished in %s.", time_it(overall, time()))    
        

if __name__ == "__main__":
    main()
    