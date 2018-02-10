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
from sklearn.preprocessing import OneHotEncoder

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


class MerLConfig:
    PROJECT_ROOT_DIR = '/home/ubuntu'
    INPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "MerL")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "MerL")
    TF_LOG_DIR = os.path.join(OUTPUT_DIR, "tf_logs")
    TRAIN_FILE = "train.tsv"
    TRAIN_PREP_FILE = "mercari_train_prep"
    VAL_PREP_FILE = "mercari_val_prep"
    COMP_FILE = "test.tsv"
    TEST_PREP_FILE = "mercari_test_prep"
    W2I_ITEM_DESC_FILE = "mercari_w2i_item_desc"
    W2I_NAME_FILE = "mercari_w2i_name"
    EM_ITEM_DESC_FILE = "mercari_em_item_desc"
    EM_NAME_FILE = "mercari_em_name"
    TRAIN_NAME_INDEX_FILE = "mercari_train_name_index"
    TRAIN_ITEM_DESC_INDEX_FILE = "mercari_train_item_desc_index"
    VAL_NAME_INDEX_FILE = "mercari_val_name_index"
    VAL_ITEM_DESC_INDEX_FILE = "mercari_val_item_desc_index"
    TEST_NAME_INDEX_FILE = "mercari_test_name_index"
    TEST_ITEM_DESC_INDEX_FILE = "mercari_test_item_desc_index"

    PAD = '___PAD___'
    START = '___START___'
    OOV = '___OOV___'
    REMOVED_PRICE = '[rm]'
    EMPTY = '___VERY_EMPTY___'

    PAD_I = 0
    START_I = 1
    OOV_I = 2
    REMOVED_PRICE_I = 3
    EMPTY_I = 4
    WORD_I = 5

    NON_ENTITY_TYPE = '___NONE_ENTITY___'
    
    TRAIN_SIZE = 0.18
    VAL_SIZE = 0.02
    TEST_SIZE = 0.2
    NUM_CATEGORIES = 1114
    NUM_BRANDS = 2872
    DP = 'DP05R00'
    
    SPACY_MODEL = 'en' #en_core_web_lg' #'en_core_web_md'
    EMBED_FILE='glove.6B.50d.txt'
    INCLUDE_NER = False

    MAX_WORDS_FROM_INDEX_ITEM_DESC = 20000
    MAX_WORDS_FROM_INDEX_NAME = 20000
    MAX_WORDS_IN_ITEM_DESC = 200
    MAX_WORDS_IN_NAME = 20
    WP = 'WP07R00'

    WORD_EMBED_DIMS = 50
    CAT_EMBED_DIMS = 10
    MV = 'MV07R00'
    
    LEARNING_RATE = 0.001
    OV = 'OV01R00'

    START_EP = 0
    END_EP = 10
    BATCH_SIZE=196
    LOAD_MODEL = None
    SAVE_MODEL = 'TR042'


    @staticmethod
    def get_new_tf_log_dir():
        now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        log_dir = "{}/run-{}/".format(MerLConfig.TF_LOG_DIR, now)

        return log_dir
    

def time_it(start, end):
    h, r = divmod(end - start, 3600)
    m, s = divmod(r, 60)
    
    return "{:0>2}:{:0>2}:{:06.3f}".format(int(h), int(m), s)


def load_data(file_name, sep):
    if sep == ',':
        file_name += '_' + MerLConfig.DP + '.csv'
    
    data = pd.read_csv(filepath_or_buffer=os.path.join(MerLConfig.INPUT_DIR, file_name), 
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


def load_all_data(train_set, val_set, test_set, initialization):
    train_data = None
    val_data = None
    test_data = None
    
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

    return train_data, val_data, test_data


def save_all_prepared_data(train_data, val_data, test_data):
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


def execute_full_data_initialization(train_data):
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

    return train_data, val_data, test_data
    

def build_category2index(data, category, max_cats):
    category2index = data[[category + '_name', 'name']]

    category2index.columns = [category, category + '_cnt']

    category2index = category2index.groupby(category).count()

    category2index[category + '_id'] = [i for i in range(MerLConfig.WORD_I, len(category2index) + MerLConfig.WORD_I)]

    category2index = category2index.sort_values(by=category + '_cnt', ascending=False).head(max_cats)

    category2index.at[MerLConfig.OOV, category + '_id'] = MerLConfig.OOV_I
    category2index.at[MerLConfig.EMPTY, category + '_id'] = MerLConfig.EMPTY_I

    category2index[category + '_cnt'].fillna(value=0, inplace=True)
    category2index.sort_values(by=category + '_id', inplace=True)
    category2index = category2index.astype('int32')

    return category2index


def categorize_column(data, category, cat2i, max_cats):
    cat2i = cat2i.sort_values(by=category + '_cnt', ascending=False).head(max_cats)
    
    if (category + '_id') in data.columns:
        data.drop(columns=category + '_id', inplace=True)
        
    data = pd.merge(data, cat2i, left_on=category + '_name', right_index=True, how='left')

    data[category + '_id'].fillna(value=MerLConfig.OOV_I, inplace=True)
    
    data[category + '_id'] = data[category + '_id'].astype(dtype='int32')    

    data.drop(columns=[category + '_cnt'], inplace=True)
    
    return data

                        
def execute_full_categorization(train_data, val_data, test_data, category, brand):
    logger.info("Starting categorization ...")

    cat_prop = None
    brand_prop = None

    train_prop = {'data': train_data, 'file': MerLConfig.TRAIN_PREP_FILE}
    val_prop = {'data': val_data, 'file': MerLConfig.VAL_PREP_FILE}
    test_prop = {'data': test_data, 'file': MerLConfig.TEST_PREP_FILE}

    if category:
        logger.info("Building category2index for category ...")
        cat2i = build_category2index(data=train_data, category='category', max_cats=MerLConfig.NUM_CATEGORIES)
        cat_prop = {'cat2i': cat2i, 'category': 'category', 'max_cats': MerLConfig.NUM_CATEGORIES + 1}

        logger.info("Building category2index for category done.")

    if brand:
        logger.info("Building category2index for brand ...")
        
        cat2i = build_category2index(data=train_data, category='brand', max_cats = MerLConfig.NUM_BRANDS)
        brand_prop = {'cat2i': cat2i, 'category': 'brand', 'max_cats': MerLConfig.NUM_BRANDS + 1}

        logger.info("Building category2index for brand done.")

    for data_prop in [train_prop, val_prop, test_prop]:
        if data_prop['data'] is not None:
            for cat2i_prop in [cat_prop, brand_prop]:
                if cat2i_prop is not None:
                    logger.info("Performing categorization for file %s and category %s ...", 
                                data_prop['file'], cat2i_prop['category'])
                    
                    data_prop['data'] = categorize_column(data=data_prop['data'], 
                                                          category=cat2i_prop['category'],
                                                          cat2i=cat2i_prop['cat2i'],
                                                          max_cats=cat2i_prop['max_cats'])

                    logger.info("Performing categorization for file %s and category %s done.", 
                                data_prop['file'], cat2i_prop['category'])

    logger.info("Done with categorization.")    

    return train_prop['data'], val_prop['data'], test_prop['data']


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

        if not tok.is_stop and not tok.is_space:
            words.append([item_id, pos_i, tok.lower_, MerLConfig.NON_ENTITY_TYPE])

            pos_i += 1
        
        logger.debug('Token: %s %s %s %s %i', tok.text, tok.lower_, tok.norm_, tok.lemma_, tok.i)
        
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
        #doc = nlp.tokenizer(items[item_id])
        tok_cnt = len(doc)
        pos_i = 0
        tok_i = 0

        logger.debug('Entities: %s', doc.ents)
        logger.debug('Tokens: %s', doc)

        words.append([item_id, pos_i, MerLConfig.START, MerLConfig.NON_ENTITY_TYPE]) # <START>

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
            [[item_id, pos_i, MerLConfig.PAD , MerLConfig.NON_ENTITY_TYPE] for pos_i in range(pos_i, max_words + 1)])        

        progress += 1
        
    logger.info("Walkthrough for word2i for %s done in %s.", items.name, time_it(then, time()))
    
    return words, max_words_len


def build_word2i_4_items(items, nlp, max_words_in_item, max_words_from_index):
    words, max_words_len = walk_items_4_word2i(items=items, nlp=nlp, max_words=max_words_in_item)

    word2i = build_word2i_from_sequence_list(words=words, max_words_from_index=max_words_from_index)

    return word2i, words, max_words_len


def build_word2i_from_sequence_list(words, max_words_from_index):
    word2i = pd.DataFrame(words)

    word2i.columns = ['item_id', 'pos', 'word', 'entity_type']

    word2i = word2i.groupby(['word']).count()

    word2i.drop(columns=['pos', 'entity_type'], inplace=True)

    word2i.columns = ['word_cnt']
    
    print(max_words_from_index)

    max_words_from_index = max_words_from_index - MerLConfig.WORD_I
    print(max_words_from_index)

    word2i_s = word2i.sort_values(by='word_cnt', ascending=False).head(max_words_from_index)

    for word in (MerLConfig.PAD, MerLConfig.START, MerLConfig.OOV, MerLConfig.REMOVED_PRICE, MerLConfig.EMPTY):
        if word in word2i_s.index:
            max_words_from_index += 1
    print(max_words_from_index)
    word2i = word2i.sort_values(by='word_cnt', ascending=False).head(max_words_from_index)

    word2i['word_id'] = MerLConfig.WORD_I

    word2i.at[MerLConfig.PAD, 'word_id'] = MerLConfig.PAD_I
    word2i.at[MerLConfig.START, 'word_id'] = MerLConfig.START_I
    word2i.at[MerLConfig.OOV, 'word_id'] = MerLConfig.OOV_I
    word2i.at[MerLConfig.REMOVED_PRICE, 'word_id'] = MerLConfig.REMOVED_PRICE_I
    word2i.at[MerLConfig.EMPTY, 'word_id'] = MerLConfig.EMPTY_I

    word2i = word2i.sort_values(by='word_id', ascending=True)
    
    print(len(word2i))

    word2i['word_id'] = [i for i in range(len(word2i))]

    word2i['word_cnt'].fillna(value=0, inplace=True)

    word2i = word2i.astype(dtype='int32')

    word2i.sort_values('word_id', inplace=True)
    
    return word2i


def load_word2i(file_name):
    logger.info("Loading word2index from %s ...", file_name)

    file_name += '_' + MerLConfig.WP + '_' + MerLConfig.DP + '.csv'

    word2i = pd.read_csv(
        filepath_or_buffer=os.path.join(MerLConfig.INPUT_DIR, file_name),
        header=0, index_col=['word'])

    logger.info("Loading word2index from %s done.", file_name)

    return word2i


def save_word2i(word2i, file_name):
    file_name += '_' + MerLConfig.WP + '_' + MerLConfig.DP + '.csv'

    word2i.to_csv(path_or_buf=os.path.join(MerLConfig.OUTPUT_DIR, file_name))   


def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')


def build_embedding_matrix(word2i, max_words_from_index):
    embeddings_index = dict(get_coefs(*o.strip().split())
                            for o in open(os.path.join(MerLConfig.INPUT_DIR, MerLConfig.EMBED_FILE)))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words_from_index, MerLConfig.WORD_EMBED_DIMS))
    
    for word in word2i.index:
        embedding_vector = embeddings_index.get(word)
        
        if embedding_vector is not None: 
            embedding_matrix[word2i.at[word, 'word_id']] = embedding_vector

    embedding_matrix = pd.DataFrame(embedding_matrix)

    return embedding_matrix


def load_embedding_matrix(file_name):
    logger.info("Loading embedding matrix from %s ...", file_name)

    file_name += '_' + MerLConfig.WP + '_' + MerLConfig.DP + '.csv'

    embedding_matrix = pd.read_csv(
        filepath_or_buffer=os.path.join(MerLConfig.INPUT_DIR, file_name),
        header=0, index_col=None)

    columns = [str(i) for i in range(MerLConfig.WORD_EMBED_DIMS)]
    
    columns.insert(0, 'word_id')
    
    embedding_matrix.columns = columns
    
    embedding_matrix.set_index(['word_id'], inplace=True)
    
    embedding_matrix.sort_index(inplace=True)
    
    logger.info("Loading embedding matrix from %s done.", file_name)

    return embedding_matrix


def save_embedding_matrix(embedding_matrix, file_name):
    file_name += '_' + MerLConfig.WP + '_' + MerLConfig.DP + '.csv'

    embedding_matrix.to_csv(path_or_buf=os.path.join(MerLConfig.OUTPUT_DIR, file_name))   


def execute_full_word2i(name, item_description, train_data, nlp):
    then = time()
    columns = []
    word2i_name = None
    word2i_id = None
    words_raw_name = None
    words_raw_id = None
    context = "["
    
    if name:
        columns.append(['name', MerLConfig.MAX_WORDS_IN_NAME, MerLConfig.MAX_WORDS_FROM_INDEX_NAME])
        context += 'name'

    if item_description:
        columns.append(['item_description', MerLConfig.MAX_WORDS_IN_ITEM_DESC, MerLConfig.MAX_WORDS_FROM_INDEX_ITEM_DESC])
        context += 'item_description'
    
    context += ']'
    
    logger.info("Creating word2index for %s ...", context)

    for column in columns:
        column_nm = column[0]
        max_words_in_item = column[1]
        max_words_from_index = column[2]
                            
        logger.info("Building word2index for %s ...", column_nm)

        word2i, words_raw, max_words = build_word2i_4_items(items=train_data[column_nm], nlp=nlp, 
                                                            max_words_in_item=max_words_in_item,
                                                            max_words_from_index=max_words_from_index)

        logger.info("Building word2index for %s done.", column_nm)

        logger.info("Saving word2index for %s ...", column_nm)

        if column_nm == 'item_description':
            word2i_file_nm = MerLConfig.W2I_ITEM_DESC_FILE
            word2i_id = word2i
            words_raw_id = words_raw
            em_file_nm = MerLConfig.EM_ITEM_DESC_FILE
        elif column_nm == 'name':
            word2i_file_nm = MerLConfig.W2I_NAME_FILE
            word2i_name = word2i
            words_raw_name = words_raw
            em_file_nm = MerLConfig.EM_NAME_FILE

        save_word2i(word2i=word2i, file_name=word2i_file_nm)
        
        logger.info("Saving word2index for %s done.", column_nm)
        
        logger.info("Building embedding matrix for %s ...", column_nm)

        embedding_matrix = build_embedding_matrix(word2i=word2i, max_words_from_index=max_words_from_index)
        
        logger.info("Building embedding matrix for %s done.", column_nm)

        logger.info("Saving embedding matrix for %s ...", column_nm)

        save_embedding_matrix(embedding_matrix, em_file_nm)
        
        logger.info("Saving embedding matrix for %s done.", column_nm)

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
    
    index_sequence_data['word_id'].fillna(value=MerLConfig.OOV_I, inplace=True)
    
    index_sequence_data['word_id'] = index_sequence_data['word_id'].astype(dtype='int32')
    
    index_sequence_data.reset_index(inplace=True)
    
    index_sequence_data = index_sequence_data.pivot(index='item_id', columns='pos', values='word_id')
    
    index_sequence_data.sort_index(inplace=True)

    return index_sequence_data


def load_index_sequence(file_name, max_len):
    logger.info("Loading index_sequence_data from %s ...", file_name)

    file_name += '_' + MerLConfig.WP + '_' + MerLConfig.DP + '.csv'

    index_sequence_data = pd.read_csv(
        filepath_or_buffer=os.path.join(MerLConfig.INPUT_DIR, file_name),
        header=0, index_col=['item_id'])

    index_sequence_data.sort_index(inplace=True)
    
    if max_len is not None:
        cols = [str(i) for i in range(max_len + 1, index_sequence_data.shape[1])]
        
        index_sequence_data.drop(columns=cols, inplace=True)
    
    logger.info("Loading index_sequence_data from %s done.", file_name)

    return index_sequence_data


def save_index_sequence_data(index_sequence_data, file_name):
    file_name += '_' + MerLConfig.WP + '_' + MerLConfig.DP + '.csv'

    index_sequence_data.to_csv(path_or_buf=os.path.join(MerLConfig.OUTPUT_DIR, file_name))   
                              
                                
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
                      'name_file': MerLConfig.TRAIN_NAME_INDEX_FILE,
                      'id_file': MerLConfig.TRAIN_ITEM_DESC_INDEX_FILE,
                      'words_raw_name': words_raw_name,
                      'words_raw_id': words_raw_id}

    if val_data is not None:
        val_prop = {'data': val_data,
                    'name_file': MerLConfig.VAL_NAME_INDEX_FILE,
                    'id_file': MerLConfig.VAL_ITEM_DESC_INDEX_FILE,
                    'words_raw_name': None,
                    'words_raw_id': None}

    if test_data is not None:
        test_prop = {'data': test_data,
                     'name_file': MerLConfig.TEST_NAME_INDEX_FILE,
                     'id_file': MerLConfig.TEST_ITEM_DESC_INDEX_FILE,
                     'words_raw_name': None,
                     'words_raw_id': None}

    if name:
        logger.info("Picking up word2index for name ...")

        word2i_name = load_word2i(file_name=MerLConfig.W2I_NAME_FILE)

        logger.info("Picking up word2index for name done.")

        name_prop = {'word2i': word2i_name, 'col_name': 'name',
                     'max_words': MerLConfig.MAX_WORDS_IN_NAME}

    if item_desc:
        logger.info("Picking up word2index for item_description ...")

        word2i_id = load_word2i(file_name=MerLConfig.W2I_ITEM_DESC_FILE)

        logger.info("Picking up word2index for item_description done.")

        item_desc_prop = {'word2i': word2i_id, 'col_name': 'item_description',
                          'max_words': MerLConfig.MAX_WORDS_IN_ITEM_DESC}
   
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

    x_name_seq = name_seq.as_matrix()
    x_item_desc_seq = item_desc_seq.as_matrix()
    x_cat = data['category_id'].as_matrix()
    x_brand = data['brand_id'].as_matrix()
    
    x_cond = data['item_condition_id'].as_matrix()    

    x_ship = data['shipping'].as_matrix()
    y = data['price'].as_matrix()
    
    return x_name_seq, x_item_desc_seq, x_cat, x_brand, x_cond, x_ship, y


def pad_sequences(x_name_seq, x_item_desc_seq, max_seq_len_name, max_seq_len_item_desc):
    x_name_seq = sequence.pad_sequences(
        x_name_seq, maxlen=max_seq_len_name, padding='post', truncating='post')
    x_item_desc_seq = sequence.pad_sequences(
        x_item_desc_seq, maxlen=max_seq_len_item_desc, padding='post', truncating='post')

    return x_name_seq, x_item_desc_seq


def root_mean_squared_logarithmic_error(y_true, y_pred):
    ret = losses.mean_squared_logarithmic_error(y_true, y_pred)
    return K.sqrt(ret)


def root_mean_squared_error(y_true, y_pred):
    ret = losses.mean_squared_error(y_true, y_pred)
    return K.sqrt(ret)


def load_keras_model(model_file):
    model = load_model(os.path.join(MerLConfig.INPUT_DIR, model_file), 
                       custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                       'root_mean_squared_logarithmic_error': root_mean_squared_logarithmic_error})
    
    return model
    

def save_keras_model(model, model_file):
    model.save(os.path.join(MerLConfig.OUTPUT_DIR, model_file))


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
    item_desc_lstm_2 = kl.CuDNNLSTM(units=200, name='item_desc_lstm_2')
    item_desc_lstm_dropout = kl.Dropout(0.5, name='item_desc_lstm_dropout')

    name_embedding = kl.Embedding(num_words_name, word_embedding_dims, 
                                  weights=[emb_matrix_name], trainable=True, name='name_embedding')
    name_embedding_dropout = kl.SpatialDropout1D(0.5, name='name_embedding_dropout')
    #name_lstm_1 = kl.CuDNNLSTM(units=100, name='name_lstm_1', return_sequences=True)
    name_lstm_2 = kl.CuDNNLSTM(units=100, name='name_lstm_2')
    name_lstm_dropout = kl.Dropout(0.5, name='name_lstm_dropout')

    category_embedding = kl.Embedding(num_categories, cat_embedding_dims, name='category_embedding')
    category_embedding_dropout = kl.Dropout(0.5, name='category_embedding_dropout')
    category_reshape = kl.Reshape(target_shape=(cat_embedding_dims,), name='category_reshape')

    brand_embedding = kl.Embedding(num_brands, cat_embedding_dims, name='brand_embedding')
    brand_embedding_dropout = kl.Dropout(0.5, name='brand_embedding_dropout')
    brand_reshape = kl.Reshape(target_shape=(cat_embedding_dims,), name='brand_reshape')

    input_fusion = kl.Concatenate(axis=1, name='input_fusion')
    fusion_dense_1 = kl.Dense(322, activation='relu', name='fusion_dense_1')
#    fusion_dropout_1 = kl.Dropout(0.1, name='fusion_dropout_1')
    fusion_dense_2 = kl.Dense(200, activation='relu', name='fusion_dense_2')
    fusion_dense_3 = kl.Dense(1, activation='relu', name='fusion_dense_3')

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
#    output = fusion_dropout_1(output)
    output = fusion_dense_2(output)
    prediction = fusion_dense_3(output)

    model = km.Model(inputs=[cond_input, ship_input, category_input, brand_input, name_input, item_desc_input],
                     outputs=prediction)

    return model    


def compile_keras_model(model):
    adam = keras.optimizers.Adam(lr=MerLConfig.LEARNING_RATE, beta_1=0.9, beta_2=0.999, 
                                 decay=0.00, clipvalue=0.5) #epsilon=None (doesn't work)
    
    model.compile(optimizer=adam, loss=root_mean_squared_logarithmic_error, metrics=[root_mean_squared_error])

    return model


def run_training(start_epoch, end_epoch, load_model_as, save_model_as):
    x_name_seq_t, x_item_desc_seq_t, x_cat_t, x_brand_t, x_cond_t, x_ship_t, y_t = get_data_for_training(
        MerLConfig.TRAIN_PREP_FILE, 
        MerLConfig.TRAIN_NAME_INDEX_FILE,
        MerLConfig.TRAIN_ITEM_DESC_INDEX_FILE,
        MerLConfig.MAX_WORDS_IN_NAME,
        MerLConfig.MAX_WORDS_IN_ITEM_DESC)
    
    x_name_seq_v, x_item_desc_seq_v, x_cat_v, x_brand_v, x_cond_v, x_ship_v, y_v = get_data_for_training(
        MerLConfig.VAL_PREP_FILE, 
        MerLConfig.VAL_NAME_INDEX_FILE,
        MerLConfig.VAL_ITEM_DESC_INDEX_FILE,
        MerLConfig.MAX_WORDS_IN_NAME,
        MerLConfig.MAX_WORDS_IN_ITEM_DESC)

    emb_matrix_item_desc = load_embedding_matrix(MerLConfig.EM_ITEM_DESC_FILE)
    num_words_item_desc = MerLConfig.MAX_WORDS_FROM_INDEX_ITEM_DESC
    max_seq_len_item_desc = MerLConfig.MAX_WORDS_IN_ITEM_DESC + 1 # Remember: first word is always <START>

    emb_matrix_name = load_embedding_matrix(MerLConfig.EM_NAME_FILE)
    num_words_name = MerLConfig.MAX_WORDS_FROM_INDEX_NAME
    max_seq_len_name = MerLConfig.MAX_WORDS_IN_NAME + 1 # Remember: first word is always <START>

    x_name_seq_t, x_item_desc_seq_t = pad_sequences(x_name_seq=x_name_seq_t, 
                                                            x_item_desc_seq=x_item_desc_seq_t, 
                                                            max_seq_len_name=max_seq_len_name,
                                                            max_seq_len_item_desc=max_seq_len_item_desc)

    x_name_seq_v, x_item_desc_seq_v = pad_sequences(x_name_seq=x_name_seq_v, 
                                                            x_item_desc_seq=x_item_desc_seq_v, 
                                                            max_seq_len_name=max_seq_len_name,
                                                            max_seq_len_item_desc=max_seq_len_item_desc)    

    if load_model_as is None:
        model = build_keras_model(word_embedding_dims=MerLConfig.WORD_EMBED_DIMS,
                                  num_words_name=num_words_name, 
                                  emb_matrix_name=emb_matrix_name, 
                                  max_seq_len_name=max_seq_len_name, 
                                  num_words_item_desc=num_words_item_desc, 
                                  emb_matrix_item_desc=emb_matrix_item_desc, 
                                  max_seq_len_item_desc=max_seq_len_item_desc,
                                  cat_embedding_dims=MerLConfig.CAT_EMBED_DIMS,
                                  num_categories=MerLConfig.NUM_CATEGORIES + 1, 
                                  num_brands=MerLConfig.NUM_BRANDS + 1)
    else:
        model = load_keras_model(model_file=load_model_as)

    model = compile_keras_model(model)

    tf_log_dir = MerLConfig.get_new_tf_log_dir()

    callbacks = []

    tb_callback = keras.callbacks.TensorBoard(log_dir=tf_log_dir, histogram_freq=0, batch_size=MerLConfig.BATCH_SIZE, 
                                write_graph=True, write_grads=False, write_images=False, 
                                embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    
    nan_callback = keras.callbacks.TerminateOnNaN()
    
    callbacks.append(nan_callback)

    lrs_callback = keras.callbacks.LearningRateScheduler(lambda ep: MerLConfig.LEARNING_RATE * (1 - ep/10))

    callbacks.append(lrs_callback)

    if save_model_as is not None:
        file = save_model_as + '_{0}_{1}_{2}_{3}'.format(MerLConfig.MV, MerLConfig.OV, MerLConfig.WP, MerLConfig.DP)
        file += '_SE{0:02d}_EE{1:02d}'.format(start_epoch, end_epoch)
        file += '_EP{epoch:02d}-{val_loss:.4f}_'
        file += datetime.utcnow().strftime("%Y%m%d-%H%M%S") + '.hdf5'

        mc_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(MerLConfig.OUTPUT_DIR, file),
                                                      monitor='val_loss',verbose=0,save_best_only=False, 
                                                      save_weights_only=False, mode='min', period=1)   
        
        callbacks.append(mc_callback)

    history_simple = model.fit(
        [x_cond_t, x_ship_t, x_cat_t, x_brand_t, x_name_seq_t, x_item_desc_seq_t], y_t,
        batch_size=MerLConfig.BATCH_SIZE,
        epochs=end_epoch,
        verbose=1,
        callbacks=[nan_callback, mc_callback],
        shuffle=True,
        initial_epoch=start_epoch,
        steps_per_epoch=None,
        validation_data=[[x_cond_v, x_ship_v, x_cat_v, x_brand_v, x_name_seq_v, x_item_desc_seq_v], y_v])


def run_test(load_model_as):
    x_name_seq_t, x_item_desc_seq_t, x_cat_t, x_brand_t, x_cond_t, x_ship_t, y_t = get_data_for_training(
        MerLConfig.TEST_PREP_FILE, 
        MerLConfig.TEST_NAME_INDEX_FILE,
        MerLConfig.TEST_ITEM_DESC_INDEX_FILE,
        MerLConfig.MAX_WORDS_IN_NAME,
        MerLConfig.MAX_WORDS_IN_ITEM_DESC)
    
    max_seq_len_item_desc = MerLConfig.MAX_WORDS_IN_ITEM_DESC + 1 # Remember: first word is always <START>

    max_seq_len_name = MerLConfig.MAX_WORDS_IN_NAME + 1 # Remember: first word is always <START>

    x_name_seq_t, x_item_desc_seq_t = pad_sequences(x_name_seq=x_name_seq_t, 
                                                            x_item_desc_seq=x_item_desc_seq_t, 
                                                            max_seq_len_name=max_seq_len_name,
                                                            max_seq_len_item_desc=max_seq_len_item_desc)

    model = load_keras_model(model_file=load_model_as)

    result = model.evaluate(x=[x_cond_t, x_ship_t, x_cat_t, x_brand_t, x_name_seq_t, x_item_desc_seq_t],
                   y=y_t, batch_size=None, verbose=1, sample_weight=None, steps=None)
    
    print(result)


def main():
    then = time()
    
    initialization = False
    categorization = False
    word2i = False
    indexation = False
    training = False
    test = False
    
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
        elif arg == 'test':
            test = True
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
    
    if not training and not test:
        logger.info("Data preparation started ...")     

        train_data, val_data, test_data = load_all_data(
            train_set=train_set or categorization or word2i,
            val_set=val_set and not initialization and not word2i_only,
            test_set=test_set and not initialization and not word2i_only,
            initialization=initialization)

        if initialization:
            train_data, val_data, test_data = execute_full_data_initialization(
                train_data=train_data)

        if categorization:
            train_data, val_data, test_data = execute_full_categorization(
                train_data=train_data, 
                val_data=val_data,
                test_data=test_data,
                category=category, 
                brand=brand)

        if initialization or categorization:
            save_all_prepared_data(
                train_data=train_data if train_set else None, 
                val_data=val_data if val_set else None, 
                test_data=test_data if test_set else None)

        if word2i or indexation:
            nlp = load_language_model(model_nm=MerLConfig.SPACY_MODEL, include_ner=MerLConfig.INCLUDE_NER)

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
        run_training(start_epoch=MerLConfig.START_EP, end_epoch=MerLConfig.END_EP,
                    load_model_as=MerLConfig.LOAD_MODEL, save_model_as=MerLConfig.SAVE_MODEL)
    elif test:
        run_test(load_model_as=MerLConfig.LOAD_MODEL)


if __name__ == "__main__":
    main()
    