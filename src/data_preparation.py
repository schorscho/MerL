import sys
import os
import operator
import logging
import logging.config
from time import time

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import spacy
from spacy.symbols import ORTH


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
    PROJECT_ROOT_DIR = '~'
    INPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
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
    
    MAX_WORDS_FROM_INDEX_4_ITEM_DESC = 50000
    MAX_WORDS_FROM_INDEX_4_NAME = 40000
    MAX_WORDS_IN_ITEM_DESC = 500
    MAX_WORDS_IN_NAME = 25
    
    TRAIN_SIZE = 0.2
    VAL_SIZE = 0.02


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


def load_all_data(training, validation, test, initialization):
    train_data = None
    val_data = None
    test_data = None
    
    sep = '\t' if initialization else ','
    
    if training:
        if initialization:
            logger.info("Loading initial training data ...")
        
            train_data = load_data(file_name=MercariConfig.TRAINING_SET_FILE, sep=sep)

            logger.info("Loading initial training data done.")
        else:
            logger.info("Loading prepared training data ...")
        
            train_data = load_data(file_name=MercariConfig.TRAINING_SET_PREP_FILE, sep=sep)
        
            logger.info("Loading prepared training data done.")

    if validation:
        logger.info("Loading prepared validation data ...")

        val_data = load_data(file_name=MercariConfig.VALIDATION_SET_PREP_FILE, sep=sep)

        logger.info("Loading prepared validation data done.")

    if test:
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
    words = []
    progress = 0
    set_len = len(items)

    if max_words is None:
        max_words = 10000
        
    max_words_len = 0

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
    columns = []
    word2i_name = None
    word2i_id = None
    words_raw_name = None
    words_raw_id = None
    context = ""
    
    if name:
        columns.append(['name', MercariConfig.MAX_WORDS_IN_NAME])
        context = 'name'
    if item_description:
        if name:
            context += ' and '
            
        columns.append(['item_description', MercariConfig.MAX_WORDS_IN_ITEM_DESC])
        context += 'item_description'
        
    logger.info("Creating word2index for %s ...", context)
    then = time()

    for column in columns:
        column_nm = column[0]
        max_words = column[1]
                            
        logger.info("Creating word2index for %s ...", column_nm)

        word2i, words_raw, max_words = build_word2i_4_items(items=train_data[column_nm], nlp=nlp, max_words=max_words)

        logger.info("Creating word2index for %s done.", column_nm)

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

    logger.info("Creating word2index for %s done in %.3f seconds.", columns, time() - then)     

    return word2i_name, word2i_id, words_raw_name, words_raw_id


def build_index_sequence(word2i, words, max_words):
    words_raw = pd.DataFrame(words)

    words_raw.columns = ['item_id', 'pos', 'word', 'entity_type']

    words_raw.drop(columns=['entity_type'], inplace=True)
    
    if max_words is not None:
        words_raw = words_raw[words_raw.pos <= max_words]

    words_raw.set_index(['item_id', 'pos'], inplace=True)
    
    index_sequence_data = pd.merge(words_raw, word2i, left_on='word', right_index=True, how='left')
    
    index_sequence_data.drop(columns=['word', 'word_cnt'], inplace=True)
    
    index_sequence_data['word_id'].fillna(value=MercariConfig.OOV_I, inplace=True)
    
    index_sequence_data['word_id'] = index_sequence_data['word_id'].astype(dtype='int32')
    
    index_sequence_data.sort_index(inplace=True)

    return index_sequence_data


def load_index_sequence(file_name, max_len):
    logger.info("Loading index_sequence_data from %s ...", file_name)

    index_sequence_data = pd.read_csv(
        filepath_or_buffer=os.path.join(MercariConfig.INPUT_DIR, file_name),
        header=0, index_col=['item_id', 'pos'])

    index_sequence_data.sort_index(inplace=True)

    logger.info("Loading index_sequence_data from %s done.", file_name)

    return index_sequence_data


def save_index_sequence_data(index_sequence_data, file_name):
    index_sequence_data.to_csv(path_or_buf=os.path.join(MercariConfig.OUTPUT_DIR, file_name))   


def load_item_desc_sequence(file_name):
    seq = load_index_sequence(file_name, MercariConfig.MAX_WORDS_IN_ITEM_DESC).as_matrix()

    seq = np.reshape(a=seq, newshape=(-1, MercariConfig.MAX_WORDS_IN_ITEM_DESC + 1))
    
    return seq


def load_name_sequence(file_name):
    seq = load_index_sequence(file_name, MercariConfig.MAX_WORDS_IN_NAME).as_matrix()

    seq = np.reshape(a=seq, newshape=(-1, MercariConfig.MAX_WORDS_IN_NAME + 1))
    
    return seq

                                
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
                    if word2i_prop == name_prop:
                        file_name = data_prop['name_file']
                        words = data_prop['words_raw_name']
                    else:    
                        file_name = data_prop['id_file']
                        words = data_prop['words_raw_id']
                        
                    data = data_prop['data']
                    col_name = word2i_prop['col_name']
                    max_words = word2i_prop['max_words']
                    word2i = word2i_prop['word2i']

                    logger.info("Indexation for file %s ...", file_name)
                    
                    if words is None:
                        words, _ = walk_items_4_word2i(items=data[col_name], nlp=nlp, max_words=max_words)

                    index_sequence_data = build_index_sequence(word2i=word2i, words=words, max_words=max_words)
                    
                    logger.info("Indexation for file %s done.", file_name)

                    logger.info("Saving inndexation file %s ...", file_name)

                    save_index_sequence_data(index_sequence_data=index_sequence_data, file_name=file_name)

                    logger.info("Saving inndexation file %s done.", file_name)

    logger.info("Done with indexation.")   
    
    return train_data, val_data, test_data

        
def main():
    initialization = False
    categorization = False
    word2i = False
    indexation = False
    
    training = False
    validation = False
    test = False
    
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
        elif arg == 'validation':
            validation = True
        elif arg == 'test':
            test = True
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
    #validation = True
    #test = True

    #category = True
    #brand = True
    #name = True
    #item_desc = True

    if (not initialization and not categorization and not word2i and not indexation):
        initialization = True
        categorization = True
        word2i = True
        indexation = True
    
    if (not training and not test and not validation):
        training = True
        validation = True
        test = True
        
    if (not brand and not category):
        category = True
        brand = True
        
    if (not name and not item_desc and (word2i or indexation)):
        name = True
        item_desc = True

    word2i_only = word2i and not initialization and not categorization and not indexation
    
    train_data, val_data, test_data = load_all_data(
        training=training or categorization or word2i,
        validation=validation and not initialization and not word2i_only,
        test=test and not word2i_only,
        initialization=initialization)
        
    if initialization:
        train_data, val_data, test_data = execute_full_data_initialization(
            train_data=train_data if training else None, 
            test_data=test_data if test else None)
        
    if categorization:
        train_data, val_data, test_data = execute_full_categorization(
            train_data=train_data if training else None, 
            val_data=val_data if validation else None,
            test_data=test_data if test else None,
            category=category, 
            brand=brand)

    if initialization or categorization:
        save_all_prepared_data(
            train_data=train_data if training else None, 
            val_data=val_data if validation else None, 
            test_data=test_data if test else None)

    if word2i or indexation:
        nlp = load_language_model(model_nm='en', include_ner=True)
    
    if word2i:
        word2i_name, word2i_id, words_raw_name, words_raw_id = execute_full_word2i(
            name, item_desc, train_data, nlp)
    
    if indexation:
        execute_full_nl_indexation(
            train_data=train_data if training else None, 
            val_data=val_data if validation else None, 
            test_data=test_data if test else None, 
            name=name, item_desc=item_desc,
            nlp=nlp, word2i_name=word2i_name, word2i_id=word2i_id,
            words_raw_name=words_raw_name if training else None,
            words_raw_id=words_raw_id if training else None)
    

if __name__ == "__main__":
    main()

