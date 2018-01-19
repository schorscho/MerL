import os
import logging
import sys
import operator

import pandas as pd
import spacy
from spacy.symbols import ORTH

from mercari_config import MercariConfig
import data_preparation as mdp


logger = logging.getLogger('MerL.word_2_index')


def word_2_index(data, column_nm, nlp):
    if column_nm == 'item_description':
        word2index_file_nm = MercariConfig.WORD_2_INDEX_4_ITEM_DESC_FILE
    elif column_nm == 'name':
        word2index_file_nm = MercariConfig.WORD_2_INDEX_4_NAME_FILE

    logger.info("Creating word2index for %s ...", column_nm)

    word2index, max_words = build_word2index(data[column_nm], nlp)

    logger.info("Saving word2index for %s ...", column_nm)

    save_word2index(word2index=word2index, file_name=word2index_file_nm)
    
    logger.info("Saved word2index for %s ...", column_nm)


def build_word2index(items, nlp):
    word = init_word_dict()

    max_words = walk_items_4_word_2_index(items, nlp=nlp, word=word)

    word2index = build_word2index_from_dict(word=word)

    return word2index, max_words


def walk_items_4_word_2_index(items, nlp, word):
    progress = 0
    set_len = len(items)

    max_item_len = 0

    for item in items:
        if not progress % 1000:
            logger.info("Progress: %3.2f", (progress * 100.0 / set_len))

        doc = nlp(item)
        tok_cnt = len(doc)
        ent_cnt = len(doc.ents)    
        item_len = 0
        tok_i = 0

        logger.debug('Entities: %s', doc.ents)
        logger.debug('Tokens: %s', doc)

        for ent in doc.ents:
            item_len += walk_tokens_4_word_2_index(doc, word, tok_i, ent.start)

            tok_i = ent.end

            if not ent.text in word:
                word[ent.text] = [1, ent.label_]
            else:
                word[ent.text][0] += 1

            item_len += 1
            
            logger.debug('Entity: %s %i %i %s', ent.text, ent.start, ent.end, ent.label_)

        item_len += walk_tokens_4_word_2_index(doc, word, tok_i, tok_cnt)

        max_item_len = item_len if max_item_len < item_len else max_item_len

        progress += 1
            
    return max_item_len


def walk_tokens_4_word_2_index(doc, word, start, end):
    tok_len = 0
    
    for i in range(start, end):
        tok = doc[i]

        if not tok.text in word:
            word[tok.text] = [1, MercariConfig.NON_ENTITY_TYPE]
        else:
            word[tok.text][0] += 1

        tok_len += 1
        
        logger.debug('Token: %s %i', tok.text, tok.i)
        
    return tok_len


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


def build_word2index_from_dict(word):
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

    word2index['word_cnt'] = word2index['comp'].map(operator.itemgetter(0))
    word2index['entity_type'] = word2index['comp'].map(operator.itemgetter(1))

    word2index = word2index[['word_id', 'entity_type', 'word_cnt']].sort_values(by='word_id')    

    return word2index


def load_spacy_model(model_nm='en', include_ner=False):
    disabled = ['parser', 'tagger']
    
    if (not include_ner):
        disabled.append('ner')
        
    nlp = spacy.load(model_nm, disable=disabled)  
    nlp.tokenizer.add_special_case('[rm]', [{ORTH: '[rm]'}])
    
    return nlp


def load_word2index(file_name, max_words_from_index=None):
    word2index = pd.read_csv(filepath_or_buffer=os.path.join(MercariConfig.DATASETS_DIR, file_name), 
                        header=0, index_col=['word'])

    word2index_h = word2index.sort_values(by='word_cnt', ascending=False).head(max_words_from_index)

    for index in word2index[word2index['word_id'] < MercariConfig.WORD_I].index:
        word2index_h.loc[index] = word2index.loc[index]

    return word2index_h


def save_word2index(word2index, file_name):
    word2index.to_csv(path_or_buf=os.path.join(MercariConfig.DATASETS_DIR, file_name))   


def execute_full_w2i(include_name, include_item_description):
    columns = []
    
    if include_name:
        columns.append('name')
    if include_item_description:
        columns.append('item_description')

    logger.info("Starting word2index for %s ...", columns)
    logger.info("Loading prepared training data ...")

    train_data = mdp.load_data(MercariConfig.TRAINING_SET_PREP_FILE)

    logger.info("Loading language model ...")
    
    nlp = load_spacy_model(model_nm='en_core_web_md', include_ner=True)
    
    for column_nm in columns:
        word_2_index(train_data, column_nm, nlp)

    logger.info("Done with word2index for %s ...", columns)    


def main():
    b_nm = False
    b_id = False
    
    for arg in sys.argv[1:]:
        if arg == 'name':
            b_nm = True
        elif arg == 'item_description':
            b_id = True
    
    if (not b_nm and not b_id):
        b_nm = True
    
    execute_full_w2i(b_nm, b_id)

    
if __name__ == "__main__":
    main()
