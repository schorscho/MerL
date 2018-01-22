import os
import logging
import sys

import pandas as pd

import data_preparation as mdp
import word_2_index as w2i
from mercari_config import MercariConfig


logger = logging.getLogger('MerL.nl_indexation')


def walk_data_4_indexation(data, nlp, word2index, col_name, max_words, index_col_prefix):
    for i in range(max_words + 1):
        data[index_col_prefix + str(i)] = 0

    data_len = len(data)

    progress = 0

    row_iterator = data.iterrows()

    for index, _ in row_iterator:
        if not progress % 1000:
            logger.info("Progress: %3.2f", (progress * 100.0 / data_len))

        item = data.at[index, col_name]
        doc = nlp(item)
        tok_cnt = len(doc)
        tok_i = 0
        seq_i = 1
        
        data.at[index, index_col_prefix + '0'] = MercariConfig.START_I # <START>

        logger.debug('Entities: %s', doc.ents)
        logger.debug('Tokens: %s', doc)

        for ent in doc.ents:
            if seq_i <= max_words:
                seq_i = walk_tokens_4_indexation(data=data, index=index, doc=doc, word2index=word2index, 
                                                 seq_i=seq_i, 
                                                 start=tok_i, end=min(ent.start, tok_i + max_words - seq_i + 1),
                                                 index_col_prefix=index_col_prefix)

                if ent.text in word2index.index:
                    data.at[index, index_col_prefix + str(seq_i)] = word2index.at[ent.text, 'word_id']
                else:
                    data.at[index, index_col_prefix + str(seq_i)] = MercariConfig.OOV_I # <OOV>

                tok_i = ent.end
                seq_i += 1
                
                logger.debug('Entity: %s %i %i %s', ent.text, ent.start, ent.end, ent.label_)

            else:
                break

        walk_tokens_4_indexation(data=data, index=index, doc=doc, word2index=word2index, 
                                 seq_i=seq_i, start=tok_i, end=min(tok_cnt, tok_i + max_words - seq_i + 1),
                                 index_col_prefix=index_col_prefix)

        progress += 1


def walk_tokens_4_indexation(data, index, doc, word2index, seq_i, start, end, index_col_prefix):
    for i in range(start, end):
        tok = doc[i]

        if tok.text in word2index.index:
            data.at[index, index_col_prefix + str(seq_i)] = word2index.at[tok.text, 'word_id']
        else:
            data.at[index, index_col_prefix + str(seq_i)] = MercariConfig.OOV_I # <OOV>
        
        seq_i += 1

        logger.debug('Token: %s %i', tok.text, tok.i)
    
    return seq_i


def execute_full_nl_indexation(include_train, include_val, include_test, include_name, include_item_description):
    logger.info("Starting nl indexation ...")

    word2i_name = 0
    word2i_item_desc = 0
    train_data = 0
    val_data = 0

    if include_train:
        logger.info("Loading prepared training data ...")

        data = mdp.load_data(MercariConfig.TRAINING_SET_PREP_FILE)
        train_data = {'data': data, 'file': MercariConfig.TRAINING_SET_PREP_FILE}

    if include_val:
        logger.info("Loading prepared validation data ...")

        data = mdp.load_data(MercariConfig.VALIDATION_SET_PREP_FILE)
        val_data = {'data': data, 'file': MercariConfig.VALIDATION_SET_PREP_FILE}

    if include_test:
        logger.info("Loading prepared test data ...")

        data = mdp.load_data(MercariConfig.TEST_SET_PREP_FILE, test=True)
        test_data = {'data': data, 'file': MercariConfig.TEST_SET_PREP_FILE}

    if include_name:
        logger.info("Loading word2index for name ...")
        
        word2i = w2i.load_word2index(file_name=MercariConfig.WORD_2_INDEX_4_NAME_FILE,
                                     max_words_from_index=MercariConfig.MAX_WORDS_FROM_INDEX_4_NAME)
        word2i_name = {'word2i': word2i, 'col_name': 'name', 
                       'max_words': MercariConfig.MAX_WORDS_IN_NAME, 'index_col_prefix': 'nm'}

    if include_item_description:
        logger.info("Loading word2index for item_description ...")

        word2i = w2i.load_word2index(file_name=MercariConfig.WORD_2_INDEX_4_ITEM_DESC_FILE,
                                     max_words_from_index=MercariConfig.MAX_WORDS_FROM_INDEX_4_ITEM_DESC)
        word2i_item_desc = {'word2i': word2i, 'col_name': 'item_description', 
                            'max_words': MercariConfig.MAX_WORDS_IN_ITEM_DESC, 'index_col_prefix': 'id'}
   
    logger.info("Loading language model ...")
    
    nlp = w2i.load_spacy_model(model_nm='en_core_web_md', include_ner=True)

    for data in [train_data, val_data, test_data]:
        if data != 0:
            for word2i in [word2i_name, word2i_item_desc]:
                if word2i != 0:
                    logger.info("Performing indexation for file %s column %s", data['file'], word2i['col_name'])
                    
                    walk_data_4_indexation(data=data['data'], nlp=nlp,
                                           word2index=word2i['word2i'],
                                           col_name=word2i['col_name'],
                                           max_words=word2i['max_words'],
                                           index_col_prefix=word2i['index_col_prefix'])

            logger.info("Saving prepared data file %s", data['file'])

            mdp.save_data(data['data'], data['file'])

    logger.info("Done with nl indexation.")    


def main():
    b_nm = False
    b_id = False
    b_train = False
    b_val = False
    b_test = False
    
    for arg in sys.argv[1:]:
        if arg == 'name':
            b_nm = True
        elif arg == 'item_description':
            b_id = True
        elif arg == 'training':
            b_train = True
        elif arg == 'validation':
            b_val = True
        elif arg == 'test':
            b_test = True
    
    if (not b_nm and not b_id):
        b_nm = True
        b_id = True

    if (not b_train and not b_val and not b_test):
        b_train = True
        b_val = True
        b_test = True
        
    execute_full_nl_indexation(include_train=b_train, include_val=b_val, include_test=b_test,
                               include_name=b_nm, include_item_description=b_id)

    
if __name__ == "__main__":
    main()


