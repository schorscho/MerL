import os
import pandas as pd
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from mercari_config import MercariConfig


def load_data(file_name, head=None):
    data = pd.read_csv(filepath_or_buffer=os.path.join(MercariConfig.DATASETS_DIR, file_name), 
                            header=0, index_col=['train_id'])
    
    if head != None:
        data = data.head(head)
        
    return data


def save_data(data, file_name):
    data.to_csv(path_or_buf=os.path.join(MercariConfig.DATASETS_DIR, file_name))


def load_word2index(max_words_from_index):
    word2index = pd.read_csv(filepath_or_buffer=os.path.join(MercariConfig.DATASETS_DIR, MercariConfig.WORD_2_INDEX_FILE), 
                        header=0, index_col=['word'])

    word2index_h = word2index.sort_values(by='count', ascending=False).head(max_words_from_index)
    word2index = word2index_h.append(word2index.loc[word2index['word_id'] < 4])
    
    return word2index


def index_item_desc(data, max_words_item_desc, word2index):
    progress = 0

    nlp = spacy.load('en')
    tokenizer = Tokenizer(nlp.vocab)

    data_len = len(data)

    word_seq = np.zeros(shape=(data_len, max_words_item_desc + 1), dtype=int)
    row_i = 0

    for desc in data.item_description:
        desc_doc = tokenizer(desc)
        seq_i = 1

        word_seq[row_i][0] = MercariConfig.START_I # <START>

        for token in desc_doc:
            if token.text in word2index.index:
                word_seq[row_i][seq_i] = word2index.at[token.text, 'word_id']
            else:
                word_seq[row_i][seq_i] = MercariConfig.OOV_I # <OOV>

            seq_i += 1

        row_i += 1

        progress += 1

        if progress % 100:
            print("Progress: %3.2f" % (progress * 100.0 / data_len))

    data = pd.concat([data.index.to_series(), pd.DataFrame(word_seq)], axis=1)

    data.set_index(['train_id'], inplace=True)

    return data
