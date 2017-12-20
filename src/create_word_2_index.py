import os
import pandas as pd
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
import timer
from mercari_config import MercariConfig


nlp = spacy.load('en')
tokenizer = Tokenizer(nlp.vocab)


df = pd.read_csv(filepath_or_buffer=os.path.join(MercariConfig.DATASETS_DIR, MercariConfig.TRAINING_SET_FILE), 
                    header=0, sep='\t', index_col=['train_id'])

set_len = len(df)

word2index = pd.DataFrame(columns=['word', 'word_id', 'count'])
word2index = word2index.astype(dtype={'word': str, 'word_id': int, 'count': int})

word2index.set_index(['word'], inplace=True)

word2index.loc[MercariConfig.PAD] = (0, 0)
word2index.loc[MercariConfig.START] = (1, 0)
word2index.loc[MercariConfig.OOV] = (2, 0)
word2index.loc[MercariConfig.EMPTY_DESC] = (3, 0)
word2index.loc[MercariConfig.REMOVED_PRICE] = (4, 0)


max_word_id = 4
max_words_in_item_desc = 0

with timer.Timer():
    progress = 0
    
    for desc in df['item_description']:
        desc_doc = tokenizer(desc)
        words_in_col = len(desc_doc)

        for token in desc_doc:
            if not token.text in word2index.index:
                word2index.loc[token.text] = (max_word_id, 1)
                max_word_id += 1
            else:
                word2index.at[token.text, 'count'] += 1

        max_words_in_item_desc = words_in_col if max_words_in_item_desc < words_in_col else max_words_in_col
        
        progress += 1
        
        if progress % 100:
            print("Progress: %3.2f" % (progress * 100.0 / set_len))

word2index.to_csv(path_or_buf=os.path.join(MercariConfig.DATASETS_DIR, MercariConfig.WORD_2_INDEX_FILE))   

df_max_words = pd.DataFrame(columns=['file', 'max_words_in_item_desc'], dtype=int)

df_max_words.set_index(['file'], inplace=True)


df_max_words.loc['train'] = max_words_in_item_desc

df_max_words.to_csv(path_or_buf=os.path.join(MercariConfig.DATASETS_DIR, MercariConfig.TRAINING_SET_META_DATA))
