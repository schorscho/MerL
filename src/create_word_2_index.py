import os
import pandas as pd
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer

import timer
import data_preparation as mdp
from mercari_config import MercariConfig


nlp = spacy.load('en')
tokenizer = Tokenizer(nlp.vocab)

df = mdp.load_data(MercariConfig.TRAINING_SET_PREP_FILE)

set_len = len(df)

word2index = pd.DataFrame(columns=['word', 'word_id', 'count'])
word2index = word2index.astype(dtype={'word': str, 'word_id': int, 'count': int})

#word2index.set_index(['word'], inplace=True)

#word2index.loc[MercariConfig.PAD] = (0, 0)
#word2index.loc[MercariConfig.START] = (1, 0)
#word2index.loc[MercariConfig.OOV] = (2, 0)
#word2index.loc[MercariConfig.EMPTY_DESC] = (3, 0)
#word2index.loc[MercariConfig.REMOVED_PRICE] = (4, 0)

max_word_id = 5
max_words_in_item_desc = 0
row_i = 0

with timer.Timer():
    progress = 0
    
    for desc in df['item_description']:
        desc_doc = tokenizer(desc)
        words_in_col = len(desc_doc)

        for token in desc_doc:
            if not token.text in word2index.word:#index:
                #word2index.loc[token.text] = (max_word_id, 1)
                #word2index.at[token.text, 'word_id'] = max_word_id
                word2index.loc[row_i] = [token.text, max_word_id, 1]
                max_word_id += 1
                row_i += 1
            #else:
             #   word2index.at[token.text, 'count'] += 1

        max_words_in_item_desc = words_in_col if max_words_in_item_desc < words_in_col else max_words_in_item_desc
        
        progress += 1
        
        if progress % 100:
            print("Progress: %3.2f" % (progress * 100.0 / set_len))

mdp.save_word2index(word2index=word2index)

df_max_words = pd.DataFrame(columns=['file', 'max_words_in_item_desc'], dtype=int)

df_max_words.set_index(['file'], inplace=True)

df_max_words.loc['train'] = max_words_in_item_desc

df_max_words.to_csv(path_or_buf=os.path.join(MercariConfig.DATASETS_DIR, MercariConfig.TRAINING_SET_META_DATA))
