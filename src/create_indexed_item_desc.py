import data_preparation as mdp
from mercari_config import MercariConfig


word2index = mdp.load_word2index(max_words_from_index=MercariConfig.MAX_WORDS_FROM_INDEX)

train_data = mdp.load_data(MercariConfig.TRAINING_SET_PREP_FILE, head=None)

train_data = mdp.index_item_desc(data=train_data, max_words_item_desc=MercariConfig.MAX_WORDS_ITEM_DESC, word2index=word2index)

mdp.save_data(train_data, MercariConfig.TRAINING_ITEM_DESC_W2I_FILE)

