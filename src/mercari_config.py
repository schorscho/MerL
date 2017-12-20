import os
from datetime import datetime


class MercariConfig:
    ROOT_DIR = "/home/ubuntu"
    #ROOT_DIR = "/Users/gopora/MyStuff/Dev/Workspaces"
    PROJECT_ROOT_DIR = os.path.join(ROOT_DIR, "Mercari")
    DATASETS_DIR = os.path.join(ROOT_DIR, "data")
    TF_LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "tf_logs")
    TRAINING_SET_FILE = "mercari_train.csv"
    TEST_SET_FILE = "mercari_test.tsv"
    WORD_2_INDEX_FILE = "mercari_word_2_index.csv"
    TRAINING_SET_META_DATA = "mercari_train_meta_data.csv"
    TRAINING_SET_PREP_FILE = "mercari_train_prep.csv"
    TRAINING_ITEM_DESC_W2I_FILE = "mercari_train_item_desc_w2i.csv"

    PAD = '___PAD___'
    START = '___START___'
    OOV = '___OOV___'
    EMPTY_DESC = '___VERY_EMPTY_DESCRIPTION___'
    REMOVED_PRICE = '[RM]'
    EMPTY_CAT = '___VERY_EMPTY_CATEGORY___'
    EMPTY_BRAND = '___VERY_EMPTY_BRAND___'

    PAD_I = 0
    START_I = 1
    OOV_I = 2
    EMPTY_DESC_I = 3
    REMOVED_PRICE_I = 4
    
    MAX_WORDS_FROM_INDEX = 20000
    INDICATOR_WORDS_COUNT = 4
    MAX_WORDS_ITEM_DESC = 300
    
    @staticmethod
    def get_new_tf_log_dir():
        now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        log_dir = "{}/run-{}/".format(Config.TF_LOG_DIR, now)

        return log_dir