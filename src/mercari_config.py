import os
from datetime import datetime


class MercariConfig:
    ROOT_DIR = "/home/ubuntu"
    #ROOT_DIR = "/Users/gopora/MyStuff/Dev/Workspaces"
    PROJECT_ROOT_DIR = os.path.join(ROOT_DIR, "Mercari")
    DATASETS_DIR = os.path.join(ROOT_DIR, "data")
    TF_LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "tf_logs")
    TRAINING_SET_FILE = "mercari_train.csv"
    WORD_2_INDEX_4_ITEM_DESC_FILE = "mercari_word_2_index_4_item_desc.csv"
    WORD_2_INDEX_4_NAME_FILE = "mercari_word_2_index_4_name.csv"
    TRAINING_SET_PREP_FILE = "mercari_train_prep.csv"
    VALIDATION_SET_PREP_FILE = "mercari_val_prep.csv"

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
    
    MAX_WORDS_FROM_INDEX_4_ITEM_DESC = 40000
    MAX_WORDS_FROM_INDEX_4_NAME = 40000
    MAX_WORDS_IN_ITEM_DESC = 300
    MAX_WORDS_IN_NAME = 20

    
    @staticmethod
    def get_new_tf_log_dir():
        now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        log_dir = "{}/run-{}/".format(Config.TF_LOG_DIR, now)

        return log_dir