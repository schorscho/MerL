import os
from datetime import datetime


class MercariConfig:
    ROOT_DIR = "/home/ubuntu"
    #ROOT_DIR = "/Users/gopora/MyStuff/Dev/Workspaces"
    PROJECT_ROOT_DIR = os.path.join(ROOT_DIR, "Mercari")
    DATASETS_DIR = os.path.join(ROOT_DIR, "data")
    TF_LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "tf_logs")
    TRAINING_SET_FILE = "mercari_train.tsv"
    TEST_SET_FILE = "mercari_test.tsv"
    WORD_2_INDEX_FILE = "mercari_word_2_index.csv"
    PREP_TRAINING_SET_FILE = "mercari_train_prep.csv"
    
    PAD = '___PAD___'
    START = '___START___'
    OOV = '___OOV___'
    EMPTY_CAT = '___VERY_EMPTY_CATEGORY___'
    EMPTY_BRAND = '___VERY_EMPTY_BRAND___'
    EMPTY_DESC = '___VERY_EMPTY_DESCRIPTION___'

    @staticmethod
    def get_new_tf_log_dir():
        now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        log_dir = "{}/run-{}/".format(Config.TF_LOG_DIR, now)

        return log_dir