import os
from datetime import datetime


class MercariConfig:
    #ROOT_DIR = "/home/ubuntu"
    ROOT_DIR = "/Users/gopora/MyStuff/Dev/Workspaces"
    PROJECT_ROOT_DIR = os.path.join(ROOT_DIR, "Mercari")
    DATASETS_DIR = os.path.join(ROOT_DIR, "Data")
    TF_LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "tf_logs")
    TRAINING_SET_DATA_FILE = "mercari_train.tsv"
    TEST_SET_DATA_FILE = "mercari_test.tsv"

    @staticmethod
    def get_new_tf_log_dir():
        now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        log_dir = "{}/run-{}/".format(Config.TF_LOG_DIR, now)

        return log_dir