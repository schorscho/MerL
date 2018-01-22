import sys
import os
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from mercari_config import MercariConfig


logger = logging.getLogger('MerL.data_preparation')


def load_data(file_name, test=False, head=None):
    if test:
        index_col = 'test_id'
    else:
        index_col = 'train_id'
    
    data = pd.read_csv(filepath_or_buffer=os.path.join(MercariConfig.DATASETS_DIR, file_name), 
                            header=0, index_col=[index_col])
    
    data.sort_index(inplace=True)
    
    if head != None:
        data = data.head(head)
        
    return data


def save_data(data, file_name):
    data.to_csv(path_or_buf=os.path.join(MercariConfig.DATASETS_DIR, file_name))


def split_data(X, y, train_size, val_size):
    split = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=val_size, random_state=None)
 
    for train_i, val_i in split.split(X, y):
        if isinstance(X, pd.DataFrame):
            X_t = X.iloc[train_i]
            y_t = y[train_i]
            X_v = X.iloc[val_i]
            y_v = y[val_i]
        else:
            X_t = X[train_i]
            y_t = y[train_i]
            X_v = X[val_i]
            y_v = y[val_i]

    return X_t, y_t, X_v, y_v

def prepare_data(df):
    logger.info("Filling missing values ...")

    df['category_name'].fillna(value=MercariConfig.EMPTY_CAT, inplace=True)

    assert(len(df[df.category_name.isnull()]) == 0)

    df['brand_name'].fillna(value=MercariConfig.EMPTY_BRAND, inplace=True)

    assert(len(df[df.brand_name.isnull()]) == 0)

    df['item_description'].fillna(value=MercariConfig.EMPTY_DESC, inplace=True)

    assert(len(df[df.item_description.isnull()]) == 0)

    df['name'].fillna(value=MercariConfig.EMPTY_NAME, inplace=True)

    assert(len(df[df.name.isnull()]) == 0)

    logger.info("Filling missing values done.")


def execute_full_data_preparation(include_train=True, include_test=False):
    logger.info("Starting initial data preparation ...")

    if include_train:
        logger.info("Loading initial training data ...")

        df_train = load_data(MercariConfig.TRAINING_SET_FILE)
        
        logger.info("Preparing training data ...")

        prepare_data(df_train)
        
        logger.info("Preparing training data done.")

        logger.info("Splitting prepared training data ...")

        bins = np.linspace(df.price.min(), 1000, 10)

        train_data,_,val_data,_ = split_data(X=df, y=np.digitize(df.price, bins), train_size=0.16, val_size=0.04)

        logger.info("Saving prepared training data ...")

        save_data(train_data, MercariConfig.TRAINING_SET_PREP_FILE)

        logger.info("Saving prepared validation data ...")

        save_data(val_data, MercariConfig.VALIDATION_SET_PREP_FILE)

    if include_test:
        logger.info("Loading initial test data ...")

        df_test = load_data(MercariConfig.TEST_SET_FILE, test=True)
        
        logger.info("Preparing test data ...")

        prepare_data(df_test)

        logger.info("Preparing test data done.")
        logger.info("Saving prepared test data ...")

        save_data(df_test, MercariConfig.TEST_SET_PREP_FILE)

        logger.info("Saving prepared test data done.")

    logger.info("Initial data preparation done.")    
    

def main():
    b_train = False
    b_test = False
    
    for arg in sys.argv[1:]:
        if arg == 'training':
            b_train = True
        elif arg == 'test':
            b_test = True
    
    if (not b_train and not b_test):
        b_train = True
        
    execute_full_data_preparation(include_train=b_train, include_test=b_test)

    
if __name__ == "__main__":
    main()


