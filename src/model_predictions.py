import os
import logging
import sys

import numpy as np

#import keras.models as km
from keras.datasets import imdb
from keras import losses
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import sequence

import data_preparation as mdp
from mercari_config import MercariConfig


logger = logging.getLogger('MerL.nl_indexation')


def root_mean_squared_logarithmic_error(y_true, y_pred):
    ret = losses.mean_squared_logarithmic_error(y_true, y_pred)
    return K.sqrt(ret)


def root_mean_squared_error(y_true, y_pred):
    ret = losses.mean_squared_error(y_true, y_pred)
    return K.sqrt(ret)


def load_keras_model():
    model = load_model(os.path.join(MercariConfig.MODEL_DIR, 'merl_model-v2_10_10_10.h5'), 
                       custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                       'root_mean_squared_logarithmic_error': root_mean_squared_logarithmic_error})
    
    return model


def prepare_data_arrays(data, test=False):
    max_seq_len_item_desc = MercariConfig.MAX_WORDS_IN_ITEM_DESC + 1
    max_seq_len_name = MercariConfig.MAX_WORDS_IN_NAME + 1

    X_item_desc_seq = data[['id' + str(i) for i in range(max_seq_len_item_desc)]].as_matrix()
    X_name_seq = data[['nm' + str(i) for i in range(max_seq_len_name)]].as_matrix()
    x_cat = data['category_id'].as_matrix()
    x_brand = data['brand_id'].as_matrix()
    x_f = data[['item_condition_id', 'shipping']].as_matrix()
    y = None if test else data['price'].as_matrix()

    X_item_desc_seq = sequence.pad_sequences(
        X_item_desc_seq, maxlen=max_seq_len_item_desc, padding='post', truncating='post')
    X_name_seq = sequence.pad_sequences(
        X_name_seq, maxlen=max_seq_len_name, padding='post', truncating='post')

    return X_name_seq, x_f, x_cat, x_brand, X_item_desc_seq, y

    
def execute_model_predictions(model, data):
    X_name_seq, x_f, x_cat, x_brand, X_item_desc_seq, y = prepare_data_arrays(data=data, test=True)
    
    y_pred = model.predict(x=[x_f, x_cat, x_brand, X_name_seq, X_item_desc_seq], 
                           batch_size=None, verbose=1, steps=None)
    
    return y_pred


def save_predictions(data, y_pred):
    data['price'] = y_pred
    
    df = data[['price']]
    
    df.to_csv(path_or_buf=os.path.join(MercariConfig.DATASETS_DIR, 'sample_submission.csv'), header=True)


def main():
    logger.info("Loading prepared test data ...")

    data = mdp.load_data(MercariConfig.VALIDATION_SET_PREP_FILE, test=False)

    logger.info("Loading prepared test data done.")

    logger.info("Loading keras model ...")
    
    model = load_keras_model()

    logger.info("Loading keras model done.")

    logger.info("Perform model predictions ...")

    y_pred = execute_model_predictions(model, data)

    logger.info("Perform model predictions done.")

    logger.info("Saving model predictions ...")

    save_predictions(data, y_pred)

    logger.info("Saving model predictions done.")

    
if __name__ == "__main__":
    main()

