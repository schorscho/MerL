import sys
import logging
import os

import data_preparation as mdp
from mercari_config import MercariConfig


logger = logging.getLogger('MerL.data_categorization')


def build_category2index(data, category):
    category2index = data[[category + '_name', 'name']]
    category2index.columns = [category, category + '_cnt']
    category2index = category2index.groupby(category).count()

    category2index[category + '_id'] = [i for i in range(MercariConfig.WORD_I, len(category2index) + MercariConfig.WORD_I)]

    category2index.at[MercariConfig.PAD, category + '_id'] = MercariConfig.PAD_I
    category2index.at[MercariConfig.START, category + '_id'] = MercariConfig.START_I
    category2index.at[MercariConfig.OOV, category + '_id'] = MercariConfig.OOV_I
    category2index.at[MercariConfig.REMOVED_PRICE, category + '_id'] = MercariConfig.REMOVED_PRICE_I
    category2index.at[MercariConfig.EMPTY_NAME, category + '_id'] = MercariConfig.EMPTY_NAME_I
    category2index.at[MercariConfig.EMPTY_CAT, category + '_id'] = MercariConfig.EMPTY_CAT_I
    category2index.at[MercariConfig.EMPTY_BRAND, category + '_id'] = MercariConfig.EMPTY_BRAND_I
    category2index.at[MercariConfig.EMPTY_DESC, category + '_id'] = MercariConfig.EMPTY_DESC_I

    category2index[category + '_cnt'].fillna(value=0, inplace=True)
    category2index.sort_values(by=category + '_id', inplace=True)
    category2index = category2index.astype('int32')

    return category2index


def categorize_column(data, category, category2index):
    data[category + '_id'] = 0

    data_len = len(data)

    progress = 0

    row_iterator = data.iterrows()

    for index, row in row_iterator:
        if not progress % 10000:
            logger.info("Progress: %3.2f", (progress * 100.0 / data_len))

        cat_nm = row[category + '_name']

        if cat_nm in category2index.index:
            data.at[index, category + '_id'] = category2index.at[cat_nm, category + '_id']
        else:
            data.at[index, category + '_id'] = MercariConfig.EMPTY_CAT_I

        progress += 1

            
            
def execute_full_categorization(include_train, include_val, include_category, include_brand):
    logger.info("Starting categorization ...")

    cat2i_cat = 0
    cat2i_brand = 0
    train_data = 0
    val_data = 0

    logger.info("Loading prepared training data ...")

    cat_data_src = mdp.load_data(MercariConfig.TRAINING_SET_PREP_FILE)

    if include_train:
        train_data = {'data': cat_data_src, 'file': MercariConfig.TRAINING_SET_PREP_FILE}

    if include_val:
        logger.info("Loading prepared validation data ...")

        data = mdp.load_data(MercariConfig.VALIDATION_SET_PREP_FILE)
        val_data = {'data': data, 'file': MercariConfig.VALIDATION_SET_PREP_FILE}

    if include_category:
        logger.info("Building category2index for category ...")
        
        cat2i = build_category2index(data=cat_data_src, category='category')
        cat2i_cat = {'cat2i': cat2i, 'category': 'category'}

    if include_brand:
        logger.info("Building category2index for brand ...")
        
        cat2i = build_category2index(data=cat_data_src, category='brand')
        cat2i_brand = {'cat2i': cat2i, 'category': 'brand'}

    for data in [train_data, val_data]:
        if data != 0:
            for cat2i in [cat2i_cat, cat2i_brand]:
                if cat2i != 0:
                    logger.info("Performing categorization for file %s category %s", data['file'], cat2i['category'])
                    
                    categorize_column(data=data['data'], category2index=cat2i['cat2i'], category=cat2i['category'])

            logger.info("Saving prepared data file %s", data['file'])

            mdp.save_data(data['data'], data['file'])

    logger.info("Done with categorization.")    


def main():
    b_cat = False
    b_br = False
    b_train = False
    b_val = False
    
    for arg in sys.argv[1:]:
        if arg == 'category':
            b_cat = True
        elif arg == 'brand':
            b_br = True
        elif arg == 'training':
            b_train = True
        elif arg == 'validation':
            b_val = True
    
    if (not b_br and not b_cat):
        b_cat = True

    if (not b_train and not b_val):
        b_train = True
        
    execute_full_categorization(include_train=b_train, include_val=b_val, 
                                include_category=b_cat, include_brand=b_br)

    
if __name__ == "__main__":
    main()



