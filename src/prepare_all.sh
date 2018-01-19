#!/bin/bash

python data_preparation.py

python word_2_index.py name item_description

python nl_indexation.py training validation name item_description

python data_categorization.py training validation category brand