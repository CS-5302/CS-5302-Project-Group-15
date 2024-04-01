# This script serves a puropose to transform dataset such as that for MeDAL in an appropriate format for LLM input.
# Moroever, it also appropriately decreases its size to enable git commits (<= 100 MB)

# More information for the dataset can be found here: 
# MeDAL: https://github.com/McGill-NLP/medal
# SHAIP: https://www.shaip.com/
# MS^2:  https://aclanthology.org/2021.emnlp-main.594.pdf

# Neccessary Imports
import numpy as np
import pandas as pd

from pathlib import Path
import os

# Path for the data
# root_path = Path(os.getcwd() + '\\Datasets')
root_path = Path('C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/CS-5302-Project-Group-15/Datasets')
train_path = root_path / 'train'

# Loading train and file and glancing at it

train_df = pd.read_csv(train_path / 'train.csv')

# Depending on which data we are dealing with the preprocessing will be different. This script deals with MeDAL dataset.

# MeDAL dataset contains the columns 'ABSTRACT_ID, 'TEXT', 'LOCATION', 'LABEL'. The text column contains the notes for the medical procedure correspondingly named in label column.
# Therefore for appropriate processing of the column we make a new column as below. This way the LLM would be able to discern and query appropriate notes for a specific medical term
# Note that this type of querying might be more useful for docters rather than laymans. 

# This way the LLM will better understand what we feed it.
train_df['NOTES'] = 'notes on ' + train_df['LABEL'] + ': ' + train_df['TEXT']
# Sampling a small amount to maintain reasonable size (<= 100 MB)
sampled_train = train_df.sample(n = 50000)


# Saving updated file
sampled_train['NOTES'].to_csv(root_path / 'medical_definations_data.csv', index = False)

# If all notes needed as one txt file - see utils.join_text
# However seems like we will be considering all the 50000 notes as chunks each converted into its own embedding and fed into the vector database i.e. ChromaDB. 