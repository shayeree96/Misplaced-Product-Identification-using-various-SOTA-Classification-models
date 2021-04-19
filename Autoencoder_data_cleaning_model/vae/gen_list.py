import json
import os
import numpy as np
from tqdm import tqdm
import shutil
from random import shuffle
import pandas as pd

#path=pd.read_csv('/home/sreena/shayeree/training/PatchVAE/finale_matchlist.txt',error_bad_lines=False,dtype='str')

train=pd.read_csv('/home/ironman/shayeree/PatchVAE/list_4cls.txt',error_bad_lines=False,dtype='str')
#train=shuffle(train)


def train_validate_test_split(df, train_percent=.7, validate_percent=.1, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

train, valid, test = train_validate_test_split(train)

# determine the path where to save the train and test file


# save the train and test file
# again using the '\t' separator to create tab-separated-values files
train.to_csv('train_try_4cls.txt', sep=',', index=False)
test.to_csv('valid_try_4cls.txt', sep=',', index=False)
valid.to_csv('test_try_4cls.txt', sep=',', index=False)
