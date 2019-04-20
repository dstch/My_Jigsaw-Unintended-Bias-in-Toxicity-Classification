#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: my_test.py
@time: 2019/4/19 23:27
@desc: 对文本用深度学习（LSTM、Capsule、Attention），其他特征用boosting一起
"""

import pandas as pd
import numpy as np
import gc
import time
import logging
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

EMB_PATHS = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]
EMB_SIZE = 300
MAX_LEN = 220


def get_logger():
    """
    set logger format
    :return:
    """
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger()

logger.info('Load data')
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

train['comment_text'] = train['comment_text'].astype(str)
test['comment_text'] = test['comment_text'].astype(str)

# adding preprocessing from this kernel: https://www.kaggle.com/taindow/simple-cudnngru-python-keras
punct_mapping = {"_": " ", "`": " "}
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    return text


logger.info('Clean data')
train['comment_text'] = train['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
test['comment_text'] = test['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

logger.info('Fitting tokenizer')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(train['comment_text']) + list(test['comment_text']))
X_train = tokenizer.texts_to_sequences(list(train['comment_text']))
X_test = tokenizer.texts_to_sequences(list(test['comment_text']))
X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def build_embedding_matrix(embedding_path, word_index):
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
    embedding_matrix = np.zeros((len(word_index) + 1, EMB_SIZE))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    del embedding_index
    gc.collect()
    return embedding_matrix
