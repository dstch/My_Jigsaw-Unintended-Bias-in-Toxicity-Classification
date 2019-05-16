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
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, \
    concatenate, Layer
from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from keras import backend as K
from keras import initializers, regularizers, constraints
import lightgbm as lgb

EMB_PATHS = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]
EMB_SIZE = 300
MAX_LEN = 220
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 768
FOLD_NUM = 3
OOF_NAME = 'predicted_target'
BATCH_SIZE = 1024
PATIENCE = 3


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


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    return text


def load_data():
    logger.info('Load data')
    train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
    test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
    sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

    train['comment_text'] = train['comment_text'].astype(str)
    test['comment_text'] = test['comment_text'].astype(str)

    # adding preprocessing from this kernel: https://www.kaggle.com/taindow/simple-cudnngru-python-keras
    punct_mapping = {"_": " ", "`": " "}
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    logger.info('Clean data')
    train['comment_text'] = train['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
    test['comment_text'] = test['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

    train['target'] = np.where(train['target'] >= 0.5, True, False)
    return train, test, sub


def run_tokenizer(train, test):
    logger.info('Fitting tokenizer')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(train['comment_text']) + list(test['comment_text']))
    # X_train = tokenizer.texts_to_sequences(list(train['comment_text']))
    # X_test = tokenizer.texts_to_sequences(list(test['comment_text']))
    # X_train = pad_sequences(X_train, maxlen=MAX_LEN)
    # X_test = pad_sequences(X_test, maxlen=MAX_LEN)
    # word_index = tokenizer.word_index
    return tokenizer  # X_train, X_test, word_index


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


def custom_loss(y_true, y_pred):
    return binary_crossentropy(K.reshape(y_true[:, 0], (-1, 1)), y_pred) * y_true[:, 1]


# Attention GRU network
# https://www.jianshu.com/p/31c0acf94e0e
class AttLayer1(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer1, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer1, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


class AttLayer(Layer):
    def __init__(self, step_dim, **kwargs):
        self.step_dim = step_dim
        # initializers：初始化方法，normal是初始化采用的算法
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)  # 固定写法

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.features_dim = input_shape[-1]
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        # eij = K.tanh(K.dot(x, self.W))
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.features_dim


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def build_model(embedding_matrix, X_train, y_train, X_valid, y_valid):
    '''
    credits go to: https://www.kaggle.com/thousandvoices/simple-lstm/
    '''
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)

    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    att = Attention(MAX_LEN)(x)
    hidden = concatenate([att, GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x), ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    # loss如果是列表，则模型的output需要是对应的列表
    # model.compile(loss=[custom_loss, 'binary_crossentropy'], optimizer='adam', metrics=["accuracy"])

    # clr
    def scheduler(epoch):
        # 每隔100个epoch，学习率减小为原来的1/10
        if epoch % 100 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)

    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=3, validation_data=(X_valid, y_valid),
              verbose=2, callbacks=[early_stop])  # reduce_lr add in callbacks's list
    # aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    #
    # model = Model(inputs=words, outputs=[result, aux_result])
    # model.compile(loss=[custom_loss, 'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer='adam')

    return model


def train_model(X, X_test, y, tokenizer, embedding_matrix):
    oof = np.zeros((len(X), 1))
    prediction = np.zeros((len(X_test), 1))
    scores = []
    test_tokenized = tokenizer.texts_to_sequences(test['comment_text'])
    X_test = pad_sequences(test_tokenized, maxlen=MAX_LEN)
    folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=11)
    weights = []
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        valid_df = X_valid.copy()

        train_tokenized = tokenizer.texts_to_sequences(X_train['comment_text'])
        valid_tokenized = tokenizer.texts_to_sequences(X_valid['comment_text'])

        X_train = pad_sequences(train_tokenized, maxlen=MAX_LEN)
        X_valid = pad_sequences(valid_tokenized, maxlen=MAX_LEN)

        model = build_model(embedding_matrix, X_train, y_train, X_valid, y_valid)
        pred_valid = model.predict(X_valid)
        oof[valid_index] = pred_valid
        valid_df[OOF_NAME] = pred_valid

        # bias_metrics_df = compute_bias_metrics_for_model(valid_df, identity_columns, OOF_NAME, 'target')
        # scores.append(get_final_metric(bias_metrics_df, calculate_overall_auc(valid_df, OOF_NAME)))

        prediction += model.predict(X_test, batch_size=1024, verbose=1)
        # weights.append(2 ** fold_n)

    # preds = np.average(prediction, weights=weights, axis=0)
    preds = prediction / FOLD_NUM
    # print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    return preds


def lgb_model(train, test):
    features = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'rating', 'funny', 'wow', 'sad',
                'likes', 'disagree', 'sexual_explicit', 'identity_annotator_count', 'toxicity_annotator_count']
    X_train = train[features]
    y_train = train['target']
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
        max_depth=15, n_estimators=6000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4
    )
    clf.fit(X_train, y_train)
    prediction = clf.predict(test[features])
    return prediction


if __name__ == '__main__':
    train, test, sub = load_data()
    # prediction = lgb_model(train, test)
    tokenizer = run_tokenizer(train, test)
    embedding_matrix = np.concatenate(
        [build_embedding_matrix(f, tokenizer.word_index) for f in EMB_PATHS], axis=-1)
    prediction = train_model(train, test, train['target'], tokenizer, embedding_matrix)
    sub['prediction'] = prediction
    sub.to_csv('submission.csv', index=False)
"""
['id        ', 'target    ', 'comment_text', 'severe_toxicity', 'obscene   ', 'identity_attack', 'insult    ',
'threat    ', 'created_date', 'publication_id', 'article_id', 'rating    ', 'funny     ', 'wow       ', 'sad       ',
'likes     ', 'disagree  ', 'sexual_explicit', 'identity_annotator_count', 'toxicity_annotator_count']
"""
