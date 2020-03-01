# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for glove_tf_21.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

import os

import numpy as np
import pytest
from glove_tf_21.preprocessing_glove import PreprocessingGlove
from glove_tf_21.smart_label_encoder import SmartLabelEncoder
from glove_tf_21.utils.tfrecords_utils import create_example

from glove_tf_21.glove_model import GloveModel

from scipy.sparse import coo_matrix

import tensorflow as tf


CORPUS_PATH = "tests/resources/corpus/"
TEMP_FOLDER_PATH = "tests/resources/temp_folder"
CORPUS_FILE_NAME = "test_corpus_train.txt"
CORPUS_FILE_PATH = os.path.join(CORPUS_PATH, CORPUS_FILE_NAME)

COOC_MATRIX = np.array([
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 10., 5., 3.3333332538604736, 2.5, 2., 0., 0., 0., 0., 0.],
    [0., 0., 0., 10., 5., 3.3333332538604736, 2.5, 2., 0., 0., 0., 0.],
    [0., 0., 0., 0., 10., 5., 3.3333332538604736, 2.5, 2., 0., 0., 0.],
    [0., 0., 0., 0., 0., 10., 5., 3.3333332538604736, 2.5, 2., 0., 0.],
    [0., 0., 0., 0., 0., 0., 10., 5., 3.3333332538604736, 2.5, 2., 0.],
    [0., 0., 0., 0., 0., 0., 0., 10., 5., 3.3333332538604736, 2.5, 2.],
    [0., 0., 0., 0., 0., 0., 0., 0., 10., 5., 3.3333332538604736, 2.5],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 5., 3.3333332538604736],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 5.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
])

COOC_ROWS = np.array(
    [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9,
     9, 10])

COOC_COLS = np.array(
    [2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9, 6, 7, 8, 9, 10, 7, 8, 9, 10, 11, 8, 9, 10, 11, 9, 10,
     11, 10, 11, 11])

COOC_DATA = np.array(
    [10., 5., 3.3333332538604736, 2.5, 2., 10., 5., 3.3333332538604736, 2.5, 2., 10., 5., 3.3333332538604736, 2.5, 2.,
     10., 5., 3.3333332538604736, 2.5, 2., 10., 5., 3.3333332538604736, 2.5, 2., 10., 5., 3.3333332538604736, 2.5, 2.,
     10., 5., 3.3333332538604736, 2.5, 10., 5., 3.3333332538604736, 10., 5., 10.])

""" SENTENCES """


@pytest.fixture(scope="module")
def sentence():
    return [
        'All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.',
        'All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.',
        'potato', 'UNK'
    ]


@pytest.fixture(scope="module")
def sentences():
    return [
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['potato', 'UNK']
    ]


@pytest.fixture(scope="module")
def sentences_full():
    return [
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.']
    ]


""" SEQUENCES """


@pytest.fixture(scope="module")
def ix_sequence():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0]


@pytest.fixture(scope="module")
def ix_sequence_full():
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    ]


@pytest.fixture(scope="module")
def ix_sequences():
    return [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 0]
    ]


@pytest.fixture(scope="module")
def ix_sequences_full():
    return [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ]


""" FILES """


@pytest.fixture(scope="module")
def corpus_folder_path():
    return CORPUS_PATH


@pytest.fixture(scope="module")
def corpus_file_path():
    return CORPUS_FILE_PATH


@pytest.fixture(scope="module")
def temp_folder_path():
    return TEMP_FOLDER_PATH


@pytest.fixture(scope="module")
def load_corpus():
    return [line.split() for line in open(CORPUS_FILE_PATH, "r").readlines()]


""" CO-OCCURRENCE """


@pytest.fixture(scope="module")
def cooc_matrix():
    return COOC_MATRIX


@pytest.fixture(scope="module")
def cooc_matrix_sparse():
    return coo_matrix(COOC_MATRIX)


@pytest.fixture(scope="module")
def cooc_rows():
    return COOC_ROWS


@pytest.fixture(scope="module")
def cooc_cols():
    return COOC_COLS


@pytest.fixture(scope="module")
def cooc_data():
    return COOC_DATA


@pytest.fixture(scope="module")
def cooc_dict():
    return {
        (row, col): data
        for row, col, data in zip(COOC_ROWS, COOC_COLS, COOC_DATA)}


""" VOCAB """


@pytest.fixture(scope="module")
def vocab():
    return ['UNK', 'All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.']


@pytest.fixture(scope="module")
def ix2val():
    return {
        0: 'UNK', 1: 'All', 2: 'work', 3: 'and', 4: 'no', 5: 'play', 6: 'makes', 7: 'Jack', 8: 'a', 9: 'dull',
        10: 'boy', 11: '.'
    }


@pytest.fixture(scope="module")
def val2ix():
    return {
        'UNK': 0, 'All': 1, 'work': 2, 'and': 3, 'no': 4, 'play': 5, 'makes': 6, 'Jack': 7, 'a': 8, 'dull': 9,
        'boy': 10, '.': 11
    }


""" TF-RECORDS """


@pytest.fixture(scope="module")
def dummy_example():
    return create_example(1, 4, 2.5)


""" PRE-PROCESSING """


@pytest.fixture(scope="module")
def smart_label_encoder():
    return SmartLabelEncoder(min_occurrence=2, max_features=100, unk_token="UNK")


@pytest.fixture(scope="module")
def smart_label_encoder_fit():
    sle = SmartLabelEncoder(min_occurrence=2, max_features=100, unk_token="UNK")
    sle.fit(X=CORPUS_FILE_PATH)
    return sle


@pytest.fixture(scope="module")
def preprocessing_glove():
    prep_glove = PreprocessingGlove(data_path=CORPUS_PATH, min_occurrence=2, max_features=100)
    return prep_glove


@pytest.fixture(scope="module")
def preprocessing_glove_fit():
    prep_glove = PreprocessingGlove(data_path=CORPUS_PATH, min_occurrence=2, max_features=100)
    prep_glove()
    return prep_glove


""" TRAIN """


@pytest.fixture(scope="module")
def train_glove_model(preprocessing_glove_fit, cooc_rows, cooc_cols, cooc_data):

    vocab = preprocessing_glove_fit.get_labels()

    glove_model = GloveModel(vocab_size=len(vocab), dim=2)
    glove_model.build(input_shape=(2, 2))
    glove_model.compile(optimizer="adam", loss=glove_model.glove_loss)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (np.hstack([cooc_rows.reshape(-1, 1), cooc_cols.reshape(-1, 1)]),
         cooc_data.reshape(-1, 1))
    ).batch(2)

    glove_model.fit(test_dataset)






