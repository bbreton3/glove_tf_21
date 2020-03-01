from glove_tf_21 import PreprocessingGlove
from glove_tf_21.utils.file_utils import save_labels
from scipy.sparse import coo_matrix

import numpy as np
import pytest
import os


def test_cooc_count(preprocessing_glove, ix_sequences_full, cooc_dict):
    output_cooc = dict()
    for ix_seq in ix_sequences_full:
        output_cooc = preprocessing_glove.cooc_count(output_cooc, ix_seq)

    assert len(output_cooc) == len(cooc_dict)

    for key, val in cooc_dict.items():
        assert np.allclose(output_cooc[key], val)


def test_cooc_dict_to_sparse(preprocessing_glove_fit, cooc_dict, cooc_matrix_sparse):
    sparse_cooc_mat = preprocessing_glove_fit.cooc_dict_to_sparse(cooc_dict)
    assert np.sum(sparse_cooc_mat != cooc_matrix_sparse) == 0.0


def test_glove_formatter(preprocessing_glove, cooc_matrix_sparse, cooc_rows, cooc_cols, cooc_data):
    test_cooc_rows, test_cooc_cols, test_cooc_data = preprocessing_glove.glove_formatter(cooc_matrix_sparse)

    assert np.allclose(test_cooc_rows, cooc_rows)
    assert np.allclose(test_cooc_cols, cooc_cols)
    assert np.allclose(test_cooc_data, cooc_data)


def test_get_labels(preprocessing_glove_fit, vocab):
    assert preprocessing_glove_fit.get_labels() == vocab


def test_get_cooc_mat(preprocessing_glove_fit, corpus_file_path, cooc_matrix_sparse, temp_folder_path):
    test_cooc_matrix_sparse = preprocessing_glove_fit.get_cooc_mat(corpus_file_path)
    assert np.sum(test_cooc_matrix_sparse != cooc_matrix_sparse) == 0.0

    empty_file_path = os.path.join(temp_folder_path, "empty_file.txt")
    save_labels([""], empty_file_path)
    assert np.sum(preprocessing_glove_fit.get_cooc_mat(empty_file_path)) == 0.0

    os.remove(empty_file_path)


def test_call(preprocessing_glove_fit):

    cooc_rows, cooc_cols, cooc_data, cooc = preprocessing_glove_fit()

    assert len(cooc_rows) == 40
    assert len(cooc_cols) == 40
    assert len(cooc_data) == 40
