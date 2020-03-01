from glove_tf_21.utils import flatten_lists_of_lists, safe_key_extractor, \
    get_files_in_folder, get_train_files, get_val_files, save_labels, \
    load_file, load_corpus_file, load_size

import os


def test_flatten_lists_of_lists(sentence, sentences):
    assert flatten_lists_of_lists(sentences) == sentence


def test_safe_key_extractor():
    dico = {"a": 1, "b": 2}

    assert safe_key_extractor(dico, "a", default_val=0) == 1
    assert safe_key_extractor(dico, "c", default_val=0) == 0


def test_get_files_in_folder(corpus_folder_path, corpus_file_path):

    assert get_files_in_folder(corpus_folder_path) == [corpus_file_path]
    assert get_files_in_folder(corpus_file_path) == [corpus_file_path]


def test_get_train_files(corpus_folder_path, corpus_file_path):

    assert get_train_files(corpus_folder_path) == [corpus_file_path]
    assert get_train_files(corpus_file_path) == [corpus_file_path]


def test_get_val_files(corpus_folder_path, corpus_file_path):

    assert get_val_files(corpus_folder_path) == []
    assert get_val_files(corpus_file_path) == []


def test_save_labels(temp_folder_path):

    labels_file_path = os.path.join(temp_folder_path, "temp_labels.txt")

    save_labels(["a", "b", "c"], labels_file_path)
    assert open(labels_file_path, "r").read().split("\n") == ["a", "b", "c"]

    os.remove(labels_file_path)


def test_load_file(corpus_file_path):

    list_of_elems = load_file(corpus_file_path)

    assert len(list_of_elems) == 10
    assert len(list_of_elems) == 10
    assert type(list_of_elems) == list
    assert type(list_of_elems[0]) == str
    assert list_of_elems[0][0:3] == "All"


def test_load_corpus_file(corpus_file_path):

    corpus = load_corpus_file(corpus_file_path)

    assert len(corpus) == 10
    assert len(corpus[0]) == 11
    assert isinstance(corpus, list)
    assert isinstance(corpus[0], list)
    assert isinstance(corpus[0][0], str)
    assert corpus[0][0] == "All"


def test_load_size(temp_folder_path):
    size_file_path = os.path.join(temp_folder_path, "temp_size.txt")

    save_labels([42], size_file_path)
    assert load_size(size_file_path) == 42

    os.remove(size_file_path)


