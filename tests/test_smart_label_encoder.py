import numpy as np


def test_fit(smart_label_encoder_fit, vocab):

    assert smart_label_encoder_fit.features_n == len(vocab)
    assert smart_label_encoder_fit.classes_ == vocab


def test_dict(smart_label_encoder_fit, ix2val, val2ix):
    assert smart_label_encoder_fit.ix2val == ix2val
    assert smart_label_encoder_fit.val2ix == val2ix

    assert smart_label_encoder_fit.index_to_val(0) == "UNK"
    assert smart_label_encoder_fit.index_to_val(1) == "All"
    assert smart_label_encoder_fit.index_to_val(100) == "UNK"

    assert smart_label_encoder_fit.val_to_index("UNK") == 0
    assert smart_label_encoder_fit.val_to_index("All") == 1
    assert smart_label_encoder_fit.val_to_index("Tomato") == 0


def test_process_array_or_str(smart_label_encoder):

    def plus_one(x):
        return x + 1

    assert smart_label_encoder.process_array_or_str(list(), plus_one, int) == list()
    assert smart_label_encoder.process_array_or_str([1, 2], plus_one, int) == [2, 3]


def test_transform(sentence, corpus_file_path,  ix_sequence, ix_sequences, smart_label_encoder):

    assert smart_label_encoder.fit_transform(X=np.array(sentence)) == ix_sequence
    assert smart_label_encoder.transform(X=sentence) == ix_sequence

    smart_label_encoder.fit(X=corpus_file_path)
    assert smart_label_encoder.transform(X=sentence) == ix_sequence

    smart_label_encoder.fit(X=[corpus_file_path])
    assert smart_label_encoder.transform(X=np.array(sentence)) == ix_sequence


def test_inverse_transform(ix_sequence, ix_sequences, smart_label_encoder_fit):
    assert smart_label_encoder_fit.inverse_transform(X=ix_sequence) == [
        'All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.',
        'All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.',
        'UNK', 'UNK'
    ]
    assert smart_label_encoder_fit.inverse_transform(X=ix_sequences) == [
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['All', 'work', 'and', 'no', 'play', 'makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ['UNK', 'UNK']
    ]
