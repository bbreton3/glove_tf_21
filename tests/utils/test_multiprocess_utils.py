import os

import numpy as np
import tensorflow as tf
from glove_tf_21.utils.multiprocess_utils import split_share_data, parallelize_function
from multiprocessing.sharedctypes import RawArray


def test_split_share_data(cooc_rows, cooc_cols, cooc_data):
    raw_array_int_10 = RawArray(typecode_or_type='i', size_or_initializer=10)
    raw_array_float_10 = RawArray(typecode_or_type='f', size_or_initializer=10)

    raws_rows_list, raws_cols_list, raws_coocs_list = split_share_data(cooc_rows, cooc_cols, cooc_data, 4)

    # Check Length
    assert len(raws_rows_list) == len(raws_cols_list) == len(raws_coocs_list)
    assert len(raws_rows_list[0]) == len(raws_cols_list[0]) == len(raws_coocs_list[0])

    # Check Type
    assert type(raws_rows_list[0]) == type(raws_cols_list[0]) == type(raw_array_int_10)
    assert type(raws_coocs_list[0]) == type(raw_array_float_10)


def test_parallelize_function(temp_folder_path):
    tfrecords_paths = [os.path.join(temp_folder_path, f"train_{ix}.tfrecord") for ix in range(2)]

    int_list = [1, 2]

    def write_int(writer, int_val):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "int": tf.train.Feature(int64_list=tf.train.Int64List(value=[int_val]))
                }
            )
        )
        writer.write(example.SerializeToString())
        writer.close()

    test_writers = [tf.io.TFRecordWriter(tfrecords_path) for tfrecords_path in tfrecords_paths]

    parallelize_function(write_int, 2, test_writers, int_list)

    def parse_int(example_proto):
        feature_description = {
            "int": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0),
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        return example["int"]

    test_dataset = tf.data.TFRecordDataset(tfrecords_paths).map(parse_int)

    assert np.hstack([x for x in test_dataset.as_numpy_iterator()]).tolist() == int_list

    [os.remove(tfrecords_path) for tfrecords_path in tfrecords_paths]