import os
import tensorflow as tf
from glove_tf_21.utils.tfrecords_utils import parse_function, create_example, save_example_to_writer


def test_create_example(dummy_example):
    feature = dummy_example.features.feature

    assert len(feature) == 3

    assert feature.get("rows").int64_list.value == [1]
    assert feature.get("cols").int64_list.value == [4]
    assert feature.get("cooc").float_list.value == [2.5]


def test_parse_function(temp_folder_path, dummy_example):

    tfrecord_path = os.path.join(temp_folder_path, f"train_parse.tfrecord")

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        writer.write(dummy_example.SerializeToString())
        writer.close()

    test_dataset = tf.data.TFRecordDataset(tfrecord_path).map(parse_function)

    input_data, label_data = [x for x in test_dataset.as_numpy_iterator()][0]

    assert input_data.tolist() == [1, 4]
    assert label_data.tolist() == [2.5]

    os.remove(tfrecord_path)


def test_save_example_to_writer(temp_folder_path, dummy_example):

    tfrecord_path = os.path.join(temp_folder_path, f"train_save_example.tfrecord")

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        save_example_to_writer(writer, [1], [4], [2.5])

    test_dataset = tf.data.TFRecordDataset(tfrecord_path).map(parse_function)

    input_data, label_data = [x for x in test_dataset.as_numpy_iterator()][0]

    assert input_data.tolist() == [1, 4]
    assert label_data.tolist() == [2.5]

    os.remove(tfrecord_path)
