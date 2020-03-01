import tensorflow as tf


""" TfRecords section """


def create_example(row_ix, col_ix, cooc_val):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "rows": tf.train.Feature(int64_list=tf.train.Int64List(value=[row_ix])),
                "cols": tf.train.Feature(int64_list=tf.train.Int64List(value=[col_ix])),
                "cooc": tf.train.Feature(float_list=tf.train.FloatList(value=[cooc_val]))
            }
        )
    )


def parse_function(example_proto):

    feature_description = {
        "rows": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0),
        "cols": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0),
        "cooc": tf.io.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=0.0)
    }

    example = tf.io.parse_single_example(example_proto, feature_description)
    return tf.concat([example["rows"], example["cols"]], axis=0), example["cooc"]


def save_example_to_writer(writer, rows, cols, coocs):
    for row, col, cooc in zip(rows, cols, coocs):
        example = create_example(row, col, cooc)
        writer.write(example.SerializeToString())
    writer.close()
