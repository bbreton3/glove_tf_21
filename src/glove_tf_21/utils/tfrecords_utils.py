import tensorflow as tf


""" TfRecords section """


def create_example(row_ix, col_ix, cooc_val):
    """
    Create a tf.train.Example object from the information of the sparse co-occurrence matrix

    Args:
        row_ix: index row of the matrix
        col_ix: index col of the matrix
        cooc_val: value of the matrix

    Returns: tf.train.Example

    """
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
    """
    Parse a Protobuf Example with the keys: 'rows', 'cols', 'cooc'
    return the result as tuple of:
    -   input: index rows and column rows concatenated (int)
    -   output: co-occurrence
    Args:
        example_proto: protobuf example

    Returns: input, output

    """
    feature_description = {
        "rows": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0),
        "cols": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0),
        "cooc": tf.io.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=0.0)
    }

    example = tf.io.parse_single_example(example_proto, feature_description)
    return tf.concat([example["rows"], example["cols"]], axis=0), example["cooc"]


def save_example_to_writer(writer, rows, cols, coocs):
    """
    save Examples to a tf.io.TFRecordWriter object

    Args:
        writer: tf.io.TFRecordWriter object
        rows: list or row indexes (int)
        cols: list or col indexes (int)
        coocs: list of co-occurrence (float)

    Returns: None

    """
    for row, col, cooc in zip(rows, cols, coocs):
        example = create_example(row, col, cooc)
        writer.write(example.SerializeToString())
    writer.close()
