import logging
from multiprocessing import Process
from time import time

import numpy as np
import tensorflow as tf
from multiprocessing.sharedctypes import RawArray


def split_share_data(rows, cols, coocs, split_n):
    """
    This method takes the rows, cols and cooc(currence) from the glove co-occurrence sparse matrix and splits it
    in sub-lists, formatted in RawArray to be accessed by multiple processes in parallel.
    This allows keeps the GIL from replicating the memory space when multiprocessing

    :param rows: indexes of the non-empty rows of the co-occurrence matrix
    :param cols: indexes of the non-empty cols of the co-occurrence matrix
    :param coocs: non-empty values of the co-occurrence matrix
    :param split_n: number in which split the arrays
    :return: 3 lists of RawArrays
    """
    total_length = len(rows)

    raws_rows_list = list()
    raws_cols_list = list()
    raws_coocs_list = list()

    for ix in range(split_n):
        min_ix = ix * total_length // split_n
        max_ix = min((ix + 1) * total_length // split_n, total_length - 1)
        split_len = max_ix - min_ix

        # Create the empty RawArrays
        rows_raw = RawArray(typecode_or_type='i', size_or_initializer=split_len)
        cols_raw = RawArray(typecode_or_type='i', size_or_initializer=split_len)
        coocs_raw = RawArray(typecode_or_type='f', size_or_initializer=split_len)

        # Cast the c-types to numpy types, and reshape
        rows_np = np.frombuffer(rows_raw, dtype=np.int32).reshape(split_len)
        cols_np = np.frombuffer(cols_raw, dtype=np.int32).reshape(split_len)
        coocs_np = np.frombuffer(coocs_raw, dtype=np.float32).reshape(split_len)

        # Copy data to our shared array
        np.copyto(rows_np, rows[min_ix: max_ix])
        np.copyto(cols_np, cols[min_ix: max_ix])
        np.copyto(coocs_np, coocs[min_ix: max_ix])

        # Add data to the lists
        raws_rows_list.append(rows_raw)
        raws_cols_list.append(cols_raw)
        raws_coocs_list.append(coocs_raw)

    return raws_rows_list, raws_cols_list, raws_coocs_list


def parallelize_function(function, parallel_n, *iterables):
    """
    Parallelize a function between parallel_n processes,
    the tf.train.Coordinator allows for the tfrecords

    :param function: function to parallelize
    :param parallel_n: number of processes to run in parallel
    :param iterables: elements to apply the function on
    :return: None
    """
    coord = tf.train.Coordinator()
    processes = []
    start_time = time()

    for process_ix in range(parallel_n):
        args = tuple([iterable[process_ix] for iterable in iterables])
        p = Process(target=function, args=args)
        p.start()
        processes.append(p)
    coord.join(processes)
    logging.info(f"Saved to tfrecords {time() - start_time:2f} seconds")

