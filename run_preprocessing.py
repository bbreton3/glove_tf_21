import argparse
import os
from multiprocessing import cpu_count
from time import time

import tensorflow as tf
from glove_tf20.preprocessing_glove import PreprocessingGlove
from glove_tf20.utils import save_example_to_writer, save_labels, parallelize_function, split_share_data
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Preprocess ')
parser.add_argument('--data_path', help='path to the .txt file or the folder containing the .txt files')
parser.add_argument('--output_path', help='path where you want to store the tfrecords')
parser.add_argument('--max_features', help='size of the vocabulary to consider the UNK token is counted in this vocabulary size', default=10000)
parser.add_argument('--val_size', help='size of the corpus to keep for validation', default=0.05)
parser.add_argument('--tf_split', help='number of tfrecords files, default is number of cpu cores', default=None)
args = parser.parse_args()

data_path = args.data_path
output_path = args.output_path
if not os.path.isdir(output_path):
    os.mkdir(output_path)

max_features = int(args.max_features)
val_size = float(args.val_size)
tf_split = cpu_count() if args.tf_split is None else int(args.tf_split)

prep_glove = PreprocessingGlove(data_path, max_features=max_features)

glove_rows, glove_cols, glove_cooc, _ = prep_glove()


glove_rows_train, glove_rows_val, glove_cols_train, glove_cols_val, glove_cooc_train, glove_cooc_val = train_test_split(
    glove_rows, glove_cols, glove_cooc, test_size=val_size, shuffle=True, random_state=42)


train_writers = [tf.io.TFRecordWriter(os.path.join(output_path, f"train_{ix}.tfrecord")) for ix in range(tf_split)]
val_writers = [tf.io.TFRecordWriter(os.path.join(output_path, f"val_{ix}.tfrecord")) for ix in range(tf_split)]

processes = []

start_time = time()
train_raws_rows_list, train_raws_cols_list, train_raws_coocs_list = split_share_data(glove_rows_train, glove_cols_train, glove_cooc_train, split_n=tf_split)
parallelize_function(save_example_to_writer, tf_split, train_writers, train_raws_rows_list, train_raws_cols_list, train_raws_coocs_list)

save_labels([len(glove_rows_train)], os.path.join(output_path, "train_size.txt"))
#
# start_time = time()
val_raws_rows_list, val_raws_cols_list, val_raws_coocs_list = split_share_data(glove_rows_val, glove_cols_val, glove_cooc_val, split_n=tf_split)

parallelize_function(save_example_to_writer, tf_split, val_writers, val_raws_rows_list, val_raws_cols_list, val_raws_coocs_list)

save_labels([len(glove_rows_val)], os.path.join(output_path, "val_size.txt"))

save_labels(prep_glove.get_labels(), os.path.join(output_path, "labels.txt"))
