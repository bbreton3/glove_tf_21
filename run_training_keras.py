import argparse
import os

import tensorflow as tf
from glove_tf20.callbacks.embeddings_callback import EmbeddingCallback
from glove_tf20.callbacks.lr_tensorboard_callback import LrTensorboardCallback
from glove_tf20.callbacks.save_model_callback import SaveModelCallback
from glove_tf20.glove_model import GloveModel
from glove_tf20.utils.file_utils import load_file, load_size, get_train_files, get_val_files
from glove_tf20.utils.tfrecords_utils import parse_function

parser = argparse.ArgumentParser(description="Preprocess ")
parser.add_argument("--data_path", help="path to the tfrecords files")
parser.add_argument("--training_name", help="path where you want to store output model in the summaries folder")
parser.add_argument("--dim", help="dimension of the embeddings", default=100)
parser.add_argument("--batch_size", help="batch size", default=10000)
parser.add_argument("--epochs_number", help="total number of epochs", default=50)
parser.add_argument("--initial_epoch", help="initial epoch number", default=0)
parser.add_argument("--save_embeddings_every_epoch",
                    help="number of epochs between saving the embeddings in numpy and in the tensorboard", default=5)
args = parser.parse_args()

data_path = args.data_path
training_path = os.path.join("summaries", args.training_name)
dim = int(args.dim)
batch_size = int(args.batch_size)
epochs_number = int(args.epochs_number)
initial_epoch = int(args.initial_epoch)
save_embeddings_every_epoch = int(args.save_embeddings_every_epoch)

""" Load Metadata """
vocab = load_file(os.path.join(data_path, "labels.txt"))
train_size = load_size(os.path.join(data_path, "train_size.txt"))
val_size = load_size(os.path.join(data_path, "val_size.txt"))

""" Load Data"""
train_ds = tf.data.TFRecordDataset(get_train_files(data_path))
val_ds = tf.data.TFRecordDataset(get_val_files(data_path))
train_ds = train_ds.map(parse_function).shuffle(100000, reshuffle_each_iteration=True).batch(batch_size,
                                                                                             drop_remainder=True).repeat()
val_ds = val_ds.map(parse_function).shuffle(100000, reshuffle_each_iteration=True).batch(batch_size,
                                                                                         drop_remainder=True).repeat()

""" Create Glove Model"""
glove_model = GloveModel(vocab_size=len(vocab), dim=dim)
glove_model.build(input_shape=(batch_size, 2))
glove_model.compile(optimizer="adam", loss=glove_model.glove_loss)

""" Load previous weights, and restart training from the last epoch saved """
save_model_path = os.path.join(training_path, "save_model", "ckpt")
epoch_file_no_path = f"{save_model_path}.last_epoch_number.txt"
if os.path.isfile(epoch_file_no_path):
    initial_epoch = load_size(epoch_file_no_path)
    glove_model.load_weights(save_model_path)

optimizer = tf.keras.optimizers.Adam()

""" Callbacks """
save_model_callback = SaveModelCallback(filepath=save_model_path)
lr_tensorboard_callback = LrTensorboardCallback(log_dir=os.path.join(training_path, "tensorboard"))
embedding_callback = EmbeddingCallback(file_writer_path=os.path.join(training_path, "embeddings"),
                                       layer_names=["context_embedding", "target_embedding"], labels=vocab,
                                       max_number=5000, combined_embeddings=True,
                                       save_every_epoch=save_embeddings_every_epoch)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)

print(glove_model.summary())
glove_model.fit(x=train_ds, validation_data=val_ds, epochs=epochs_number,
                steps_per_epoch=train_size // batch_size,
                validation_steps=val_size // batch_size,
                callbacks=[lr_tensorboard_callback, embedding_callback, save_model_callback, reduce_lr],
                initial_epoch=initial_epoch
                )
# Save model
glove_model.save(os.path.join(training_path, "save_model"))
