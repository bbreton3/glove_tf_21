from glove_tf20.glove_model import GloveModel
from glove_tf20.utils.file_utils import load_file, load_size, get_train_files, get_val_files
from glove_tf20.utils.tfrecords_utils import parse_function
from tqdm import tqdm
import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--data_path", help="path to the folder containing the tfrecords")
parser.add_argument("--training_name", help="path where you want to store output model in the summaries folder")
parser.add_argument("--dim", help="dimension of the vectors", default=100)
parser.add_argument("--batch_size", help="batch size", default=10000)
parser.add_argument("--epochs_number", help="total number of epochs", default=50)
parser.add_argument("--initial_epoch", help="initial epochs number", default=0)
args = parser.parse_args()

data_path = args.data_path
training_path = os.path.join("summaries", args.training_name)
dim = int(args.dim)
batch_size = int(args.batch_size)
epochs_number = int(args.epochs_number)
initial_epoch = int(args.initial_epoch)

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
print(glove_model.summary())


@tf.function
def train_step(examples, labels):
    with tf.GradientTape() as train_tape:
        predictions = glove_model(examples)
        train_loss = glove_model.glove_loss(labels, predictions)

    gradients = train_tape.gradient(train_loss, glove_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, glove_model.trainable_variables))

    return train_loss


@tf.function
def val_step(examples, labels):
    predictions = glove_model(examples)
    val_loss = glove_model.glove_loss(labels, predictions)

    return val_loss


train_loss_mean = tf.keras.metrics.MeanTensor(name="train_loss_mean")
val_loss_mean = tf.keras.metrics.MeanTensor(name="val_loss_mean")


for epoch in range(initial_epoch, epochs_number, 1):

    print(f"\nEpoch {epoch + 1} :")
    train_steps_per_epoch = train_size // batch_size
    train_pbar = tqdm(
        train_ds.enumerate(), total=train_steps_per_epoch, dynamic_ncols=True
    )
    average_train_loss = 0
    for train_ix, (examples, labels) in train_pbar:
        train_loss = train_step(examples, labels).numpy()
        average_train_loss = train_loss_mean(train_loss)
        train_pbar.set_postfix({
            "ave_train_loss": average_train_loss.numpy(),
            "inst_train_loss": train_loss
        })

    # Save model at the end of each epoch
    glove_model.save(os.path.join(training_path, "save_model"))

    val_steps_per_epoch = val_size // batch_size
    val_pbar = tqdm(val_ds.enumerate(), total=val_steps_per_epoch, dynamic_ncols=True)
    average_val_loss = 0
    for val_ix, (examples, labels) in val_pbar:
        val_loss = val_step(examples, labels)
        average_val_loss = val_loss_mean(val_loss)
        val_pbar.set_postfix({"val_loss": average_val_loss.numpy()})

    train_loss_mean.reset_states()
    val_loss_mean.reset_states()

# Save model
glove_model.save(os.path.join(training_path, "save_model"))
