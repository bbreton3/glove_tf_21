from glove_tf_21.utils import save_labels

import tensorflow as tf


class SaveModelCallback(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self, file_path):

        """
        This Callback saves the best checkpoint (in terms of val loss) at the end of each epoch
        It also saves the current epoch number for training to be able to resume there

        :param file_path: path where you save the checkpoints (usually ./summaries/training_name/saved_checkpoint)
        """
        super().__init__(filepath=file_path)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch=epoch, logs=logs)
        save_labels([epoch + 1], f"{self.filepath}.last_epoch_number.txt")
