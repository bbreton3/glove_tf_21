import tensorflow as tf


class LrTensorboardCallback(tf.keras.callbacks.TensorBoard):
    """
    Inspired from : https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard

    This class adds the learning rate to the tensorboard
    """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
