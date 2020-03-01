from tensorboard.plugins import projector
from sklearn.cluster import KMeans
import tensorflow as tf
import pandas as pd
import numpy as np
import os


class EmbeddingCallback(tf.keras.callbacks.Callback):

    def __init__(self, file_writer_path, layer_names, labels, save_every_epoch=5, max_number=None, cluster_numbers=(10, 20, 50),
                 combined_embeddings=False):
        """

        Callback class that saves embeddings to

        :param file_writer_path:
        :param layer_names:
        :param labels:
        :param max_number:
        :param cluster_numbers:
        :param combined_embeddings:
        """
        super().__init__()

        self.file_writer_path = file_writer_path
        self.layer_names = layer_names
        if max_number is None:
            self.labels = labels
        else:
            self.labels = labels[:max_number]

        self.save_every_epoch = save_every_epoch
        self.max_number = max_number
        self.cluster_numbers = cluster_numbers
        self.combined_embeddings = combined_embeddings

    """ Tensorboard Embeddings """
    def save_metadata(self, embeddings_numpy, file_path):
        metadata_df = pd.DataFrame()
        metadata_df["name"] = self.labels
        metadata_df["__init__"] = metadata_df.index

        for cluster_number in self.cluster_numbers:
            kmeans = KMeans(n_clusters=cluster_number, n_jobs=-1)
            metadata_df[f"{cluster_number}_clusters"] = kmeans.fit_predict(embeddings_numpy)

        metadata_df.to_csv(file_path, encoding='utf-8', index=None, sep="\t")

    @staticmethod
    def add_tensor_info(embeddings_numpy, name, projector_config):

        metadata_file_name = f"metadata_{name}.tsv"

        embedding = projector_config.embeddings.add()
        embedding.tensor_name = name
        embedding.metadata_path = metadata_file_name

        tensor_embedding = tf.Variable(embeddings_numpy, name=name)

        return tensor_embedding, metadata_file_name

    def on_epoch_end(self, epoch, logs=None):

        if (epoch > 0) and (epoch % self.save_every_epoch == 0):

            file_writer_epoch_path = os.path.join(self.file_writer_path, f"epoch_{epoch + 1}")

            tf.summary.create_file_writer(file_writer_epoch_path)
            projector_config = projector.ProjectorConfig()

            tensor_embeddings = list()
            embeddings_numpy_list = list()
            embeddings_numpy_reduced_list = list()
            for layer_name in self.layer_names:

                name = f"{layer_name}_{epoch + 1}"
                embedding_layer = self.model.get_layer(layer_name)
                embeddings_numpy = embedding_layer.embeddings.numpy()
                if self.max_number is None:
                    embeddings_numpy_reduced = embeddings_numpy
                else:
                    embeddings_numpy_reduced = embeddings_numpy[:self.max_number, :]

                embeddings_numpy_list.append(embeddings_numpy)
                embeddings_numpy_reduced_list.append(embeddings_numpy_reduced)

                tensor_embedding, metadata_file_name = self.add_tensor_info(embeddings_numpy_reduced, name, projector_config)

                tensor_embeddings.append(tensor_embedding)

                self.save_metadata(embeddings_numpy=embeddings_numpy_reduced,
                                   file_path=os.path.join(file_writer_epoch_path, metadata_file_name))
                np.save(os.path.join(file_writer_epoch_path, name), embeddings_numpy)

            if self.combined_embeddings:
                name = f"combined_embeddings_{epoch + 1}"

                embeddings_numpy = np.mean(embeddings_numpy_list, axis=0)
                embeddings_numpy_reduced = np.mean(embeddings_numpy_reduced_list, axis=0)

                tensor_embedding, metadata_file_name = self.add_tensor_info(
                    embeddings_numpy_reduced, name, projector_config)

                tensor_embeddings.append(tensor_embedding)
                self.save_metadata(embeddings_numpy=embeddings_numpy_reduced,
                                   file_path=os.path.join(file_writer_epoch_path, metadata_file_name))
                np.save(os.path.join(file_writer_epoch_path, "combined_embeddings"), embeddings_numpy)

            projector.visualize_embeddings(file_writer_epoch_path, projector_config)
            saver = tf.compat.v1.train.Saver(tensor_embeddings)  # Must pass list or dict
            saver.save(sess=None, global_step=self.params['steps'] * (epoch + 1),
                       save_path=os.path.join(file_writer_epoch_path, "embeddings-ckpt"))



