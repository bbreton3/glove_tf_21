import glob
import os
import shutil

import pandas as pd
import numpy as np


def test_embeddings_callbacks(embeddings_folder_path, smart_label_encoder_fit, glove_model_train):
    vocab_size = len(smart_label_encoder_fit.classes_)

    # Check that the callback saved files
    assert set(os.listdir(embeddings_folder_path)) == {"epoch_2", "epoch_3", ".gitignore"}
    epoch_2_path = os.path.join(embeddings_folder_path, "epoch_2")
    epoch_3_path = os.path.join(embeddings_folder_path, "epoch_3")

    # Check that the embeddings were properly saved
    context_embeddings_3 = np.load(os.path.join(epoch_3_path, "context_embeddings_3.npy"))
    assert np.array_equal(context_embeddings_3, glove_model_train.context_embeddings.weights[0].numpy())

    target_embeddings_3 = np.load(os.path.join(epoch_3_path, "target_embeddings_3.npy"))
    assert np.array_equal(target_embeddings_3, glove_model_train.target_embeddings.weights[0].numpy())

    assert np.load(os.path.join(epoch_3_path, "target_embeddings_3.npy")).shape[0] == vocab_size

    # Check that the metadata was saved
    columns_set = {"name", "__init__", "10_clusters"}
    metadata_context_embeddings_3_df = pd.read_csv(os.path.join(epoch_3_path, "metadata_context_embeddings_3.tsv"),
                                                   sep="\t")
    assert len(metadata_context_embeddings_3_df) == vocab_size
    assert set(metadata_context_embeddings_3_df.columns) == columns_set

    metadata_target_embeddings_3_df = pd.read_csv(os.path.join(epoch_3_path, "metadata_target_embeddings_3.tsv"),
                                                  sep="\t")
    assert len(metadata_target_embeddings_3_df) == vocab_size
    assert set(metadata_target_embeddings_3_df.columns) == columns_set

    metadata_combined_embeddings_3_df = pd.read_csv(os.path.join(epoch_3_path, "metadata_combined_embeddings_3.tsv"),
                                                    sep="\t")
    assert len(metadata_combined_embeddings_3_df) == vocab_size
    assert set(metadata_combined_embeddings_3_df.columns) == columns_set

    # Check that the embeddings are present for the visualization
    files_saved = os.listdir(epoch_3_path)
    assert "projector_config.pbtxt" in files_saved
    assert len(glob.glob(epoch_3_path + "/embeddings-ckpt-*")) > 1
    assert len(glob.glob(epoch_3_path + "/events.out.tfevents.*")) > 0
    assert len(glob.glob(epoch_3_path + "/checkpoint*")) > 0

    # Remove the files
    shutil.rmtree(epoch_2_path)
    shutil.rmtree(epoch_3_path)
