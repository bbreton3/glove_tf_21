import os
import glob
import shutil

import numpy as np
import pandas as pd


def test_training(smart_label_encoder_fit, glove_model, glove_model_train):

    vocab_size = len(smart_label_encoder_fit.classes_)
    dim = glove_model.dim

    # Check Embeddings dimensions
    assert glove_model.context_embeddings.weights[0].shape.as_list() == glove_model_train.context_embeddings.weights[
        0].shape.as_list() == [vocab_size, dim]
    assert glove_model.target_embeddings.weights[0].shape.as_list() == glove_model.target_embeddings.weights[
        0].shape.as_list() == [vocab_size, dim]

    # Check that training is happening ie: embeddings are updated
    assert ~np.array_equal(glove_model.target_embeddings.weights[0].numpy(),
                           glove_model_train.target_embeddings.weights[0].numpy())

    assert ~np.array_equal(glove_model.context_embeddings.weights[0].numpy(),
                           glove_model_train.context_embeddings.weights[0].numpy())

    # Try a predict
    assert glove_model_train.predict(np.array([[0, 1], [2, 3]])).shape == (2, 1)


def test_embeddings_callbacks(embeddings_folder_path, smart_label_encoder_fit, glove_model_train):
    vocab_size = len(smart_label_encoder_fit.classes_)

    # Check that the callback saved files
    assert set(os.listdir(embeddings_folder_path)) == {"epoch_2", ".gitignore"}
    epoch_2_path = os.path.join(embeddings_folder_path, "epoch_2")

    # Check that the embeddings were properly saved
    context_embeddings_2 = np.load(os.path.join(epoch_2_path, "context_embeddings_2.npy"))
    assert np.array_equal(context_embeddings_2, glove_model_train.context_embeddings.weights[0].numpy())

    target_embeddings_2 = np.load(os.path.join(epoch_2_path, "target_embeddings_2.npy"))
    assert np.array_equal(target_embeddings_2, glove_model_train.target_embeddings.weights[0].numpy())

    assert np.load(os.path.join(epoch_2_path, "target_embeddings_2.npy")).shape[0] == vocab_size

    # Check that the metadata was saved
    columns_set = {"name", "__init__", "10_clusters"}
    metadata_context_embeddings_2_df = pd.read_csv(os.path.join(epoch_2_path, "metadata_context_embeddings_2.tsv"),
                                                   sep="\t")
    assert len(metadata_context_embeddings_2_df) == vocab_size
    assert set(metadata_context_embeddings_2_df.columns) == columns_set

    metadata_target_embeddings_2_df = pd.read_csv(os.path.join(epoch_2_path, "metadata_target_embeddings_2.tsv"),
                                                  sep="\t")
    assert len(metadata_target_embeddings_2_df) == vocab_size
    assert set(metadata_target_embeddings_2_df.columns) == columns_set

    metadata_combined_embeddings_2_df = pd.read_csv(os.path.join(epoch_2_path, "metadata_combined_embeddings_2.tsv"),
                                                    sep="\t")
    assert len(metadata_combined_embeddings_2_df) == vocab_size
    assert set(metadata_combined_embeddings_2_df.columns) == columns_set

    # Check that the embeddings are present for the visualization
    files_saved = os.listdir(epoch_2_path)
    assert "projector_config.pbtxt" in files_saved
    assert len(glob.glob(epoch_2_path + "/embeddings-ckpt-*")) > 1
    assert len(glob.glob(epoch_2_path + "/events.out.tfevents.*")) > 0
    assert len(glob.glob(epoch_2_path + "/checkpoint*")) > 0

    # Remove the files
    shutil.rmtree(epoch_2_path)
