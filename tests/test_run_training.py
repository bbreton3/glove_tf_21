import numpy as np


def test_training(smart_label_encoder_fit, glove_model, glove_model_train):
    # import ipdb
    # ipdb.set_trace()
    vocab_size = len(smart_label_encoder_fit.classes_)
    dim = glove_model.dim

    # Check Embeddings dimensions
    assert glove_model.context_embedding.weights[0].shape.as_list() == glove_model_train.context_embedding.weights[
        0].shape.as_list() == [vocab_size, dim]
    assert glove_model.target_embedding.weights[0].shape.as_list() == glove_model.target_embedding.weights[
        0].shape.as_list() == [vocab_size, dim]

    # Check that training is happening ie: embeddings are updated
    assert ~np.array_equal(glove_model.target_embedding.weights[0].numpy(),
                           glove_model_train.target_embedding.weights[0].numpy())

    assert ~np.array_equal(glove_model.context_embedding.weights[0].numpy(),
                           glove_model_train.context_embedding.weights[0].numpy())
