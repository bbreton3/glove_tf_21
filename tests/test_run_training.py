import numpy as np


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

