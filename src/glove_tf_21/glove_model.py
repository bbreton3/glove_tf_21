import tensorflow as tf


class GloveModel(tf.keras.Model):

    def __init__(self, vocab_size, dim=100, alpha=3 / 4, x_max=100):

        """
        Glove model for training embeddings
        paper: https://nlp.stanford.edu/pubs/glove.pdf

        :param vocab_size: number of words in the vocab
        :param dim: dimension of the embeddings
        :param alpha: alpha scaling of the weight function
        :param x_max: x-max cutoff
        """
        super(GloveModel, self).__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.alpha = alpha
        self.x_max = x_max

        self.target_embeddings = tf.keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.dim, input_length=1, name="target_embeddings"
        )
        self.target_bias = tf.keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=1, input_length=1, name="target_bias"
        )

        self.context_embeddings = tf.keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.dim, input_length=1, name="context_embeddings"
        )
        self.context_bias = tf.keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=1, input_length=1, name="context_bias"
        )

        self.dot_product = tf.keras.layers.Dot(axes=-1, name="dot")

        self.prediction = tf.keras.layers.Add(name="add")

    def call(self, inputs):

        """
        Description of a forward pass duting the model training,
        index row and index cols are concatenated into an (N, 2)   tensor input

        :param inputs: Tensor of shape (N, 2)
        :return: prediction (coocurrence)
        """

        target_ix = inputs[:, 0]
        context_ix = inputs[:, 1]

        target_embeddings = self.target_embeddings(target_ix)
        target_bias = self.target_bias(target_ix)

        context_embeddings = self.context_embeddings(context_ix)
        context_bias = self.context_bias(context_ix)

        # (7) in the paper (page 4)
        dot_product = self.dot_product([target_embeddings, context_embeddings])
        prediction = self.prediction([dot_product, target_bias, context_bias])

        return prediction

    def glove_loss(self, y_true, y_pred):

        """
        Glove loss function (8) page 4 of the paper
        :param y_true: coocurrence values
        :param y_pred: predicted coocurrence values
        :return: loss
        """

        # Weigth (9) page 4 of the paper
        weight = tf.math.minimum(
            tf.math.pow(y_true / self.x_max, self.alpha), 1.0
        )

        return tf.math.reduce_mean(weight * tf.math.pow(y_pred - tf.math.log(y_true), 2.0))

