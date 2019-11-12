# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.

        return tf.pow(vector, 3)

        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start

        self.embedding_dim = int(embedding_dim)
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_transitions = num_transitions
        self.trainable_embeddings = trainable_embeddings

        self.embeddings = tf.Variable(tf.random.truncated_normal((vocab_size, embedding_dim), stddev=0.05), trainable=self.trainable_embeddings)

        self.w1 = tf.Variable(tf.random.truncated_normal([self.hidden_dim, self.num_tokens * self.embedding_dim], stddev=0.05), trainable=trainable_embeddings)
        self.bias_for_hidden_layer = tf.Variable(tf.random.truncated_normal([1, self.hidden_dim], stddev=0.05), trainable=trainable_embeddings)
        self.w2 = tf.Variable(tf.random.truncated_normal([self.num_transitions, self.hidden_dim], stddev=0.05), trainable=trainable_embeddings)
        
        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        input_shape = inputs.get_shape().as_list()
        n_batch = input_shape[0]
        num_tokens = input_shape[1]

        input_embeddings = tf.reshape(tf.nn.embedding_lookup(self.embeddings, inputs), [n_batch, num_tokens * self.embedding_dim])

        bias_for_batch = tf.tile(self.bias_for_hidden_layer, [n_batch, 1])
        hidden_layer_state = tf.reshape(tf.matmul(input_embeddings, tf.transpose(self.w1)), [n_batch, self.hidden_dim])
        hidden_layer_state = tf.add(hidden_layer_state, bias_for_batch)
        hidden_layer_state = self._activation(hidden_layer_state)

        logits = tf.reshape(tf.matmul(hidden_layer_state, tf.transpose(self.w2)), [n_batch, self.num_transitions])

        # print(str(input_embeddings.get_shape()) + " embedding shape")


        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        batch_size = logits.get_shape().as_list()[0]

        loss = None
        regularization = None

        # Compute stable-softmax, softmax(z - max(z)) -> e^{z - max(z}}/Sigma(e^{z - max(z)})
        max_logits = tf.reshape(tf.reduce_max(logits, axis=1), [batch_size, 1])
        stable_factor = tf.subtract(logits, max_logits)
        stable_factor = tf.exp(stable_factor)
        denominator_factor = tf.reshape(tf.reduce_sum(stable_factor, axis=1), [batch_size, 1])
        stable_softmax = tf.divide(stable_factor, denominator_factor)

        # cross_entropy_loss = tf.math.log(stable_softmax)

        mask = tf.greater(labels, -1)
        mask = tf.dtypes.cast(tf.dtypes.cast(mask, 'int32'), 'float32')

        cross_entropy_loss = tf.multiply(stable_softmax, labels)
        cross_entropy_loss = tf.multiply(cross_entropy_loss, mask)
        cross_entropy_loss = tf.reshape(tf.reduce_sum(cross_entropy_loss, axis=1), [batch_size, 1])
        cross_entropy_loss = tf.math.log(tf.add(cross_entropy_loss, 1e-8))

        cross_entropy_loss = -1.0 * tf.reduce_mean(cross_entropy_loss)
        loss = cross_entropy_loss

        regularization = (self._regularization_lambda) * (tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2) + tf.nn.l2_loss(self.embeddings))

        # TODO(Students) End
        return loss + regularization
