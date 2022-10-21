import tensorflow as tf

from model.neural_network.transformer.utils.utils import scaled_dot_product_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dimension_of_model, number_of_heads):
        super(MultiHeadAttention, self).__init__()

        assert dimension_of_model % number_of_heads == 0

        self.dimension_of_model = dimension_of_model
        self.number_of_heads = number_of_heads
        self.number_of_weight_features = dimension_of_model // number_of_heads

        self.query_weights = tf.keras.layers.Dense(units=dimension_of_model)
        self.key_weights = tf.keras.layers.Dense(units=dimension_of_model)
        self.value_weights = tf.keras.layers.Dense(units=dimension_of_model)
        self.output_weights = tf.keras.layers.Dense(units=dimension_of_model)

    def __call__(self, query, key, value, mask, batch_size):
        query = self.query_weights(query)
        key = self.key_weights(key)
        value = self.value_weights(value)

        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)

        attention_values, _ = scaled_dot_product_attention(query, key, value, mask, self.number_of_weight_features)
        attention_values = tf.transpose(attention_values, perm=[0, 2, 1, 3])

        concatenated_attention_values = tf.reshape(attention_values, (batch_size, -1, self.dimension_of_model))

        outputs_of_attention_layer = self.output_weights(concatenated_attention_values)

        return outputs_of_attention_layer

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, shape=(batch_size, -1, self.number_of_heads, self.number_of_weight_features))

        return tf.transpose(x, perm=[0, 2, 1, 3])
