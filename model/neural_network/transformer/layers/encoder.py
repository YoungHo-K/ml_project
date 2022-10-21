import tensorflow as tf

from model.neural_network.transformer.layers.attention import MultiHeadAttention
from model.neural_network.transformer.utils.positional_encoding import PositionalEncoding


class Encoder:
    def __init__(self, number_of_layers, dropout, number_of_feed_forward_units):
        self.number_of_layers = number_of_layers
        self.dropout = dropout
        self.number_of_feed_forward_units = number_of_feed_forward_units

    def generate(self, length_of_sequence, dimension_of_model, number_of_heads):
        input_of_encoder = tf.keras.Input(shape=(None, dimension_of_model))

        encoded_input = PositionalEncoding(length_of_sequence, dimension_of_model)(input_of_encoder)
        output_of_encoder = tf.keras.layers.Dropout(rate=self.dropout)(encoded_input)

        for _ in range(self.number_of_layers):
            output_of_encoder = self._encoder_layer(length_of_sequence=length_of_sequence,
                                                    dimension_of_model=dimension_of_model,
                                                    number_of_heads=number_of_heads)(output_of_encoder)

        return tf.keras.Model(inputs=input_of_encoder, outputs=output_of_encoder)

    def _encoder_layer(self, dimension_of_model, number_of_heads):
        input_of_layer = tf.keras.Input(shape=(None, dimension_of_model))

        attention_layer = MultiHeadAttention(dimension_of_model=dimension_of_model,
                                             number_of_heads=number_of_heads)(query=input_of_layer,
                                                                              key=input_of_layer,
                                                                              value=input_of_layer,
                                                                              mask=None,
                                                                              batch_size=tf.shape(input_of_layer)[0])
        attention_layer = tf.keras.layers.Dropout(rate=self.dropout)(attention_layer)
        attention_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(input_of_layer + attention_layer)

        feed_forward_layer = tf.keras.layers.Dense(units=self.number_of_feed_forward_units, activation="elu")(attention_layer)
        feed_forward_layer = tf.keras.layers.Dense(units=dimension_of_model)(feed_forward_layer)
        feed_forward_layer = tf.keras.layers.Dropout(rate=self.dropout)(feed_forward_layer)
        feed_forward_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_layer + feed_forward_layer)

        return tf.keras.Model(inputs=input_of_layer, outputs=feed_forward_layer)


