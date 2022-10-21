import tensorflow as tf

from model.neural_network.transformer.layers.attention import MultiHeadAttention
from model.neural_network.transformer.utils.positional_encoding import PositionalEncoding


class Decoder:
    def __init__(self, number_of_layers, number_of_feed_forward_units, dropout):
        self.number_of_layers = number_of_layers
        self.number_of_feed_forward_units = number_of_feed_forward_units
        self.dropout = dropout

    def generate(self, length_of_sequence, dimension_of_model, number_of_heads):
        input_of_decoder = tf.keras.Input(shape=(None, dimension_of_model))
        output_of_encoder = tf.keras.Input(shape=(None, dimension_of_model))
        look_ahead_mask = tf.keras.Input(shape=(1, None, None))
        padding_mask = tf.keras.Input(shape=(1, 1, None))

        encoded_input = PositionalEncoding(length_of_sequence, dimension_of_model)(input_of_decoder)
        output_of_decoder = tf.keras.layers.Dropout(rate=self.dropout)(encoded_input)

        for _ in range(self.number_of_layers):
            output_of_decoder = self._decoder_layer(dimension_of_model=dimension_of_model,
                                                    number_of_heads=number_of_heads)(
                [output_of_decoder, output_of_encoder, look_ahead_mask, padding_mask])

        return tf.keras.Model(inputs=[input_of_decoder, output_of_encoder, look_ahead_mask, padding_mask],
                              outputs=output_of_decoder)

    def _decoder_layer(self, dimension_of_model, number_of_heads):
        input_of_decoder = tf.keras.Input(shape=(None, dimension_of_model))
        output_of_encoder = tf.keras.Input(shape=(None, dimension_of_model))
        look_ahead_mask = tf.keras.Input(shape=(1, None, None))
        padding_mask = tf.keras.Input(shape=(1, 1, None))

        first_attention_layer = MultiHeadAttention(dimension_of_model=dimension_of_model,
                                                   number_of_heads=number_of_heads)(query=input_of_decoder,
                                                                                    key=input_of_decoder,
                                                                                    value=input_of_decoder,
                                                                                    mask=look_ahead_mask,
                                                                                    batch_size=
                                                                                    tf.shape(input_of_decoder)[0])
        first_attention_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(input_of_decoder + first_attention_layer)

        second_attention_layer = MultiHeadAttention(dimension_of_model=dimension_of_model,
                                                    number_of_heads=number_of_heads)(query=first_attention_layer,
                                                                                     key=output_of_encoder,
                                                                                     value=output_of_encoder,
                                                                                     mask=padding_mask,
                                                                                     batch_size=
                                                                                     tf.shape(output_of_encoder)[0])
        second_attention_layer = tf.keras.layers.Dropout(rate=self.dropout)(second_attention_layer)
        second_attention_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(first_attention_layer + second_attention_layer)

        feed_forward_layer = tf.keras.layers.Dense(units=self.number_of_feed_forward_units, activation="elu")(second_attention_layer)
        feed_forward_layer = tf.keras.layers.Dense(units=dimension_of_model)(feed_forward_layer)
        feed_forward_layer = tf.keras.layers.Dropout(rate=self.dropout)(feed_forward_layer)
        feed_forward_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(feed_forward_layer + second_attention_layer)

        return tf.keras.Model(inputs=[input_of_decoder, output_of_encoder, look_ahead_mask, padding_mask],
                              outputs=feed_forward_layer)



