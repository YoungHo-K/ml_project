import tensorflow as tf

from model.neural_network.transformer.layers.encoder import Encoder
from model.neural_network.transformer.layers.decoder import Decoder
from model.neural_network.transformer.utils.utils import create_padding_mask, create_look_ahead_mask


class Transformer:
    def __init__(self, number_of_layers, number_of_feed_forward_units, dropout):
        self.number_of_layers = number_of_layers
        self.number_of_feed_forward_units = number_of_feed_forward_units
        self.dropout = dropout

    def generate(self, length_of_sequence_for_encoder, length_of_sequence_for_decoder, number_of_features, number_of_output_features, dimension_of_model, number_of_heads):
        input_of_encoder = tf.keras.Input(shape=(None, number_of_features))
        input_of_decoder = tf.keras.Input(shape=(None, number_of_features))

        transformed_input_of_encoder = tf.keras.layers.Dense(units=dimension_of_model, activation="linear")(input_of_encoder)
        transformed_input_of_decoder = tf.keras.layers.Dense(units=dimension_of_model, activation="linear")(input_of_decoder)

        look_ahead_mask_of_encoder = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None))(tf.reduce_sum(input_of_decoder, axis=-1))
        padding_mask_of_decoder = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None))(tf.reduce_sum(input_of_encoder, axis=-1))

        encoder_architecture = Encoder(number_of_layers=self.number_of_layers, dropout=self.dropout, number_of_feed_forward_units=self.number_of_feed_forward_units)
        encoder = encoder_architecture.generate(length_of_sequence=length_of_sequence_for_encoder, dimension_of_model=dimension_of_model, number_of_heads=number_of_heads)
        output_of_encoder = encoder(inputs=transformed_input_of_encoder)

        decoder_architecture = Decoder(number_of_layers=self.number_of_layers, dropout=self.dropout, number_of_feed_forward_units=self.number_of_feed_forward_units)
        decoder = decoder_architecture.generate(length_of_sequence=length_of_sequence_for_decoder, dimension_of_model=dimension_of_model, number_of_heads=number_of_heads)
        output_of_decoder = decoder(inputs=[transformed_input_of_decoder, output_of_encoder, look_ahead_mask_of_encoder, padding_mask_of_decoder])

        outputs = tf.keras.layers.Dense(units=number_of_output_features, activation="linear")(output_of_decoder)

        return tf.keras.Model(inputs=[input_of_encoder, input_of_decoder], outputs=outputs)
