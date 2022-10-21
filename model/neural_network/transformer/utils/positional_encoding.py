import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, number_of_features, name="positional_encoding"):
        super(PositionalEncoding, self).__init__(name=name)

        self.position = position
        self.number_of_features = number_of_features

        self.positional_encoding = self._initialize()

    def __call__(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]

    def _initialize(self):
        angle_values = self.calculate_angles()

        sin = tf.math.sin(angle_values[:, 0::2])
        cos = tf.math.cos(angle_values[:, 1::2])

        values = np.zeros(angle_values.shape)
        values[:, 0::2] = sin
        values[:, 1::2] = cos

        positional_encoding = tf.constant(values)
        positional_encoding = positional_encoding[tf.newaxis, ...]

        print(f"[INFO] Shape of Positional Encoding : {positional_encoding.shape}")

        return tf.cast(positional_encoding, tf.float32)

    def calculate_angles(self):
        position = tf.range(self.position, dtype=tf.float32)[:, tf.newaxis]
        index = tf.range(self.number_of_features, dtype=tf.float32)[tf.newaxis, :]

        angles = 1 / tf.pow(10000, (2 * (index // 2)) / tf.cast(self.number_of_features, tf.float32))

        return position * angles