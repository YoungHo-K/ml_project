import tensorflow as tf


class EntropySequenceModel:
    @staticmethod
    def generate(input_shape=None, number_of_classes=2):
        if number_of_classes is None:
            raise Exception("[ERROR] Invalid number of classes.")

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.LSTM(8, activation='tanh', input_shape=input_shape))
        model.add(tf.keras.layers.Dense(units=6, activation='relu'))
        model.add(tf.keras.layers.Dense(units=number_of_classes, activation='softmax'))

        return model


class TestModel:
    @staticmethod
    def generate(input_shape=None, channel_order=None):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv1D(filters=4, kernel_size=3, strides=1, padding='same', activation='relu',
                                         kernel_initializer='he_normal', input_shape=input_shape, data_format=channel_order))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=32, activation='relu'))

        model.add(tf.keras.layers.Dense(units=16, activation='relu'))

        model.add(tf.keras.layers.Dense(units=32, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Reshape((32, 4)))
        model.add(tf.keras.layers.UpSampling1D(size=2))
        model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='same', activation='relu',
                                         kernel_initializer='he_normal', input_shape=input_shape, data_format=channel_order))

        return model

