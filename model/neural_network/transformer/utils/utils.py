import tensorflow as tf


def scaled_dot_product_attention(query, key, value, mask, number_of_weight_features):
    matmul_query_and_key = tf.matmul(query, key, transpose_b=True)

    number_of_weight_features = tf.cast(number_of_weight_features, tf.float32)
    attention_score_matrix = matmul_query_and_key / tf.math.sqrt(number_of_weight_features)

    if mask is not None:
        attention_score_matrix += (mask * -1e-9)

    attention_weights = tf.nn.softmax(attention_score_matrix, axis=-1)
    attention_values = tf.matmul(attention_weights, value)

    return attention_values, attention_weights


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)

    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    length_of_sequence = tf.shape(x)[1]

    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((length_of_sequence, length_of_sequence)), -1, 0)

    padding_mask = create_padding_mask(x)

    return tf.maximum(look_ahead_mask, padding_mask)
