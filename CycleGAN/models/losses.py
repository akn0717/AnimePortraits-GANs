import tensorflow as tf
def consistency_loss(G, x, generated):
    return tf.reduce_mean(tf.abs(x - G(generated, training = True)))

def identity_loss(G, x):
    return tf.reduce_mean(tf.abs(G(x) - x))

def GAN_loss(y_true, logits):
    return tf.reduce_mean(tf.square(y_true - logits))