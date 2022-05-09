import tensorflow as tf
def consistency_loss(G, x, generated):
    return tf.reduce_mean(tf.square(G(generated, training = True) - x))

def identity_loss(G, y):
    pass

def GAN_loss(y_true, logits):
    return tf.reduce_mean(tf.losses.binary_crossentropy(y_true, logits))