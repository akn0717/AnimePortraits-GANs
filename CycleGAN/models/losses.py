import tensorflow as tf
def consistency_loss(G, x, generated):
    return tf.reduce_mean(tf.losses.mean_absolute_error(x, G(generated, training = True)))

def identity_loss(G, y):
    pass

def GAN_loss(y_true, logits):
    return tf.reduce_mean(tf.losses.mse(y_true, logits))