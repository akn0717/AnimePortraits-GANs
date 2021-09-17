from keras.layers import Layer
import tensorflow as tf
import keras.backend as K

class AdaIN(Layer):
    def __init__(self, 
             axis=-1,
             epsilon=1e-3,
             **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
    
    
    def build(self, input_shape):
        super(AdaIN, self).build(input_shape) 
    
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        x = inputs[0]
        beta_1 = inputs[1]
        beta_2 = inputs[2]

        axis = list(range(0, len(input_shape)))

        if self.axis is not None:
            del axis[self.axis]
        del axis[0]

        mean = K.mean(x, axis = axis, keepdims=True)
        stddev = K.std(x, axis = axis, keepdims=True) + self.epsilon
        normalized_x = (x - mean) / stddev

        return normalized_x * beta_2 + beta_1
    
    def compute_output_shape(self, input_shape):
    
        return input_shape[0]
