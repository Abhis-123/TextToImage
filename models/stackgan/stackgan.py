import tensorflow as tf
from tensorflow.keras import backend as  K 
from tensorflow import keras
from tensorflow.keras import layers as L
import numpy as np
############################################################
# Conditioning Augmentation Network
############################################################
class ConditioningAugmentation(keras.Model):



    def __init__(self, *args, **kwargs):
        super(ConditioningAugmentation, self).__init(*args, **kwargs)
        self.dense = L.Dense(256)
        self.activation = L.LeakyReLU(alpha=0.2)
        

    def call(self,input,training=False):
        x  = self.dense(input,training)
        phi= self.activation(x)
        mean = phi[:,:128]
        std =tf.math.exp(phi[:,128:])
        epsilon = K.random_normal(shape=(mean.shape[0],),dtype='int32')
        output = mean + std*epsilon
        return output,phi

class EmbeddingCompresssor(keras.Model):
    def __init__(self):
        super(EmbeddingCompresssor, self).__init__()
        self.dense = L.Dense(128)

    def call(self,input,training=False):
        x = self.dense(input)
        x = L.LeakyReLU(0.2)(x)
        return x


############################################################
# Stage 1 Generator Network (CGAN)
############################################################


def UpSamplingBlock(input,num_filters):
    x = L.UpSampling1D(size=(2,2))(input)
    x = L.Conv2D(num_filters,kernel_size=3,padding='same',stride=1,use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return x


class Stage1Generator(keras.Model):
    def __init__(self):
        super(Stage1Generator, self).__init__()
        self.augmentation = ConditioningAugmentation()
        self.concat = L.Concatenate(axis=1)
        self.dense = tf.keras.layers.Dense(units = 128*8*4*4, kernel_initializer = tf.random_normal_initializer(stddev = 0.02))
        self.reshape = tf.keras.layers.Reshape(target_shape = (4, 4, 128*8), input_shape = (128*8*4*4, ))
        self.batchnorm1 = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.99)


    def call(self,inputs,training=False):
        embedding , Z = inputs
        c , phi = self.augmentation(embedding)

        gen_input = self.concat([c,Z])
        




        

