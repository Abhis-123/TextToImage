import tensorflow as tf
from tensorflow.keras import backend as  K 
from tensorflow import keras
from tensorflow.keras import layers as L
import numpy as np
from tensorflow.python.ops.gen_math_ops import tanh
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
        embedding , noise = inputs
        c , phi = self.augmentation(embedding)

        gen_input = self.concat([c,noise])
        
        x = self.dense(gen_input)
        x = self.reshape(x)
        x =  self.batchnorm1(x)
        x = UpSamplingBlock(x, 512)
        x = UpSamplingBlock(x, 256)
        x = UpSamplingBlock(x,128)
        x = UpSamplingBlock(x,3)
        x = L.Conv2D(3,kernel_size=3,padding='same')(x)
        x = L.Activation('tanh')(x)

        return x,phi

class Stage1Discriminator(keras.Model):
    def __init__(self,*args, **kwargs):
        super(Stage1Discriminator, self).__init__(*args,**kwargs)

        self.l1 = L.Conv2D(64,kernel_size=4,stride=2,padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev = 0.02 ))
        self.l2 = L.Conv2D(128,kernel_size=4,stride=2,padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev =0.02))
        self.l3 = L.BatchNormalization(axis = -1)
        self.l4 = L.Conv2D(256,kernel_size=4,stride=2,padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev = 0.02))
        self.l5 = L.BatchNormalization(axis = -1)
        self.l6 = L.Conv2D(516,kernel_size=4,stride=2,padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev = 0.02))
        self.l7 = L.BatchNormalization(axis = -1)
        self.embedding = EmbeddingCompresssor()
        self.l9 = L.Reshape(target_shape = (1,1,128))
        self.concat= L.Concatenate() 
        self.l11= L.Conv2D(filters = 1024, kernel_size=4,stride=2,padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev = 0.02))
        self.l12= L.BatchNormalization(axis = -1)
        self.l13= L.Conv2D(filters = 1, kernel_size=4,stride=2,padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev = 0.02))
        

    def call(self,inputs,training=False):
        I , E = inputs #
        x  = self.l1(I)
        x  = self.l2(x)
        x  = self.l3(x)
        x  = self.l4(x)
        x  = self.l5(x)
        x  = self.l6(x)
        x  = self.l7(x)

        t  = self.embedding(E)
        t  = self.l9(t)
        t  = tf.tile(t,(1,4,4,1))

        merged_input  = self.concat([t,x])

        y = self.l11(merged_input)
        y =self.l12(y)
        y =L.LeakyReLU()(y)

        y = self.l13(y)

        return tf.squeeze(y)




class Stage1Model(keras.Model):
    def __init__(self):
        super(Stage1Model, self).__init__()
        self.generator = Stage1Generator() 
        self.discriminator= Stage1Discriminator()

    def summary(self):
        print('x')




        

