from re import template
from numpy.lib.type_check import imag
import tensorflow as tf
from tensorflow.keras import backend as  K 
from tensorflow import keras
from tensorflow.keras import layers as L
import time
import os
############################################################
# Conditioning Augmentation Network
############################################################
class ConditioningAugmentation(keras.Model):
    def __init__(self, *args, **kwargs):
        super(ConditioningAugmentation, self).__init__(*args, **kwargs)
        self.dense = L.Dense(256)
        self.activation = L.LeakyReLU(alpha=0.2)
        

    def call(self,input):
        x  = self.dense(input)
        phi= self.activation(x)
        mean = phi[:,:128]
        std =tf.math.exp(phi[:,128:])
        epsilon = K.random_normal(shape = K.constant((mean.shape[1], ), dtype = 'int32'))
        output = mean + std*epsilon
        return output,phi

class EmbeddingCompresssor(keras.Model):
    def __init__(self):
        super(EmbeddingCompresssor, self).__init__()
        self.dense = L.Dense(128)

    def call(self,input):
        x = self.dense(input)
        x = L.LeakyReLU(0.2)(x)
        return x


############################################################
# Stage 1 Generator Network (CGAN)
############################################################


def UpSamplingBlock(input,num_filters):
    x = L.UpSampling2D(size=2)(input)
    x = L.Conv2D(num_filters,kernel_size=3,padding='same',strides=1,use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return x


class Stage1Generator(keras.Model):
    def __init__(self):
        super(Stage1Generator, self).__init__(name='stage_1_generator')
        self.augmentation = ConditioningAugmentation()
        self.concat = L.Concatenate(axis=1)
        self.dense = tf.keras.layers.Dense(units = 128*8*4*4, kernel_initializer = tf.random_normal_initializer(stddev = 0.02))
        self.reshape = tf.keras.layers.Reshape(target_shape = (4, 4, 128*8), input_shape = (128*8*4*4, ))
        self.batchnorm1 = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.99)
        self.activation = L.ReLU()

    def call(self,inputs):
        embedding , noise = inputs
        c , phi = self.augmentation(embedding)
        gen_input = self.concat([c,noise])
        x = self.dense(gen_input)
        x = self.reshape(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
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
        self.l1 = L.Conv2D(64,kernel_size=4,strides=2,padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev = 0.02 ))
        self.l2 = L.Conv2D(128,kernel_size=4,strides=2,padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev =0.02))
        self.l3 = L.BatchNormalization(axis = -1)
        self.l4 = L.Conv2D(256,kernel_size=4,strides=2,padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev = 0.02))
        self.l5 = L.BatchNormalization(axis = -1)
        self.l6 = L.Conv2D(516,kernel_size=4,strides=2,padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev = 0.02))
        self.l7 = L.BatchNormalization(axis = -1)
        self.embedding = EmbeddingCompresssor()
        self.l9 = L.Reshape(target_shape = (1,1,128))
        self.concat= L.Concatenate() 
        self.l11= L.Conv2D(filters = 1024, kernel_size=4,strides=2,padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev = 0.02))
        self.l12= L.BatchNormalization(axis = -1)
        self.l13= L.Conv2D(filters = 1  , activation='sigmoid'
                            ,kernel_size=4,strides=2,padding='same',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev = 0.02))
        

    def call(self,inputs):
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
        y = tf.squeeze(y)
        return y


def KL_loss(y_true, y_pred):
    kl = tf.keras.losses.KLDivergence()
    loss =kl(y_true, y_pred)
    loss = tf.reduce_mean(loss)
    return loss

def loss(true_label,predicted_label):
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(true_label, predicted_label))
    return loss
    

class Stage1Model(tf.keras.Model):
  def __init__(self):
    super(Stage1Model, self).__init__()
    self.generator = Stage1Generator()
    self.discriminator = Stage1Discriminator()
    self.generator_optimizer = keras.optimizers.Adam(learning_rate= 0.00002, beta_1= 0.5 , beta_2= 0.999)
    self.discriminator_optimizer = keras.optimizers.Adam(learning_rate= 0.00002, beta_1= 0.5 , beta_2= 0.999)
    self.noise_dim = 100
    self.c_dim = 128
    self.loss = {}


  def load_weights(self,path):
    z_noise = tf.random.normal((1, self.noise_dim))
    embedding = tf.random.normal((1,1024))
    image, phi = self.generator([embedding, z_noise])
    logit = self.discriminator([image,embedding])
    self.generator.load_weights(path+"/stage1_generator.h5")
    self.discriminator.load_weights(path+"/stage1_discriminator.h5")

  def train(self, train_ds, batch_size = 64, num_epochs = 600,save_weights_epoch=5,train_length=8855):
    for epoch in range(num_epochs):
      print("Epoch %d/%d:\n "%(epoch + 1, num_epochs), end = "")
      start_time = time.time()
      if epoch % 100 == 0:
        K.set_value( self.generator_optimizer.learning_rate,  self.generator_optimizer.learning_rate / 2)
        K.set_value( self.generator_optimizer.learning_rate,  self.generator_optimizer.learning_rate / 2)
    
      generator_loss_log = []
      discriminator_loss_log = []
      steps_per_epoch = train_length//batch_size
      batch_iter = iter(train_ds)
      for i in range(steps_per_epoch):
        if i % 5 == 0:
          print("=", end = "")
        image_batch, embedding_batch = next(batch_iter)
        batch_size = image_batch.shape[0]
        z_noise = tf.random.normal((batch_size, self.noise_dim))

        mismatched_images = tf.roll(image_batch, shift = 1, axis = 0)

        real_labels = tf.random.uniform(shape = (batch_size, ), minval = 0.9, maxval = 1.0)
        fake_labels = tf.random.uniform(shape = (batch_size, ), minval = 0.0, maxval = 0.1)
        mismatched_labels = tf.random.uniform(shape = (batch_size, ), minval = 0.0, maxval = 0.1)

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
          fake_images,phi = self.generator([embedding_batch, z_noise])
          real_logits = self.discriminator([image_batch, embedding_batch])
          fake_logits = self.discriminator([fake_images, embedding_batch])
          mismatched_logits = self.discriminator([mismatched_images, embedding_batch])

          l_sup = loss(real_labels, fake_logits)
            
          l_klreg = KL_loss(tf.random.normal((phi.shape[0], phi.shape[1])), phi)
          generator_loss = l_sup + 2.0*l_klreg
          l_real = loss(real_labels, real_logits)
          l_fake = loss(fake_labels, fake_logits)
          l_mismatched = loss(mismatched_labels, mismatched_logits)
          discriminator_loss = 0.5*tf.add(l_real, 0.5*tf.add(l_fake, l_mismatched))
        
        generator_gradients = generator_tape.gradient(generator_loss, self.generator.trainable_variables)
        discriminator_gradients = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
        
        generator_loss_log.append(generator_loss)
        discriminator_loss_log.append(discriminator_loss)

      end_time = time.time()

      if epoch % 1 == 0:
        epoch_time = end_time - start_time
        template = " - generator_loss: {:.4f} - discriminator_loss: {:.4f} - epoch_time: {:.2f} s"
        print(template.format(tf.reduce_mean(generator_loss_log), tf.reduce_mean(discriminator_loss_log), epoch_time))

      if (epoch + 1) % save_weights_epoch == 0 or epoch == num_epochs - 1:
        save_path = "./lr_results/epoch_" + str(epoch + 1)
        temp_embeddings = None
        for _, embeddings in train_ds:
          temp_embeddings = embeddings.numpy()
          break
        if os.path.exists(save_path) == False:
          os.makedirs(save_path)
        temp_batch_size = 10
        temp_z_noise = tf.random.normal((temp_batch_size, self.noise_dim))
        temp_embedding_batch = temp_embeddings[0:temp_batch_size]
        fake_images, _ = self.generator([temp_embedding_batch, temp_z_noise])
        for i, image in enumerate(fake_images):
          image = 127.5*image + 127.5
          image = image.numpy().astype('uint8')
          image  = keras.preprocessing.image.array_to_img(image)
          image.save(save_path + "/gen_%d.png"%(i))

        weights_path = f"./weights/weights_{epoch+1}"

        if os.path.exists(weights_path)== False:
          os.makedirs(weights_path)
        self.generator.save_weights(weights_path+"/stage1_generator.h5")
        self.discriminator.save_weights(weights_path+"/stage1_discriminator.h5")
        clear_output(wait=True)

    
def generate_image(self, embedding, batch_size= 64):
        #self.generator.compile(loss = "mse", optimizer = "adam")
        #self.generator.load_weights("stage1_generator_600.ckpt").expect_partial()
        z_noise = tf.random.normal((batch_size, self.noise_dim))
        generated_image = self.generator([embedding, z_noise])
        return generated_image                
               

           



def ResidualBlock(input, num_filters):
    x  =  L.Conv2D(filters = num_filters, kernel_size= 3 , strides=1, padding='same')(input)
    x  =  L.BatchNormalization()(x)
    x  =  L.ReLU()(x)
    x  =  L.Conv2D(filters = num_filters, kernel_size=3, strides=1, padding='same')(x)
    x  =  L.BatchNormalization()(x)
    x  =  L.ReLU()(x)
    return x


class Stage2Generator(keras.Model):
    def __init__(self,*args, **kwargs):
        super(Stage2Generator, self).__init__(*args, **kwargs)
        self.augmentation = ConditioningAugmentation()
        self.reshape = tf.keras.layers.Reshape(target_shape = (1, 1, 128))

    def call(self,inputs):
        image , embedding = inputs
        c,phi = self.augmentation(embedding)
        c = K.expand_dims(c, axis=1)
        c = K.expand_dims(c, axis=1)
        c = K.tile(c, [1, 16, 16, 1])
        x = DownSamplingBlock(image,num_filters=64,kernel_size = 3, strides = 1,batch_norm=False)
        x = DownSamplingBlock(x,num_filters=256)
        x = DownSamplingBlock(x,num_filters=512)
        x = K.concatenate([c, x], axis = 3)
                
        x = ResidualBlock(x, 128)
        x = ResidualBlock(x, 256)
        x = ResidualBlock(x, 128)
        x = UpSamplingBlock(x,256)
        x = ResidualBlock(x,256)
        x = UpSamplingBlock(x,256)
        x = ResidualBlock(x,128)
        x = UpSamplingBlock(x,256)
        x = ResidualBlock(x,128)
        x = UpSamplingBlock(x,3)
        
        return x,phi

def DownSamplingBlock(  inputs,
                        num_filters, 
                        kernel_size= 4,
                        strides = 2,
                        batch_norm=True,
                        activation= True):
    x = L.Conv2D(filters = num_filters, kernel_size= kernel_size , strides=strides, padding='same')(inputs)
    if batch_norm:
        x = L.BatchNormalization()(x)
    if activation:
        x = L.LeakyReLU()(x)
    #print(f" x shape {x.shape}")
    return x


class Stage2Discriminator(tf.keras.Model):
  def __init__(self):
    super(Stage2Discriminator, self).__init__()
    self.embed = EmbeddingCompresssor()
    self.reshape = tf.keras.layers.Reshape(target_shape = (1, 1, 128))
    self.conv_out = tf.keras.layers.Conv2D(filters = 1, kernel_size = 4, strides = 1, padding = "valid", kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.02))

  def call(self, inputs):
    I, E = inputs
    T = self.embed(E)
    T = self.reshape(T)
    T = tf.tile(T, (1, 4, 4, 1))
    
    x = DownSamplingBlock(I,num_filters=64,batch_norm=False)
    x = DownSamplingBlock(x,num_filters=128)
    x = DownSamplingBlock(x, num_filters=256)
    x = DownSamplingBlock(x,num_filters=512)
    x = DownSamplingBlock(x,num_filters=1024)
    x = DownSamplingBlock(x,num_filters=512)
    x = DownSamplingBlock(x,num_filters=128,kernel_size=1,strides=1)
    
    y = DownSamplingBlock(x, num_filters=128,kernel_size=1,strides=1)
    y = DownSamplingBlock(y, num_filters=256,kernel_size=3,strides=1)
    y = DownSamplingBlock(y, num_filters=128,kernel_size=3,strides=1)

    A = tf.keras.layers.Add()([x,y])
    A = tf.nn.leaky_relu(A)
    merged_input = tf.keras.layers.concatenate([A, T])
    
    z = DownSamplingBlock(merged_input,128,kernel_size=1,strides=1)
    z = self.conv_out(z)
    z = tf.squeeze(z)
    return z


class Stage2Model(keras.Model):
    def __init__(self,stage1model):
        super(Stage2Model, self).__init__()
        self.generator1 = stage1model.generator
        self.generator2 = Stage2Generator()
        self.discriminator2 = Stage2Discriminator()
        self.generator2_optimizer = keras.optimizers.Adam(learning_rate= 0.0001, beta_1= 0.5 , beta_2= 0.999)
        self.discriminator2_optimizer = keras.optimizers.Adam(learning_rate= 0.0001, beta_1= 0.5 , beta_2= 0.999)
        self.noise_dim = 100
    def load_weights(self,path_to_weights):
        z_noise = tf.random.normal((1, self.noise_dim))
    
        embedding = tf.random.normal((1,1024))
        image, phi = self.generator1([embedding, z_noise])
        hr_image, _ = self.generator2([image,embedding])
        logits    = self.discriminator2([hr_image,embedding])
        self.generator2.load_weights(path_to_weights+"/stage2_generator.h5")
        self.discriminator2.load_weights(path_to_weights+"/stage2_discriminator.h5")

    def train(self, train_ds, batch_size= 64, num_epochs =1,steps_per_epoch =125):

        for epoch in range(num_epochs):
            print("Epoch %d/%d:\n  "%(epoch + 1, num_epochs), end = "")
            start_time = time.time()
            if epoch % 100 == 0:
                K.set_value(self.generator2_optimizer.learning_rate, self.generator2_optimizer.learning_rate / 2)
                K.set_value(self.discriminator2_optimizer.learning_rate, self.discriminator2_optimizer.learning_rate / 2)
            
            generator_loss_log = []
            discriminator_loss_log = []
            steps_per_epoch = steps_per_epoch
            batch_iter = iter(train_ds) 
            for i in range(steps_per_epoch):
                if i% 5 ==0:
                    print("=", end = "")
                hr_image_batch,embedding_batch = next(batch_iter)
                batch_size = hr_image_batch.shape[0]
                z_noise = tf.random.normal((batch_size, self.noise_dim))
                mismatched_images = tf.roll(hr_image_batch, shift=1, axis = 0)
                real_labels = tf.random.uniform(shape = (batch_size, ), minval = 0.9, maxval=1.0)
                fake_labels = tf.random.uniform(shape = (batch_size, ), minval = 0.0 , maxval = 0.1)
                mismatched_labels = tf.random.uniform(shape = (batch_size, ), minval = 0.0, maxval = 0.1)
                with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                    lr_fake_images , _ = self.generator1([embedding_batch,z_noise])
                    hr_fake_images  ,  phi  = self.generator2([lr_fake_images,embedding_batch])
                    real_logits = self.discriminator2([hr_image_batch, embedding_batch])
                    del hr_image_batch  ## clear memory used by hr_image_batch 
                    fake_logits = self.discriminator2([hr_fake_images, embedding_batch])
                    del hr_fake_images  ## clear memory used by hr_fake_images 
                    mismatched_logits = self.discriminator2([mismatched_images, embedding_batch])
                    del mismatched_images  ## clear memory used by mismatched_images
                    l_sup = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_labels,real_logits))
                    l_klreg = KL_loss(tf.random.normal((phi.shape[0], phi.shape[1])), phi)
                    generator_loss = l_sup + 2.0*l_klreg
                    
                    
                    l_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_labels,real_logits))
                    l_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_logits,fake_labels))
                    l_mismatched = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(mismatched_logits,mismatched_labels))
                    discriminator_loss = 0.5*tf.add(l_real,tf.add(l_fake, l_mismatched))

                generator_gradients = generator_tape.gradient(generator_loss,self.generator2.trainable_variables)
                self.generator2_optimizer.apply_gradients(zip(generator_gradients, self.generator2.trainable_variables))             
                del generator_gradients
                discriminator_gradients = discriminator_tape.gradient(discriminator_loss,self.discriminator2.trainable_variables)
                self.discriminator2_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator2.trainable_variables))
                del discriminator_gradients
                generator_loss_log.append(generator_loss)
                discriminator_loss_log.append(discriminator_loss)

                end_time = time.time()

            if epoch % 1 == 0:
                epoch_time = end_time - start_time
                template = " - generator_loss: {:.4f} - discriminator_loss: {:.4f} - epoch_time: {:.2f} s"
                print(template.format(tf.reduce_mean(generator_loss_log), tf.reduce_mean(discriminator_loss_log), epoch_time))

            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                save_path = "./hr_results/epoch_" + str(epoch + 1)
                temp_embeddings = None
                for _, embeddings in train_ds:
                    temp_embeddings = embeddings.numpy()
                    break
                if os.path.exists(save_path) == False:
                    os.makedirs(save_path)
                temp_batch_size = temp_embeddings.shape[0]
                temp_z_noise = tf.random.normal((temp_batch_size, self.noise_dim))
                temp_embedding_batch = temp_embeddings[0:temp_batch_size]
                lr_temp_images,_ = self.generator1([temp_embedding_batch, temp_z_noise])
                fake_images,_= self.generator2([lr_temp_images,temp_embedding_batch])
                for i, image in enumerate(fake_images):
                        image = 127.5*image + 127.5
                        image = image.numpy().astype('uint8')
                        image  = keras.preprocessing.image.array_to_img(image)
                        image.save(save_path + "/gen_%d.png"%(i))

                weights_path = f"./weights/hr_weights_{epoch+1}"
                if os.path.exists(weights_path)== False:
                    os.makedirs(weights_path)
                self.generator2.save_weights(weights_path+"/stage2_generator.h5")
                self.discriminator2.save_weights(weights_path+"/stage2_discriminator.h5")



if __name__ == '__main__':
    import os
    disc = Stage1Discriminator()
    disc.summary()
        

