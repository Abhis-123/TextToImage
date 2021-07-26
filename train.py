


import os 

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from utils.dataset import Dataset
from models.stackgan import Stage1Model
import tensorflow as tf

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #model = Stage1Model()
    dataset = Dataset(image_size=(64,64),data_base_path="./data",batch_size=64)
    train_data = dataset.get_train_ds()
    #model.train(train_data,num_epochs=1)
    batch_iter = iter(train_data)
    print(len(dataset.train_filenames))

