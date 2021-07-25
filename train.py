


import os 

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from utils.dataset import Dataset
from models.stackgan import Stage2Model
import tensorflow as tf

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = Stage2Model()
    dataset = Dataset(image_size=(256,256),data_base_path="./data")
    train_data = dataset.get_test_ds()
    model.train(train_data,num_epochs=1)
