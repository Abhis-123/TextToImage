


import os 

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from utils.dataset import Dataset
from models.stackgan import Stage1Model
import tensorflow as tf

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = Stage1Model()
    dataset = Dataset(image_size=(64,64),data_base_path="./data",batch_size=64)
    test_data = dataset.get_train_ds()
    #model.load_weights("./weights/weights_10")
    #model.train(train_data,num_epochs=50)
    # batch_iter = iter(train_data)
    # print(len(dataset.train_filenames))
    import time
    # t1 =    time.time()
    # d = iter(test_data)
    # for i in range():
    #     t,e = next(d)
    #     shape = t.shape
    # t2 = time.time()
    # print(f"time for prefetch data {t2- t1}")
    
    test_data= dataset.get_train_ds(prefetch=False)
    t1 =    time.time()
    i=0
    for t,e in test_data:
        shape = t.shape
        
        i=i+1
        print(f"index {i}")
    t2 = time.time()
    print(f"time without prefetch data {t2- t1}   ;[; {i}")



