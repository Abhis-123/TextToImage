if __name__ == "__main__":
  import os 
  os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import cv2
import pickle

WORKERS = tf.data.experimental.AUTOTUNE

class Dataset():
  def __init__(self, image_size, data_base_path,batch_size = 64):
    data_dir = data_base_path+"/birds"
    train_dir = data_dir + "/train"
    test_dir = data_dir + "/test"
    train_embeddings_path = train_dir + "/char-CNN-RNN-embeddings.pickle"
    test_embeddings_path = test_dir + "/char-CNN-RNN-embeddings.pickle"
    train_filenames_path = train_dir + "/filenames.pickle"
    test_filenames_path = test_dir + "/filenames.pickle"
    cub_dataset_dir = data_base_path+"/CUB_200_2011"
    bounding_boxes_path = cub_dataset_dir + "/bounding_boxes.txt"
    image_ids_path = cub_dataset_dir + "/images.txt"
    images_path = cub_dataset_dir + "/images"
    
    self.image_width = image_size[0]
    self.image_height = image_size[1]
    self.batch_size = batch_size

    with open(train_filenames_path, 'rb') as f:
      self.train_filenames = pickle.load(f, encoding='latin1')
      self.train_filenames = [images_path+"/"+filename+".jpg" for filename in self.train_filenames]
    
    with open(test_filenames_path, 'rb') as f:
      self.test_filenames = pickle.load(f, encoding='latin1')
      self.test_filenames = [images_path+"/"+filename+".jpg" for filename in self.test_filenames]

    with open(train_embeddings_path, 'rb') as f:
      self.train_embeddings = pickle.load(f, encoding = 'latin1')

    with open(test_embeddings_path, 'rb') as f:
      self.test_embeddings = pickle.load(f, encoding = 'latin1')

    bounding_boxes = {}
    with open(bounding_boxes_path, 'rb') as f:
      box_coordinates = f.read()
      box_coordinates = box_coordinates.splitlines()
      box_coordinates = [box_coordinate.decode('utf-8') for box_coordinate in box_coordinates]
      for i in range(len(box_coordinates)):
        bounding_box = box_coordinates[i].split()
        bounding_boxes[bounding_box[0]] = [int(float(c)) for c in box_coordinates[i].split()][1:]

    image_ids_mapping = {}
    with open(image_ids_path, 'rb') as f:
      image_ids = f.read()
      image_ids = image_ids.splitlines()
      image_ids = [image_id.decode('utf-8') for image_id in image_ids]
      for i in range(len(image_ids)):
        image_id = image_ids[i].split()
        image_ids_mapping[image_id[0]] = image_id[1]

    bounding_boxes_mapping = {}
    for image_id in bounding_boxes.keys():
      bounding_boxes_mapping[images_path + "/" + image_ids_mapping[image_id]] = bounding_boxes[image_id]

    self.train_bounding_boxes = []
    self.test_bounding_boxes = []
    for i in range(len(self.train_filenames)):
      self.train_bounding_boxes.append(bounding_boxes_mapping[self.train_filenames[i]])
    for i in range(len(self.test_filenames)):
      self.test_bounding_boxes.append(bounding_boxes_mapping[self.test_filenames[i]])

  def crop(self, image, bounding_box):
    image = image.numpy()
    if bounding_box is not None:
      x, y, width, height = bounding_box
      image = image[y:(y + height), x:(x + width)]
      image = cv2.resize(image, (self.image_width, self.image_height))
    return image

  def parse_function(self, image_path, embeddings, bounding_box):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.py_function(func = self.crop, inp = [image, bounding_box], Tout = tf.float32)
    image.set_shape([self.image_width, self.image_height, 3])
    image = (image - 127.5) / 127.5

    embedding_index = np.random.randint(0, embeddings.shape[0] - 1)
    embedding = embeddings[embedding_index]
    return image, embedding
  
  def get_train_ds(self,prefetch=True):
    BUFFER_SIZE = len(self.train_filenames)
    ds = tf.data.Dataset.from_tensor_slices((self.train_filenames, self.train_embeddings, self.train_bounding_boxes))
    ds = ds.shuffle(BUFFER_SIZE)
    ds = ds.repeat()
    ds = ds.map(self.parse_function, num_parallel_calls = WORKERS)
    ds = ds.batch(self.batch_size, drop_remainder = True)
    if prefetch ==True:
      ds = ds.prefetch(1)
    return ds
  
  def get_test_ds(self, prefetch=True):
    BUFFER_SIZE = len(self.test_filenames)
    ds = tf.data.Dataset.from_tensor_slices((self.test_filenames, self.test_embeddings, self.test_bounding_boxes))
    ds = ds.shuffle(BUFFER_SIZE)
    ds = ds.repeat(1)
    ds = ds.map(self.parse_function, num_parallel_calls = WORKERS)
    ds = ds.batch(self.batch_size, drop_remainder = True)
    if prefetch== True:
      ds = ds.prefetch(1)
    return ds


if __name__ == '__main__':
    print("in dataset")
    dataset = Dataset(image_size=(64,64),data_base_path="../data") 
    train_data = dataset.get_train_ds()

    print(train_data.take(1))

