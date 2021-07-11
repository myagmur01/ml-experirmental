#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import cv2
import time
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[11]:


def load_data_using_keras(folders):
    
    """
    Loading data using ImageDataGenerator
    
    Returns: Data generator
    """
    
    image_generator = {}
    data_generator = {}
    
    for x in folders:
        image_generator[x] = ImageDataGenerator(rescale=1/255.)
        
        shuffle_images = True if x == 'train' else False
        
        data_generator[x] = image_generator[x].flow_from_directory(
        
            batch_size = batch_size,
            directory = os.path.join(dir_path, x),
            shuffle = shuffle_images,
            target_size = (img_dims[0], img_dims[1]),
            class_mode = 'categorical'
        )
        
        return data_generator
    
    
def load_data_using_tfdata(folders):
    
    """
    Loading data using tfdata
    
    Returns: Data generator
    """
    
    def parse_image(file_path):
        
        parts = tf.strings.split(file_path, os.path.sep)
        class_names = np.array(os.listdir(dir_path + '/train'))
        
        label = parts[-2] == class_names
        
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, float32)
        img = tf.image.resize(img, [img_dims[0], img_dims[1]])
        
        return img, label
    
    def prepare_for_training(ds, cache=True, shuffle_buffer=1000):
        
        if cache:
            if isinstance(cache, str):
                ds  = ds.cache(cache)
            else:
                ds = ds.cache()
                
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        
        ds = ds.prefetch(buffer_size = AUTOTUNE)
        
        return ds
    
    data_generator = {}
    for x in folders:
        dir_extend = dir_path + '/' + x
        list_ds = tf.data.Dataset.list_files(str(dir_extend + '/*/*'))
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        
        labeled_ds = list_ds.map(
        parse_image, num_parallel_calls=AUTOTUNE)
        
        data_generator[x] = prepare_for_training(
        labeled_ds, cache='./data.tfcache')
        
        return data_generator
    
    
if __name__ == '__main__':

    # Need to change this w.r.t data
    dir_path = '~/Desktop/Work/DevWorks/CloudRecog-API/CloudCls/datasets/TrainingSets/WNI-App-v3'
    folders = ['train', 'val','test']
    load_data_using = 'tfdata'

    batch_size = 32
    img_dims = [256, 256]

    if load_data_using == 'keras':
        data_generator = load_data_using_keras(folders)
    elif load_data_using == 'tfdata':
        data_generator = load_data_using_tfdata(folders)
        


# In[ ]:




