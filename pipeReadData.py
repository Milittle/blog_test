# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 09:22:18 2018

@author: milittle
"""

import tensorflow as tf
import numpy as np

def generate_data():
    num = 25
    label = np.asarray(range(0, num))
    images = np.random.random([num, 5])
    images1 = tf.identity(images)
    print('label size :{}, image size {}'.format(label.shape, images.shape))
    return label, images, images1

def get_batch_data():
    label, images, images1 = generate_data()
    images = tf.cast(images, tf.float32)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([images, images1, label], num_epochs = None, shuffle = True)
    image_batch, image_batch1, label_batch = tf.train.batch(input_queue, batch_size = 128, num_threads = 1, capacity = 64)
    return image_batch, image_batch1, label_batch

image_batch, image_batch1, label_batch = get_batch_data()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    i = 0
    try:
       for i in range(200):
           image_batch_v, image_batch_v1, label_batch_v = sess.run([image_batch, image_batch1, label_batch])
           i += 1
           print(image_batch_v, image_batch_v1, label_batch_v)
           print("label")
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    # coord.join(threads)