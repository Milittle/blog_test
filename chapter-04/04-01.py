# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 17:05:40 2018

@author: milittle
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()

# 创建数据为了feed
my_array = np.array([[1., 3., 5., 7., 9.],
                    [-2., 0., 2., 4., 6.],
                    [-6., -3., 0., 3., 6.]])

# 复制
x_vals = np.array([my_array, my_array + 1])

# 声明placeholder
x_data = tf.placeholder(tf.float32, shape = [3, 5])

# 声明常数来操作
m1 = tf.constant([[1.], [0.], [-1.], [2.], [4]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# 声明操作
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.matmul(prod2, a1)

# 打印验证结果
for x_val in x_vals:
    print(sess.run(add1, feed_dict = {x_data: x_val}))
    
merged = tf.summary.merge_all(key = 'summaries')
if not os.path.exists('tensorboard_logs/'):
    os.makedirs('tensorflow_logs/')
my_writer = tf.summary.FileWriter('tensorboard_logs/', sess.graph)