# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 17:23:54 2018

@author: milittle
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()

x_shape = [1, 4, 4, 1]
# 定义一个4 * 4 大小的随机矩阵
x_val = np.random.uniform(size = x_shape)

x_data = tf.placeholder(tf.float32, shape = x_shape)

# 定义一个空间移动窗口，也就是卷积操作的卷积核
# 大小是2 * 2， 步长是 2
# filter的值是一个固定的值0.25
my_filter = tf.constant(0.25, shape = [2, 2, 1, 1])
my_strides = [1, 2, 2, 1]
mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding = 'SAME', name = 'Moving_Avg_Window')

# 第二层
def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape = [2, 2])
    print((input_matrix.shape))
    output = tf.add(tf.matmul(A, input_matrix_sqeezed), b)
    return tf.nn.relu(output)
with tf.name_scope('custom_layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)

# 运行结果
print(sess.run(mov_avg_layer, feed_dict = {x_data: x_val}))

print(sess.run(custom_layer1, feed_dict = {x_data: x_val}))

merged = tf.summary.merge_all(key = 'summaries')

if not os.path.exists('tensorboard_logs/'):
    os.makedirs('tensorboard_logs/')
my_writer = tf.summary.FileWriter('tensorboard_logs', sess.graph)