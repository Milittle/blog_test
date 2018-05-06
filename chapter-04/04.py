# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:50:06 2018

@author: milittle
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

# 创建的数据是要喂给下面的placeholder的
x_vals = np.array([1., 3., 5., 7., 9.])

# 创建placeholder
x_data = tf.placeholder(tf.float32)

# 创建一个乘数
m = tf.constant(3.)

# 乘法
prod = tf.multiply(x_data, m)
for x_val in x_vals:
    print(sess.run(prod, feed_dict = {x_data: x_val}))

merged = tf.summary.merge_all(key = 'summary')
if not os.path.exists('tensorboard_logs/'):
    os.makedirs('tensorboard_logs/')

my_writer = tf.summary.FileWriter('./tensorboard_logs/', sess.graph)