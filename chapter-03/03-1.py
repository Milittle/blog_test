# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:28:30 2018

@author: milittle
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.Session()
# 定义一个placeholder
x = tf.placeholder(tf.float32, shape = (4, 4))

# 随机生成4 * 4的矩阵
rand_array = np.random.rand(4, 4)
y = tf.identity(x)
print(sess.run(y, feed_dict={x: rand_array}))

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./tmp/variable_logs", sess.graph)