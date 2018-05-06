# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:49:59 2018

@author: milittle
"""

import tensorflow as tf
sess = tf.Session()
init = tf.global_variables_initializer() # 此处的init是全局变量初始化器，TensorFlow的session必须执行这个初始化器才能执行前面建立好的图，所以，这个是很重要的一点，后续也会强调
sess.run(init)
hello = tf.constant('hello world')
print(sess.run(hello))