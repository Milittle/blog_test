#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/6 13:16
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : Linear_re_no_train.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.Session()


def run():
    # 构造数据
    x_vals = np.linspace(0, 10, 100)
    y_vals = x_vals + np.random.normal(0, 1, 100)

    x_vals_column = np.transpose(np.matrix(x_vals))
    ones_column = np.transpose(np.matrix(np.repeat(1, 100)))

    A = np.column_stack((x_vals_column, ones_column))

    y = np.transpose(np.matrix(y_vals))

    A_tensor = tf.constant(A)
    y_tensor = tf.constant(y)

    #利用矩阵的逆解决这个线性问题。
    t_A_A = tf.matmul(tf.transpose(A_tensor), A_tensor) #求矩阵转置和本身的乘积
    t_A_A_inverse = tf.matrix_inverse(t_A_A) # 求矩阵的逆
    product = tf.matmul(t_A_A_inverse, tf.transpose(A_tensor))
    solution = tf.matmul(product, y_tensor)

    solution_eval = sess.run(solution)

    W = solution_eval[0][0]

    bias = solution_eval[1][0]

    print('W: ' + str(W))
    print('bias: ' + str(bias))

    # Get best fit line
    best_fit = []
    for i in x_vals:
        best_fit.append(W * i + bias)
    plt.plot(x_vals, y_vals, 'o', label = 'data')
    plt.plot(x_vals, best_fit, 'r-', label = 'best fit line', linewidth = 3)
    plt.legend(loc='upper left')
    plt.show()

def main(_):
    run()

if __name__ == '__main__':
    tf.app.run()