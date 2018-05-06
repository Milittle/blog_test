#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/6 13:51
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : CholeskyMethos_lr.py
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

    tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
    L = tf.cholesky(tA_A)

    tA_y = tf.matmul(tf.transpose(A_tensor), y)
    sol1 = tf.matrix_solve(L, tA_y)


    sol2 = tf.matrix_solve(tf.transpose(L), sol1)
    solution_eval = sess.run(sol2)

    W = solution_eval[0][0]
    bias = solution_eval[1][0]

    print('slope: ' + str(W))
    print('y_intercept: ' + str(bias))


    best_fit = []
    for i in x_vals:
        best_fit.append(W * i + bias)
    plt.plot(x_vals, y_vals, 'o', label='Data')
    plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
    plt.legend(loc='upper left')
    plt.show()

def main(_):
    run()

if __name__ == '__main__':
    tf.app.run()