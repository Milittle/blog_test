#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/6 14:46
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : LASSO_Ridge_lr.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()


regression_type = 'LASSO'


def run():
    sess = tf.Session()
    x_vals = np.linspace(0, 10, 100)
    y_vals = x_vals + np.random.normal(0, 1, 100)
    batch_size = 125

    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    seed = 13
    np.random.seed(seed)
    tf.set_random_seed(seed)

    A = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    model_output = tf.add(tf.matmul(x_data, A), b)



    if regression_type == 'LASSO':
        lasso_param = tf.constant(0.9)
        heavyside_step = tf.truediv(1., tf.add(1., tf.exp(tf.multiply(-50., tf.subtract(A, lasso_param)))))
        regularization_param = tf.multiply(heavyside_step, 99.)
        loss = tf.add(tf.reduce_mean(tf.square(y_target - model_output)), regularization_param)

    elif regression_type == 'Ridge':
        ridge_param = tf.constant(1.)
        ridge_loss = tf.reduce_mean(tf.square(A))
        loss = tf.expand_dims(
            tf.add(tf.reduce_mean(tf.square(y_target - model_output)), tf.multiply(ridge_param, ridge_loss)), 0)

    else:
        print('Invalid regression_type parameter value', file=sys.stderr)



    my_opt = tf.train.GradientDescentOptimizer(0.001)
    train_step = my_opt.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    # Training loop
    loss_vec = []
    for i in range(1500):
        rand_index = np.random.choice(len(x_vals), size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss[0])
        if (i + 1) % 300 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
            print('Loss = ' + str(temp_loss))
            print('\n')
    [W] = sess.run(A)
    [bias] = sess.run(b)

    # Get best fit line
    best_fit = []
    for i in x_vals:
        best_fit.append(W * i + bias)
    # Plot the result
    plt.plot(x_vals, y_vals, 'o', label='Data Points')
    plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
    plt.legend(loc='upper left')
    plt.title('Sepal Length vs Pedal Width')
    plt.xlabel('Pedal Width')
    plt.ylabel('Sepal Length')
    plt.show()

    # Plot loss over time
    plt.plot(loss_vec, 'k-')
    plt.title(regression_type + ' Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.show()

def main(_):
    run()

if __name__ == '__main__':
    tf.app.run()