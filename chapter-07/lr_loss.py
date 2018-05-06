#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/6 14:18
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : lr_loss.py
# @Software: PyCharm


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

def run():
    ops.reset_default_graph()
    sess = tf.Session()
    batch_size = 100
    x_vals = np.linspace(0, 10, 100)
    y_vals = x_vals + np.random.normal(0, 1, 100)

    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # Create variables for linear regression
    A = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    # Declare model operations
    model_output = tf.add(tf.matmul(x_data, A), b)

    loss_l1 = tf.reduce_mean(tf.abs(y_target - model_output))

    # Declare optimizers
    my_opt_l1 = tf.train.GradientDescentOptimizer(0.0001)
    train_step_l1 = my_opt_l1.minimize(loss_l1)

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    loss_vec_l1 = []
    for i in range(1000):
        rand_index = np.random.choice(len(x_vals), size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess.run(train_step_l1, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss_l1 = sess.run(loss_l1, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec_l1.append(temp_loss_l1)
        if (i + 1) % 25 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))

    ops.reset_default_graph()

    # Create graph
    sess = tf.Session()

    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # Create variables for linear regression
    A = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    # Declare model operations
    model_output = tf.add(tf.matmul(x_data, A), b)

    loss_l2 = tf.reduce_mean(tf.square(y_target - model_output))

    # Declare optimizers
    my_opt_l2 = tf.train.GradientDescentOptimizer(0.0001)
    train_step_l2 = my_opt_l2.minimize(loss_l2)

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    loss_vec_l2 = []
    for i in range(1000):
        rand_index = np.random.choice(len(x_vals), size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess.run(train_step_l2, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss_l2 = sess.run(loss_l2, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec_l2.append(temp_loss_l2)
        if (i + 1) % 25 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))

    plt.plot(loss_vec_l1, 'k-', label='L1 Loss')
    plt.plot(loss_vec_l2, 'r--', label='L2 Loss')
    plt.title('L1 and L2 Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('L1 Loss')
    plt.legend(loc='upper right')
    plt.show()


def main(_):
    run()

if __name__ == '__main__':
    tf.app.run()