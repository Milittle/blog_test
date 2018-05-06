#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/6 14:32
# @Author  : milittle
# @Site    : www.weaf.top
# @File    : deming_lr.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()


def run():
    x_vals = np.linspace(0, 10, 100)
    y_vals = x_vals + np.random.normal(0, 1, 100)
    batch_size = 125

    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    A = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    model_output = tf.add(tf.matmul(x_data, A), b)

    # 注意这里的loss函数的求解
    demming_numerator = tf.abs(tf.subtract(tf.add(tf.matmul(x_data, A), b), y_target))
    demming_denominator = tf.sqrt(tf.add(tf.square(A), 1))
    loss = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))

    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    loss_vec = []
    for i in range(1500):
        rand_index = np.random.choice(len(x_vals), size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)
        if (i + 1) % 100 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
            print('Loss = ' + str(temp_loss))

    [W] = sess.run(A)
    [bias] = sess.run(b)

    # Get best fit line
    best_fit = []
    for i in x_vals:
        best_fit.append(W * i + bias)

    plt.plot(x_vals, y_vals, 'o', label='Data Points')
    plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
    plt.legend(loc='upper left')
    plt.title('Sepal Length vs Pedal Width')
    plt.xlabel('Pedal Width')
    plt.ylabel('Sepal Length')
    plt.show()

    # Plot loss over time
    plt.plot(loss_vec, 'k-')
    plt.title('Demming Loss per Generation')
    plt.xlabel('Iteration')
    plt.ylabel('Demming Loss')
    plt.show()

def main(_):
    run()

if __name__ == '__main__':
    tf.app.run()
