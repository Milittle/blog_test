# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 15:16:23 2018

@author: milittle
"""

# 导入一些必要的库
import numpy as np # 数学计算库
import matplotlib.pyplot as plt # 画图的一个库
import tensorflow as tf # TensorFlow的库

from tensorflow.examples.tutorials.mnist import input_data
fashion_mnist = input_data.read_data_sets('input/data', one_hot = True)

# 打印数据瞧一瞧
print('Fashion_mnist:{type}'.format(type = type(fashion_mnist)))


# 打印以下训练数据的shape瞧一瞧
print('Fashion Mnist:\n')
print("Training set (images) shape: {shape}".format(shape=fashion_mnist.train.images.shape))
print("Training set (labels) shape: {shape}".format(shape=fashion_mnist.train.labels.shape))
print("Test set (images) shape: {shape}".format(shape=fashion_mnist.test.images.shape))
print("Test set (labels) shape: {shape}".format(shape=fashion_mnist.test.labels.shape))

print(type(fashion_mnist.train.images))
print(fashion_mnist.train.images[0].shape)
print(fashion_mnist.train.labels[0])

# 由于fashion mnist是十种，所以label就是通过一个one_hot类型的数组来存储的[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 所以我们定义一个对一个的字典，好对应他们的类别属性

label_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# 获取随机的数据和它的label
sample_1 = fashion_mnist.train.images[47].reshape(28,28)
sample_label_1 = np.where(fashion_mnist.train.labels[47] == 1)[0][0]

sample_2 = fashion_mnist.train.images[23].reshape(28,28)
sample_label_2 = np.where(fashion_mnist.train.labels[23] == 1)[0][0]

# 用matplot画出这个image和label
print("y = {label_index} ({label})".format(label_index=sample_label_1, label=label_dict[sample_label_1]))
plt.imshow(sample_1, cmap='Greys')
plt.show()

print("y = {label_index} ({label})".format(label_index=sample_label_2, label=label_dict[sample_label_2]))
plt.imshow(sample_2, cmap='Greys')
plt.show()

# 接下来就是设计网络的参数了
n_hidden_1 = 128 # 第一个隐藏层的单元个数
n_hidden_2 = 128 # 第二个隐藏层的单元个数
n_input = 784 # fashion mnist输入图片的维度（单元个数） (图片大小: 28*28)
n_classes = 10 # fashion mnist的种类数目 (0-9 数字)
n_samples = fashion_mnist.train.num_examples


# 创建 placeholders
def create_placeholders(n_x, n_y):
    """
    为sess创建一个占位对象。
    
    参数:
    n_x -- 向量, 图片大小 (28*28 = 784)
    n_y -- 向量, 种类数目 (从 0 到 9, 所以是 -> 10种)
    
    返回参数:
    X -- 为输入图片大小的placeholder shape是[784, None] None在这里表示随便多少
    Y -- 为输出种类大小的placeholder shape是[10, None]
    """
    
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    
    return X, Y


# 测试上面的create_placeholders()
X, Y = create_placeholders(n_input, n_classes)

print("Shape of X: {shape}".format(shape=X.shape))
print("Shape of Y: {shape}".format(shape=Y.shape))

# 定义初始化参数参数
def initialize_parameters():
    """
    参数初始化，下面是每个参数的shape，总共有三层
                        W1 : [n_hidden_1, n_input]
                        b1 : [n_hidden_1, 1]
                        W2 : [n_hidden_2, n_hidden_1]
                        b2 : [n_hidden_2, 1]
                        W3 : [n_classes, n_hidden_2]
                        b3 : [n_classes, 1]
    
    Returns:
    包含所有权重和偏置项的dic
    """
    
    # 设置随机数种子
    tf.set_random_seed(42)
    
    # 为每一层的权重和偏置项进行初始化工作
    W1 = tf.get_variable("W1", [n_hidden_1, n_input], initializer = tf.contrib.layers.xavier_initializer(seed = 42))
    b1 = tf.get_variable("b1", [n_hidden_1, 1], initializer = tf.zeros_initializer())
    
    W2 = tf.get_variable("W2", [n_hidden_2, n_hidden_1], initializer = tf.contrib.layers.xavier_initializer(seed = 42))
    b2 = tf.get_variable("b2", [n_hidden_2, 1], initializer = tf.zeros_initializer())
    
    W3 = tf.get_variable("W3", [n_classes, n_hidden_2], initializer=tf.contrib.layers.xavier_initializer(seed = 42))
    b3 = tf.get_variable("b3", [n_classes, 1], initializer = tf.zeros_initializer())
    
    # Store initializations as a dictionary of parameters
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }
    
    return parameters

# 测试初始化参数

tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = {w1}".format(w1=parameters["W1"]))
    print("b1 = {b1}".format(b1=parameters["b1"]))
    print("W2 = {w2}".format(w2=parameters["W2"]))
    print("b2 = {b2}".format(b2=parameters["b2"]))
    

# 前向传播算法（就是神经网络的前向步骤）
def forward_propagation(X, parameters):
    """
    实现前向传播的模型 LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    上面的显示就是三个线性层，每一层结束以后，实现relu的作用，实现非线性功能，最后三层以后用softmax实现分类
    
    Arguments:
    X -- 输入训练数据的个数[784, n] 这里的n代表可以一次训练多个数据
    parameters -- 包括上面所有的定义参数三个网络中的权重W和偏置项B

    Returns:
    Z3 -- 最后的一个线性单元输出
    """
    
    # 从参数dict里面取到所有的参数
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    # 前向传播过程
    Z1 = tf.add(tf.matmul(W1,X), b1)     # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1), b2)    # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2), b3)    # Z3 = np.dot(W3,Z2) + b3
    
    return Z3


# 测试前向传播喊出
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(n_input, n_classes)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = {final_Z}".format(final_Z=Z3))

# 定义计算损失函数
# 是计算loss的时候了
def compute_cost(Z3, Y):
    """
    计算cost
    
    参数:
    Z3 -- 前向传播的最终输出（[10, n]）n也是你输入的训练数据个数
    Y -- 
    
    返回:
    cost - 损失函数 张量（Tensor）
    """
    
    # 获得预测和准确的label
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    # 计算损失
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    
    return cost

# 测试计算损失函数
tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(n_input, n_classes)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = {cost}".format(cost=cost))
    
# 这个就是关键了，因为每一层的参数都是通过反向传播来实现权重和偏置项参数更新的
# 总体的原理就是经过前向传播，计算到最后的层，利用softmax加交叉熵，算出网络的损失函数
# 然后对损失函数进行求偏导，利用反向传播算法实现每一层的权重和偏置项的更新
def model(train, test, learning_rate=0.0001, num_epochs=16, minibatch_size=32, print_cost=True, graph_filename='costs'):
    """
    实现了一个三层的网络结构: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    参数:
    train -- 训练集
    test -- 测试集
    learning_rate -- 优化权重时候所用到的学习率
    num_epochs -- 训练网络的轮次
    minibatch_size -- 每一次送进网络训练的数据个数（也就是其他函数里面那个n参数）
    print_cost -- 每一轮结束以后的损失函数
    
    返回:
    parameters -- 被用来学习的参数
    """
    
    # 确保参数不被覆盖重写
    tf.reset_default_graph()
    tf.set_random_seed(42)
    seed = 42
    # 获取输入和输出大小
    (n_x, m) = train.images.T.shape
    n_y = train.labels.T.shape[0]
    
    costs = []
    
    # 创建输入输出数据的占位符
    X, Y = create_placeholders(n_x, n_y)
    # 初始化参数
    parameters = initialize_parameters()
    
    # 进行前向传播
    Z3 = forward_propagation(X, parameters)
    # 计算损失函数
    cost = compute_cost(Z3, Y)
    # 使用AdamOptimizer优化器实现反向传播算法（最小化cost）
    # 其实我们这个地方的反向更新参数的过程都是tensorflow给做了
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # 变量初始化器
    init = tf.global_variables_initializer()
    
    # 开始tensorflow的sess 来计算tensorflow构建好的图
    with tf.Session() as sess:
        
        # 这个就是之前说过的要进行初始化的
        sess.run(init)
        
        # 训练轮次
        for epoch in range(num_epochs):
            
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            
            for i in range(num_minibatches):
                
                # 获取下一个batch的训练数据和label数据
                minibatch_X, minibatch_Y = train.next_batch(minibatch_size)
                
                # 执行优化器
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X.T, Y: minibatch_Y.T})
                
                # 更新每一轮的损失
                epoch_cost += minibatch_cost / num_minibatches
                
            # 打印每一轮的损失
            if print_cost == True:
                print("Cost after epoch {epoch_num}: {cost}".format(epoch_num=epoch, cost=epoch_cost))
                costs.append(epoch_cost)
        
        # 使用matplot画出损失的变化曲线图
        plt.figure(figsize=(16,5))
        plt.plot(np.squeeze(costs), color='#2A688B')
        plt.xlim(0, num_epochs-1)
        plt.ylabel("cost")
        plt.xlabel("iterations")
        plt.title("learning rate = {rate}".format(rate=learning_rate))
        plt.savefig(graph_filename, dpi = 300)
        plt.show()
        
        # 保存参数
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        
        # 计算预测准率
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        
        # 计算测试准率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        print ("Train Accuracy:", accuracy.eval({X: train.images.T, Y: train.labels.T}))
        print ("Test Accuracy:", accuracy.eval({X: test.images.T, Y: test.labels.T}))
        
        return parameters

# 要开始训练我们的fashion mnist网络了
train = fashion_mnist.train
test = fashion_mnist.test

parameters = model(train, test, learning_rate = 0.001, num_epochs = 16, graph_filename = 'fashion_mnist_costs')

