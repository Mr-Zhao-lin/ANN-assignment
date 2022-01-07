import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.system('export DISPLAY=:0.0')
from tensorflow._api.v2.compat import v1 as tf
tf.disable_eager_execution()
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# load and preprocess data
mnist = tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images, test_images = train_images/255.0,test_images/255.0  #归一化的原因：后面的偏置BAISe和weight如果和输入数据相差过大，那么调整参数难以学习
train_images = train_images.reshape([-1,784])
test_images = test_images.reshape([-1,784])
train_labels,test_labels=tf.Session().run(tf.one_hot(train_labels,10)),tf.Session().run(tf.one_hot(test_labels,10))


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = prediction
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 784])   # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
with tf.name_scope('convlayer_1'):
    W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
with tf.name_scope('pooling1'):
    h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
with tf.name_scope('convlayer_2'):
    W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
with tf.name_scope('pooling2'):
    h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
with tf.name_scope('funclayer_1'):
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
with tf.name_scope('funclayer_2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                reduction_indices=[1]))       # loss
    tf.summary.scalar('loss',cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# accuracy start

with tf.name_scope('accuracy'):
    accuracy_conginize=compute_accuracy(v_xs=xs, v_ys=ys)
    tf.summary.scalar('accuracy',accuracy_conginize)
# accuracy end

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12

init=tf.global_variables_initializer()
saver = tf.train.Saver()
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)

for i in range(1500):
    num_all=train_images.shape[0]
    index=np.random.choice(range(num_all),100)
    batch_xs,batch_ys = train_images[index,] ,train_labels[index,]
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        acc=sess.run(accuracy_conginize,feed_dict={xs:test_images,ys:test_labels,keep_prob: 1.0})
        rs=sess.run(merged,feed_dict={xs:test_images,ys:test_labels,keep_prob: 1.0})
        writer.add_summary(rs,i)
        print(acc)
saver.save(sess,'saved_net/Classification_cnn_drop')
sess.close()