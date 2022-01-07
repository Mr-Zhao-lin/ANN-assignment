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
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

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

def add_layer(inputs,in_size,out_size,activation_function=None,keep_prob=1):
    Weights=weight_variable([in_size,out_size])
    baises=bias_variable([out_size])
    Wx_plus_b=tf.matmul(inputs,Weights)+baises
    Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob)
    if activation_function is None:
        output=Wx_plus_b
    else:
        output=activation_function(Wx_plus_b)
    #tf.summary.histogram(layer_name+'outputs',output)
    return output

def addconvlayer(inputs,conv_patch,in_size,out_size,activation_function=None,):
    W_conv=weight_variable([conv_patch[0],conv_patch[1],in_size,out_size])
    b_conv=bias_variable([out_size])
    Wx_plus_b=conv2d(inputs,W_conv)+b_conv
    if activation_function is None:
        output=Wx_plus_b
    else:
        output=activation_function(Wx_plus_b)
    h_pool=max_pool_2x2(output)
    return h_pool 

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
#xs = tf.placeholder(tf.float32, [None, 784])/255. # 这样写是不行的，此时的xs不是placeholder而是truediv（经过了类型转换），且此问题不报错。
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])


## conv1 layer ##
conv1=addconvlayer(x_image,[5,5],1,32,activation_function=tf.nn.relu)
## conv2 layer ##
conv2=addconvlayer(conv1,[5,5],32,64,activation_function=tf.nn.relu)             

# 错误示例：tf1中定义的是静态图，即层的大小不会在运行中改变（定义）
#conv2_shape=tf.shape(conv2)
#num_conv2=conv2_shape[0]*conv2_shape[1]*conv2_shape[2]
layer1=add_layer(tf.reshape(conv2,[-1,7*7*64]),7*7*64,1024,activation_function=tf.nn.relu)
## func2 layer ##
prediction=add_layer(layer1,1024,10,activation_function= tf.nn.softmax)

# the error between prediction and real data

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)




sess = tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    num_all=train_images.shape[0]
    index=np.random.choice(range(num_all),100)
    batch_xs,batch_ys = train_images[index,] ,train_labels[index,]
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(test_images,test_labels))


# define addconvlayer and addlayer start


# define addconvlayer and addlayer end



""" cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) """