#use tensorboard
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.system('export DISPLAY=:0.0')
#默认为0：输出所有log信息
#设置为1：进一步屏蔽INFO信息
#设置为2：进一步屏蔽WARNING信息
#设置为3：进一步屏蔽ERROR信息
from tensorflow._api.v2.compat import v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_eager_execution()

def add_layer(inputs, n_layer,in_size,out_size,activation_function=None):
    layer_name='layer%d'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram(layer_name +'/weights',Weights)
        with tf.name_scope('biases'):
            Biases=tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram(layer_name+'/baises',Biases)
        with tf.name_scope('Wx_plus_B'):
            Wx_plus_B=tf.matmul(inputs,Weights)+Biases
        if activation_function is None:
            outputs = Wx_plus_B
        else :
            outputs = activation_function(Wx_plus_B)
            
        tf.summary.histogram(layer_name+'/output',outputs)
    return outputs 


x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data=np.square(x_data)-0.5+noise
with tf.name_scope('input'):
    x_step=tf.placeholder(tf.float32, [None, 1],name='x_input')
    y_step=tf.placeholder(tf.float32, [None, 1],name='y_input')


layer1=add_layer(x_step,1,1,10,activation_function=tf.nn.relu)
layer2_prediction=add_layer(layer1,2,10,1,activation_function=None)
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(y_step-layer2_prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()
sess=tf.Session()
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={x_step:x_data,y_step:y_data})
    if i%50==0:
        rs=sess.run(merged,feed_dict={x_step:x_data,y_step:y_data})
        writer.add_summary(rs,i)
        print(i,sess.run(loss,feed_dict={x_step:x_data,y_step:y_data}))
        prediction_value=sess.run(layer2_prediction,feed_dict={x_step:x_data}) 
lines=plt.plot(x_data, prediction_value, 'r-', lw=1)
plt.pause(1000) 