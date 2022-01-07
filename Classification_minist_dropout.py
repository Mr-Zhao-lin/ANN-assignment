import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.system('export DISPLAY=:0.0')
from tensorflow._api.v2.compat import v1 as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
tf.disable_eager_execution()

# load data start
digits = load_digits()
X = digits.data    #the picture 8*8
y = digits.target   #the number
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
# load data end

# define add_layer start
def add_layer(inputs,in_size,out_size,layer_name,activation_function=None,keep_prob=1):
    initializer=tf.random_normal_initializer(mean=0)
    Weights=tf.Variable(initializer([in_size,out_size]))
    baises=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+baises
    Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob)
    if activation_function is None:
        output=Wx_plus_b
    else:
        output=activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name+'outputs',output)
    return output

# define add_layer end

#define placeholder for input start
keep_prob=tf.placeholder(tf.float32)
x_input=tf.placeholder(tf.float32,[None,64],'x_input')
y_input=tf.placeholder(tf.float32,[None,10],'y_input')

#define placeholder for input end

# add output layer start
layer_1=add_layer(x_input,64,100,'layer_1',tf.nn.tanh,keep_prob)
prediction=add_layer(layer_1,100,10,'layer_output',tf.nn.softmax,keep_prob)   #注意这里是输出层（使用了softmax函数），在这里使用dropout层会出现问题（全为零）
# add output layer end

#the loss start
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_input*tf.log(prediction+(1e-7)),reduction_indices=1))
tf.summary.scalar('loss',cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init=tf.global_variables_initializer()
merged=tf.summary.merge_all()
#the loss end

# run the net start
sess=tf.Session()
sess.run(init)

train_writer=tf.summary.FileWriter('logs/train',sess.graph)
test_writer=tf.summary.FileWriter('logs/test',sess.graph)
for i in range(500):
    sess.run(train_step,feed_dict={x_input:X_train,y_input:y_train,keep_prob:0.5})
    if i%50 == 0:
        train_result=sess.run(merged,feed_dict={x_input:X_train,y_input:y_train,keep_prob:1})
        test_result=sess.run(merged,feed_dict={x_input:X_test,y_input:y_test,keep_prob:1})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)
        print(i)
sess.close
# run the net end 




# 