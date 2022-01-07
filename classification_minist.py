import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.system('export DISPLAY=:0.0')
from tensorflow._api.v2.compat import v1 as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
tf.disable_eager_execution()

# load data set start
mnist = tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images, test_images = train_images/255.0, test_images/255.0  #归一化的原因：后面的偏置BAISe和weight如果和输入数据相差过大，那么调整参数难以学习

#load data set end
# deal with datas start
train_images = train_images.reshape([-1,784])
test_images = test_images.reshape([-1,784])
train_labels,test_labels=tf.Session().run(tf.one_hot(train_labels,10)),tf.Session().run(tf.one_hot(test_labels,10))
# deal with datas end


# create the simple net start

def add_layer(inputs,n_layer,in_size,out_size,activation_function):
    layer_name='layer%d'%n_layer
    with tf.name_scope(layer_name):
        Weights=tf.Variable(tf.random_normal([in_size,out_size]),name=layer_name+'/randomweights')
        tf.summary.histogram(name=layer_name+'/weights', values=Weights)
        Baises=tf.Variable(tf.zeros([1, out_size]) + 0.1,name=layer_name+'/randombaises')
        tf.summary.histogram(name=layer_name+'/baises',values=Baises)
        with tf.name_scope(layer_name+'out_unact'):
            out_unact=tf.matmul(inputs,Weights)+Baises
            tf.summary.histogram(name=layer_name+'out_unact',values=out_unact)
    if activation_function==None:
        outputs=out_unact
    else:
        outputs=activation_function(out_unact)
    return outputs
        
def cross_entropy(train_labels,predictions):
    delta=1e-7
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(train_labels * tf.log(predictions+delta),reduction_indices=[1]))
    return cross_entropy

    # add two layers start
with tf.name_scope('input'):
    xs=tf.placeholder(tf.float32,[None,784])
    ys=tf.placeholder(tf.float32,[None,10])
layer1_out=10   # the number of hiding layer 's output
layer1=add_layer(xs,1,784,layer1_out,activation_function=tf.nn.relu)
layer_predict=add_layer(layer1,2,layer1_out,10,activation_function=tf.nn.softmax)


with tf.name_scope('loss'):
    cross_entropy=cross_entropy(train_labels=ys,predictions=layer_predict)
    tf.summary.scalar('loss',cross_entropy)

    # add two layers start
train=tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)
init=tf.global_variables_initializer()

# create the simple net end

def accuracy_cong(test_image,test_labels):
    global layer_predict
    y_predict=sess.run(layer_predict,feed_dict={xs:test_image})
    correct_prediction=tf.equal(np.argmax(y_predict,1),np.argmax(test_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.name_scope('accuracy'):

        accuracy_conginize=sess.run(accuracy,feed_dict={xs:test_image,ys:test_labels})
    return accuracy_conginize

# define session and run start
sess=tf.Session()
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)
i=0
while 1:
    i=i+1
    num_all=train_images.shape[0]
    index=np.random.choice(range(num_all),100)
    batch_x,batch_y = train_images[index,] ,train_labels[index,]
    sess.run(train,feed_dict={xs:batch_x,ys:batch_y})
    if i%50 ==0:
        acc=accuracy_cong(test_images,test_labels)
        rs=sess.run(merged,feed_dict={xs:batch_x,ys:batch_y})
        writer.add_summary(rs,i)
        print(i,acc)
        if acc>=0.95:
            break


sess.close()
# define session and run end

