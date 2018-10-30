#https://www.2cto.com/kf/201805/746214.html
# from __future__ import print_function
# from __future__ import division
# from __future__ import absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 默认的显示等级，显示所有信息修改为 只显示 warning 和 Error
# import os
import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data

LOGDIR = "D:\logs"
def conv_layer(input, size_in, size_out, name="conv"):
    # define the convolution layer
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="w")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        # 生成【变量】的监控信息，并将生成的监控信息写入【日志文件】
        #tf.summary.histogram(name,var)# # [2]name:给出了可视化结果中显示的图表名称[1]var :需要【监控】和【记录】运行状态的【张量】
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

def fc_layer(input, size_in, size_out, name="fc"):
    # define the fully-connected layer
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="w")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
        act = tf.matmul(input, w) + b
        # 生成【变量】的监控信息，并将生成的监控信息写入【日志文件】
        # tf.summary.histogram(name,var)# # [2]name:给出了可视化结果中显示的图表名称[1]var :需要【监控】和【记录】运行状态的【张量】
        tf.summary.histogram("weights", w) #[1]用来显示直方图信息
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activation", act)

        # tf.summary.histogram(name, var)
        # mean = tf.reduce_mean(var)
        # tf.summary.scalar('mean/' + name, mean)
        # stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev/' + name, stddev)

        return act
#       [1]将【计算图】中的【数据的分布/数据直方图】写入TensorFlow中的【日志文件】，以便为将来tensorboard的可视化做准备
#参数说明：
#       [1]name  :一个节点的名字，如下图红色矩形框所示
#       [2]values:要可视化的数据，可以是任意形状和大小的数据
def mnist():
    sess = tf.Session()
    # Setup placeholders, and reshape the data
    with tf.name_scope("input"):
        # 【2】定义两个【占位符】，作为【训练样本图片/此块样本作为特征向量存在】和【类别标签】的输入变量，并将这些占位符存在命名空间input中
        x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
        #并通过tf.summary.image函数定义将当前图片信息作为写入日志的操作
        tf.summary.image("image", x_image, max_outputs=3)
    #def conv_layer(input, size_in, size_out, name="conv"):
    conv1 = conv_layer(x_image, 1, 32, name="conv1")

    #tf.nn.max_pool(value, ksize, strides, padding, name=None)
    conv1_pool = tf.nn.max_pool(conv1, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    #def conv_layer(input, size_in, size_out, name="conv"):
    conv2 = conv_layer(conv1_pool, 32, 64, name="conv2")

    #tf.nn.max_pool(value, ksize, strides, padding, name=None)
    conv_out = tf.nn.max_pool(conv2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    #tf.reshape(tensor,shape, name=None) 函数的作用是将tensor变换为参数shape的形式
    # 其中shape为一个列表形式，特殊的一点是列表中可以存在 - 1。
    # -1 代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个 - 1。（当然如果存在多个 - 1，就是一个存在多解的方程了）
    #https://blog.csdn.net/zeuseign/article/details/72742559
    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    # def fc_layer(input, size_in, size_out, name="fc"):
    fc1 = tf.nn.relu(fc_layer(flattened, 7 * 7 * 64, 1024, name="fc1"))
    # def fc_layer(input, size_in, size_out, name="fc"):
    logits = fc_layer(fc1, 1024, 10, name="logits")

    with tf.name_scope("cross_entropy"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y), name="xent")
        tf.summary.scalar("loss", xent) #一般在画loss曲线和accuary曲线时会用到这个函数

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy) #一般在画loss曲线和accuary曲线时会用到这个函数
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

    mnist = input_data.read_data_sets(train_dir="MNIST_data/", one_hot=True)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    file_writer = tf.summary.FileWriter(LOGDIR)
    file_writer.add_graph(sess.graph)
    merge_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    for i in range(1001):
        batch = mnist.train.next_batch(100)
        if i % 5 == 0:
            [summaries, _] = sess.run([merge_op, accuracy], feed_dict={x: batch[0], y: batch[1]})
            file_writer.add_summary(summaries, i)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
            print("step: %d accuracy: %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

if __name__ == "__main__":
    mnist()

#========================================================================================================
#函数原型:
#       def scalar(name, tensor, collections=None, family=None)
#函数说明：
#       [1]输出一个含有标量值的Summary protocol buffer，这是一种能够被tensorboard模块解析的【结构化数据格式】
#       [2]用来显示标量信息
#       [3]用来可视化标量信息
#       [4]其实，tensorflow中的所有summmary操作都是对计算图中的某个tensor产生的单个summary protocol buffer，而
#          summary protocol buffer又是一种能够被tensorboard解析并进行可视化的结构化数据格式
#       虽然，上面的四种解释可能比较正规，但是我感觉理解起来不太好，所以，我将tf.summary.scalar()函数的功能理解为：
#       [1]将【计算图】中的【标量数据】写入TensorFlow中的【日志文件】，以便为将来tensorboard的可视化做准备
#参数说明：
#       [1]name  :一个节点的名字，如下图红色矩形框所示
#       [2]tensor:要可视化的数据、张量
#主要用途：
#       一般在画loss曲线和accuary曲线时会用到这个函数。
#=======================================================================================================




