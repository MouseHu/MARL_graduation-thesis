import tensorflow as tf
import numpy as np
from copy import copy
from create_model import *


#终于跑通..
input_dim = 64
input_x = tf.placeholder(dtype=tf.float32, shape=[1, input_dim, input_dim, 3])
input_y = tf.placeholder(dtype=tf.float32, shape=[1, input_dim, input_dim, 3])
#out = tf_model(input_x, 20, 0.5, reuse=False,
#                            is_training=True)
out = dqn_model(input_x, 10,0)
target = dqn_model(input_y, 10,1)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    # 2. 运行init operation
    sess.run(init_op)
    # 计算
    #a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    #a_out = sess.run(out, feed_dict={input_x: np.random.randn(1, input_dim, input_dim, 3)})
    # batch, width, height, channel
    x = np.random.randn(1, input_dim, input_dim, 3)
    result1 = sess.run(out, feed_dict={input_x: x})
    print(result1)
    saver = tf.train.Saver()
    saver.save(sess, 'my_test_model')
    # result2 = sess.run(target, feed_dict={input_y: x})
    # print(result2)
    # out = copy(target)
    # result3 = sess.run(out, feed_dict={input_y: x})
    #
    # print(result3)