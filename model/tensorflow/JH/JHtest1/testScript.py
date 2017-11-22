import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 100
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.0001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 40000
step_display = 1000
step_save = 10000
path_save = 'alexnet_bn'
start_from = 'alexnet_bn-40000' # HYUNRYONG: Point this to trained model


# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# define initialization
init = tf.global_variables_initializer()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

########
# TESTING CODE

opt_data_test = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../../../data/test.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_test = DataLoaderDisk(**opt_data_test)

# Launch the graph
with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)

    step = 0

    # Evaluate on training set
    predictionFile = open('pred.txt', 'w')
    testFile = open('../../../../data/test.txt', 'r')
    num_batch = loader_test.size()//batch_size
    loader_test.reset();
    for i in range(num_batch):
        image_batch, labels_batch = loader_test.next_batch(batch_size)
        top5_pred = tf.nn.top_k(logits, 5)
        result = sess.run([top5_pred], feed_dict = {x: image_batch, keep_dropout: 1., train_phase: False})
        indices = result[0].indices
        for values in indices:
            filename = testFile.readline()
            filename = filename.rstrip().split(' ')[0]
            for v in values:
                predictionFile.write("{} ".format(v))

            predictionFile.write("\n")
