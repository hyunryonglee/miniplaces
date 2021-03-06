import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 30
load_size_for_alex = 256
load_size_for_CJ = 128
fine_size_for_CJ = 116
fine_size_for_alex= 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.0001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 40000
step_display = 1000
step_save = 10000
path_save = 'test2'
start_from = [] #'alexnet_bn-40000'

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)

def alexnet(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),

        'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(100))
    }

    # Conv + ReLU + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU  + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)

    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])

    return out

def CJNet1(x, keep_dropout,train_phase):
    weights={
        'wc1': tf.Variable(tf.random_normal([8,8,3,64],stddev=np.sqrt(2./(8*8*3)))),
        'wc2': tf.Variable(tf.random_normal([5,5,64,192],stddev=np.sqrt(2./(5*5*64)))),
        'wc3': tf.Variable(tf.random_normal([3,3,192,256],stddev=np.sqrt(2./(3*3*192)))),
        'wc4': tf.Variable(tf.random_normal([3,3,256,192],stddev=np.sqrt(2./(3*3*256)))),
        'wc5': tf.Variable(tf.random_normal([3,3,192,192],stddev=np.sqrt(2./(3*3*192)))),

        'wf6': tf.Variable(tf.random_normal([8*8*192, 2048], stddev=np.sqrt(2./(7*7*192)))),
        'wf7': tf.Variable(tf.random_normal([2048, 2048], stddev=np.sqrt(2./2048))),
        'wo': tf.Variable(tf.random_normal([2048,100],stddev=np.sqrt(2./2048)))
     }

    biases={
        'bo': tf.Variable(tf.ones(100))
     }

    #conv 116 -> 58 -> 29
    conv1 = tf.nn.conv2d(x,weights['wc1'],strides=[1, 2, 2, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1,train_phase,'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

    #29 -> 14
    conv2 = tf.nn.conv2d(pool1,weights['wc2'],strides=[1, 1, 1, 1],padding='SAME')
    conv2 = batch_norm_layer(conv2,train_phase,'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    #14 -> 14
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    #14 -> 14
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    #14 -> 7
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv4, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
     
    # FC1
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)

    # FC2
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # OUtput
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])

    return out

# Construct dataloader
#opt_data_train_alex = {
    #'data_h5': 'miniplaces_256_train.h5',
#    'data_root': '../../../../data/images/',   # MODIFY PATH ACCORDINGLY
#    'data_list': '../../../../data/train.txt', # MODIFY PATH ACCORDINGLY
#    'load_size': load_size_for_alex,
#    'fine_size': fine_size_for_alex,
#    'data_mean': data_mean,
#    'randomize': True
#    }
opt_data_train_CJ = {
        #'data_h5': 'miniplaces_256_train.h5',
        'data_root': '../../../../data/images/',   # MODIFY PATH ACCORDINGLY
        'data_list': '../../../../data/train.txt', # MODIFY PATH ACCORDINGLY
        'load_size': load_size_for_CJ,
        'fine_size': fine_size_for_CJ,
        'data_mean': data_mean,
        'randomize': True
        }

#opt_data_val_alex = {
    #'data_h5': 'miniplaces_256_val.h5',
#    'data_root': '../../../../data/images/',   # MODIFY PATH ACCORDINGLY
#    'data_list': '../../../../data/val.txt',   # MODIFY PATH ACCORDINGLY
#    'load_size': load_size_for_alex,
#    'fine_size': fine_size_for_alex,
#    'data_mean': data_mean,
#    'randomize': False
#    }

opt_data_val_CJ = {
        #'data_h5': 'miniplaces_256_val.h5',
        'data_root': '../../../../data/images/',   # MODIFY PATH ACCORDINGLY
        'data_list': '../../../../data/val.txt',   # MODIFY PATH ACCORDINGLY
        'load_size': load_size_for_CJ,
        'fine_size': fine_size_for_CJ,
        'data_mean': data_mean,
        'randomize': False
        }

#loader_train_alex = DataLoaderDisk(**opt_data_train_alex)
#loader_val_alex = DataLoaderDisk(**opt_data_val_alex)
loader_train_CJ = DataLoaderDisk(**opt_data_train_CJ)
loader_val_CJ = DataLoaderDisk(**opt_data_val_CJ)

#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
#x_alex = tf.placeholder(tf.float32, [None, fine_size_for_alex, fine_size_for_alex, c])
x = tf.placeholder(tf.float32, [None, fine_size_for_CJ, fine_size_for_CJ, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits_CJ = CJNet1(x, keep_dropout, train_phase)
#logits_alex = alexnet(x_alex, keep_dropout, train_phase)
    
# Define loss and optimizer
loss_CJ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits_CJ))
#loss_alex=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits_alex))
train_optimizer_CJ = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_CJ)
#train_optimizer_alex = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_alex)

# Evaluate model
#accuracy1_alex = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_alex, y, 1), tf.float32))
accuracy1_CJ = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_CJ, y, 1), tf.float32))
#accuracy5_alex = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_alex, y, 5), tf.float32))
accuracy5_CJ = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_CJ, y, 5), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

# Launch the graph
with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)

    step = 0

    while step < training_iters:
        # Load a batch of training data
        images_batch_CJ, labels_batch_CJ = loader_train_CJ.next_batch(batch_size)
        #images_batch_alex, labels_batch_alex = loader_train_alex.next_batch(batch_size)
        
        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l_CJ, acc1_CJ, acc5_CJ = sess.run([loss_CJ, accuracy1_CJ, accuracy5_CJ], feed_dict={x: images_batch_CJ, y: labels_batch_CJ, keep_dropout: 1., train_phase: False})
            print("-Iter " + str(step) + ", Training Loss for CJ= " + \
                  "{:.6f}".format(l_CJ) + ", Accuracy Top1 for CJ = " + \
                  "{:.4f}".format(acc1_CJ) + ", Top5 for CJ= " + \
                  "{:.4f}".format(acc5_CJ))

           #   l_alex, acc1_alex, acc5_alex = sess.run([loss_alex, accuracy1_alex, accuracy5_alex], feed_dict={x: images_batch_alex, y: labels_batch_alex, keep_dropout: 1., train_phase: False})
           #  print("-Iter " + str(step) + ", Training Loss= " + \
           #       "{:.6f}".format(l) + ", Accuracy Top1 = " + \
           #      "{:.4f}".format(acc1) + ", Top5 = " + \
           #     "{:.4f}".format(acc5))
            
            # Calculate batch loss and accuracy on validation set
            images_batch_val_CJ, labels_batch_val_CJ = loader_val_CJ.next_batch(batch_size)
            l_CJ, acc1_CJ, acc5_CJ = sess.run([loss_CJ, accuracy1_CJ, accuracy5_CJ], feed_dict={x: images_batch_val_CJ, y: labels_batch_val_CJ, keep_dropout: 1., train_phase: False})
            print("-Iter " + str(step) + ", Validation Loss for CJ= " + \
                  "{:.6f}".format(l_CJ) + ", Accuracy Top1 for CJ= " + \
                  "{:.4f}".format(acc1_CJ) + ", Top5 for CJ= " + \
                  "{:.4f}".format(acc5_CJ))

           # images_batch_val_alex, labels_batch_val_alex = loader_val_alex.next_batch(batch_size)
           # l_alex, acc1_alex, acc5_alex = sess.run([loss_alex, accuracy1_alex, accuracy5_alex], feed_dict={x: images_batch_val_alex, y: labels_batch_val_alex, keep_dropout: 1., train_phase: False})
           # print("-Iter " + str(step) + ", Validation Loss= " + \
           #       "{:.6f}".format(l) + ", Accuracy Top1 = " + \
           #       "{:.4f}".format(acc1) + ", Top5 = " + \
           #       "{:.4f}".format(acc5))

        # Run optimization op (backprop)
        sess.run(train_optimizer_CJ, feed_dict={x: images_batch_CJ, y: labels_batch_CJ, keep_dropout: dropout, train_phase: True})
        #sess.run(train_optimizer_alex, feed_dict={x: images_batch_alex, y: labels_batch_alex, keep_dropout: dropout, train_phase: True})
        
        step += 1

        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))

    print("Optimization Finished!")


    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch_CJ = loader_val_CJ.size()//batch_size
    num_batch_alex = loader_val_alex.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    
    loader_val_CJ.reset()
    for i in range(num_batch_CJ):
        images_batch_CJ, labels_batch_CJ = loader_val_CJ.next_batch(batch_size)
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch_CJ, y: labels_batch_CJ, keep_dropout: 1., train_phase: False})
        acc1_total += acc1
        acc5_total += acc5
        #print("Validation Accuracy Top1 = " + \
        #      "{:.4f}".format(acc1) + ", Top5 = " + \
        #      "{:.4f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation whole valsets Finished CJ! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))

    #acc1_total = 0.
    #acc5_total = 0.
    #loader_val_alex.reset()
    #for i in range(num_batch_alex):
    #    images_batch_alex, labels_batch_alex = loader_val_alex.next_batch(batch_size)
    #    acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch_alex, y: labels_batch_alex, keep_dropout: 1., train_phase: False})
    #    acc1_total += acc1
    #    acc5_total += acc5
        #print("Validation Accuracy Top1 = " + \
        #      "{:.4f}".format(acc1) + ", Top5 = " + \
        #      "{:.4f}".format(acc5))
    
    #acc1_total /= num_batch
    #acc5_total /= num_batch
    #print('Evaluation whole valsets Finished alex! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
