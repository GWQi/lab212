# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: resnet.py
# Description: this file construct a classical cnn network
# *************************************************

import os
import sys
import logging

import tensorflow as tf

ASC_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ASC_PROJECT_ROOT)

from ASC.code.base.ASCDataIterator import ASCDataIterator
from ASC.code.base import fparam

MODEL_ROOT = '/home/guwenqi/dataset/ASC/model/cnn/resnet'
checkpoint_prefix = os.path.join(MODEL_ROOT, 'ckpt')

DATA_ROOT = '/home/guwenqi/dataset/ASC/feature'
LABELS_PATH = '/home/guwenqi/dataset/ASC/feature/label.txt'
logfile = '/home/guwenqi/dataset/ASC/log/log.txt'

def CNN():

  network = {}
  # inputs of this network, shape=[batch, mbe_order, T_frame, channel]
  inputs = tf.placeholder(tf.float32, shape=[None, 64, 100])
  # shape=[batch, K_class], for music/speech classification, K=2
  targets = tf.placeholder(tf.int32, shape=[None])
  # is_training
  is_training = tf.placeholder(tf.bool)
  # expend one dimension
  batches = tf.shape(inputs)[0]
  inputs_network = tf.expand_dims(inputs, axis=-1)

  # cnn layer before resnet
  with tf.name_scope('cnn/1'):
    conv_outputs = tf.layers.conv2d(inputs_network,
                                    filters=16, kernel_size=[7, 7],
                                    strides=[1, 1], padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
    relu_ouputs = tf.nn.relu(conv_outputs)
    pool_ouputs = tf.layers.max_pooling2d(relu_ouputs, pool_size=2, strides=2, padding='SAME')
    dropout_outputs = tf.layers.dropout(pool_ouputs, rate=0.5)

  res_ouputs = dropout_outputs

  # resnet
  channels = [16, 32, 64, 128]
  for n, channel in enumerate(channels):
    res_ouputs = residual_block(res_ouputs, 3, channel, is_training, 0.5, 'relu', n+1)

  # I get this 2048 through sees.run(tf.shape(fnn_inputs))
  fnn_inputs = tf.reshape(res_ouputs, [batches, 512])

  # fnn layers
  fnn_1 = fnn_bn_relu_drop_layer(fnn_inputs, 512, 0.5, is_training, 1)
  with tf.variable_scope('fnn/2'):
    w = tf.get_variable('w', shape=[fnn_1.get_shape().as_list()[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', shape=[1], initializer=tf.zeros_initializer())
    # logits.shape = [batches, K_class]
    logits = tf.squeeze(tf.matmul(fnn_1, w) + b)

  network['inputs'] = inputs
  network['targets'] = targets
  network['is_training'] = is_training
  network['logits'] = logits
  
  return network

def train():

  # iterator initialization
  asciter = ASCDataIterator()
  try:
    asciter.load(os.path.join(MODEL_ROOT, 'ASCDataIterator.ckpt'))
  except:
    asciter.configure(DATA_ROOT, LABELS_PATH)

  # logger configuration
  logging.basicConfig(filename=logfile,
                      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                      filemode='a',
                      level=logging.DEBUG)
  logger = logging.getLogger()




  g_train = tf.Graph()
  with g_train.as_default():
    network = CNN()
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(network['targets']), logits=network['logits']))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    predicts = tf.where(tf.nn.sigmoid(network['logits']) >= 0.5, tf.ones_like(network['logits'], dtype=tf.int32), tf.zeros_like(network['logits'], dtype=tf.int32))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, network['targets']), dtype=tf.float32))
    saver = tf.train.Saver(max_to_keep=5)

  with tf.Session(graph=g_train) as sess:
    try:
      saver.restore(sess, checkpoint_prefix)
    except:
      tf.global_variables_initializer().run()

    print('Begain training')

    count = 0
    cost_all = 0
    accuracy_all = 0
    while True:
      # fetch data
      data, targets, epoch_done = asciter.next_batch()

      if data is not None:

        cost_, _, accuracy_ = sess.run([cost, optimizer, accuracy],
                            feed_dict={
                            network['inputs'] : data,
                            network['targets'] : targets,
                            network['is_training'] : True
                            })

        print(cost_, accuracy_)
        count += 1
        cost_all += cost_
        accuracy_all += accuracy_

        if asciter.ith_batch % 100 == 0:
          cost_ave = cost_all / count
          accuracy_ave = accuracy_all / count
          cost_all = 0
          accuracy_all = 0
          count = 0


          saver.save(sess, checkpoint_prefix)
          asciter.save(os.path.join(MODEL_ROOT, 'ASCDataIterator.ckpt'))
          logger.info("{}'th epoch, {}'th batch save model and iterator.".format(asciter.kth_epoch, asciter.ith_batch))
          logger.info("{}'th epoch, {}'th batch average cost: {}, average accuracy: {}.".format(asciter.kth_epoch, asciter.ith_batch, cost_ave, accuracy_ave))

        if epoch_done:
          # save model
          saver.save(sess, checkpoint_prefix)
          asciter.save(os.path.join(MODEL_ROOT, 'ASCDataIterator.ckpt'))
          logger.info("{}'th epoch done, save model and iterator.".format(asciter.kth_epoch-1))

          # validation
          val_batch_num = int(len(asciter.val_list) / asciter.batch_size) + 1
          val_accuracy_all = 0
          for i in list(range(val_batch_num)):
            val_data, val_targets = asciter.fetch_data(i*asciter.batch_size, (i+1)*asciter.batch_size)
            if len(val_data) != 0:
              val_accuracy_ = sess.run(accuracy,
                                       feed_dict={
                                       network['inputs'] : val_data,
                                       network['targets'] : val_targets,
                                       network['is_training'] : False
                                       })
              val_accuracy_all += val_accuracy_
          logger.info("{}'th epoch done, average accuracy on validation set: {}.".format(asciter.kth_epoch-1, val_accuracy_all/val_batch_num))
      else:
        break
if __name__ == '__main__':
  train()
