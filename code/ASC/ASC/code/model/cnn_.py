# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: cnn.py
# Description: this file construct a classical cnn network
# *************************************************

import os
import sys
import math
import logging
import tensorflow as tf

ASC_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ASC_PROJECT_ROOT)

from ASC.code.base.ASCDataIterator import ASCDataIterator
from ASC.code.base.cnn_layers import conv2d_bn_relu_pool_drop_layer
from ASC.code.base.dnn_layers import fnn_bn_relu_drop_layer
from ASC.code.tools.audio import audio2MBE_inputs
from ASC.code.tools.post_procession import MBEProbs2speech_music_single

MODEL_ROOT = '/home/guwenqi/Documents/ASC/train/model/cnn/classical'
checkpoint_prefix = os.path.join(MODEL_ROOT, 'ckpt')
model_prefix = ""

DATA_ROOT = '/home/guwenqi/Documents/ASC/train/feature/mbe_feature'
LABELS_PATH = '/home/guwenqi/Documents/ASC/train/feature/mbe_feature/label.txt'
logfile = '/home/guwenqi/Documents/ASC/train/log/log.txt'


def CNN(inputs, is_training):
  """
  param : inputs, inputs of this network should have shape=[batch, mbe_order, T_frames]
  param : is_training, bool, weather in train phrase
  """
  batches = tf.shape(inputs)[0]
  
  # cnn layers
  # insert one dim at end
  cnn_inputs = tf.expand_dims(inputs, axis=-1)
  with tf.name_scope("cnn/1"):
    conv_outputs = tf.layers.conv2d(cnn_inputs,
                                    filters=4, kernel_size=[5, 5],
                                    strides=[2, 2], padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
    batch_norm = tf.layers.batch_normalization(conv_outputs, training=is_training)
    relu_outputs = tf.nn.relu(batch_norm)
    pool_outputs = tf.layers.max_pooling2d(relu_outputs, pool_size=[2, 2], strides=[2, 2], padding='same')
    dropout_outputs = tf.layers.dropout(pool_outputs, training=is_training)


  fnn_inputs = tf.reshape(dropout_outputs, [batches, 8*25*4])
  with tf.name_scope('fnn/1'):
    fnn_outputs = tf.layers.dense(fnn_inputs, units=24, use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  kernel_regularizer=tf.contrib.layers.l1_regularizer(0.01))
    batch_norm = tf.layers.batch_normalization(fnn_outputs, training=is_training)
    relu_outputs = tf.nn.relu(batch_norm)
    dropout_outputs = tf.layers.dropout(relu_outputs, rate=0.3, training=is_training)

  with tf.name_scope('logits'):
    logits = tf.squeeze(tf.layers.dense(dropout_outputs, units=1,
                        kernel_initializer=tf.contrib.layers.xavier_initializer()))
  return logits

def CNN2(inputs, is_training):
  """
  construct a two cnn layers + 1 fnn network
  param : inputs, inputs of this network should have shape=[batch, mbe_order, T_frames]
  param : is_training, bool, weather in train phrase
  """
  # network configuration
  cnn_layers_num = 2
  filters = [8, 16]
  kernel_sizes = [[5, 5], [5, 5]]
  conv_strides = [[2, 2], [1, 1]]
  pool_sizes = [[2, 2], [2, 2]]
  pool_strides = [[2, 2], [2, 2]]
  hight = 32
  width = 100
  for i in list(range(cnn_layers_num)):
    hight = int(math.ceil(math.ceil(32/conv_strides[i][0]) / pool_strides[i][0]))
    width = int(math.ceil(math.ceil(32/conv_strides[i][1]) / pool_strides[i][1]))
  cnn_units = hight * width * filters[-1]


  # now start construct network
  batches = tf.shape(inputs)[0]
  # cnn layers
  # insert one dim at end
  cnn_inputs = tf.expand_dims(inputs, axis=-1)
  for i in list(range(cnn_layers_num)):
    with tf.name_scope("cnn/%d" % (i+1)):
      cnn_inputs = conv2d_bn_relu_pool_drop_layer(cnn_inputs, filters[i], kernel_sizes[i], conv_strides[i],
                                                   pool_sizes[i], pool_strides[i], is_training, 0.5,
                                                   dilation_rate=(1, 1), conv_pad='same', pool_pad='same')
  
  # fnn layers
  fnn_inputs = tf.reshape(cnn_inputs, [batches, cnn_units])
  with tf.name_scope('fnn/1'):
    fnn_outputs = fnn_bn_relu_drop_layer(fnn_inputs, cnn_units, keep_prob=0.5, is_training=is_training,
                                         kernel_regularizer=tf.contrib.layers.l1_regularizer(0.01))

  # logits
  with tf.name_scope('logits'):
    logits = tf.squeeze(tf.layers.dense(fnn_outputs, units=1,
                        kernel_initializer=tf.contrib.layers.xavier_initializer()))
  return logits


def create_inference_graph():
  """
  create a inference graph on default graph
  """
  inputs = tf.placeholder(tf.float32, shape=[None, 32, 100])
  logits = CNN(inputs, False)
  probs = tf.nn.sigmoid(logits)

  return inputs, probs


def single_file_inference(filepath, labelpath=None):
  """
  tag one singel audio file
  param filepath : string, path of the audio
  param labelpath : string, path where to store the labels
  """
  inputs, probabilities = create_inference_graph()
  saver = tf.train.Saver()

  # ce=reate session
  with tf.Session() as sess:
    saver.restore(sess, model_prefix)

    data = audio2MBE_inputs(filepath)
    probs = sess.run(probabilities, feed_dict={inputs : data})

    MBEProbs2speech_music_single(filepath, probs, labelpath=labelpath, context=2)

def batch_file_inference(wavdir, labeldir=None):
  """
  tag all wav files under wavdir
  param wavdir : string, directory path under which wav files need to be tagged
  param labeldir : string, directory under which to store the label files
  """
  """
  inputs, probabilities = create_inference_graph()
  saver = tf.train.Saver()

  # ce=reate session
  with tf.Session() as sess:
    saver.restore(sess, model_prefix)

  """
  pass

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

  # construct training graph
  g_train = tf.Graph()
  with g_train.as_default():

    # inputs of this network, shape=[batch, mbe_order, T_frame, channel]
    inputs = tf.placeholder(tf.float32, shape=[None, 32, 100])
    # shape=[batch, K_class], for music/speech classification, K=2
    targets = tf.placeholder(tf.int32, shape=[None])
    # is_training
    is_training = tf.placeholder(tf.bool)
    # learning rate
    learning_rate = tf.placeholder(tf.float32)

    # get logits from network
    logits = CNN(inputs, is_training)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(targets), logits=logits))
    predicts = tf.where(tf.nn.sigmoid(logits) >= 0.5, tf.ones_like(logits, dtype=tf.int32), tf.zeros_like(logits, dtype=tf.int32))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, targets), dtype=tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # because the batch normalization relies on no-gradient updates, so we need add tf.control_dependencies
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)

    saver = tf.train.Saver(max_to_keep=10)


  with tf.Session(graph=g_train) as sess:
    try:
      last_checkpoint = ''
      with open(os.path.join(MODEL_ROOT, 'checkpoint'), 'r') as f:
        last_checkpoint = f.readline().strip().split()[-1].strip('"')
      saver.restore(sess, last_checkpoint)
    except:
      tf.global_variables_initializer().run()

    print('Begain training')

    count = 0
    loss_all = 0
    accuracy_all = 0
    while True:
      # fetch data
      data, labels, epoch_done = asciter.next_batch()

      if data is not None:

        loss_, _, accuracy_ = sess.run([loss, train_op, accuracy],
                            feed_dict={
                            inputs : data,
                            targets : labels,
                            is_training : True,
                            learning_rate : 0.001 * 0.8**asciter.kth_epoch
                            })
        # print("epoch: {}, batch: {}, loss: {}, accuracy: {}".format(asciter.kth_epoch, asciter.ith_batch, loss_, accuracy_))
        count += 1
        loss_all += loss_
        accuracy_all += accuracy_

        if asciter.ith_batch % 100 == 0:
          loss_ave = loss_all / count
          accuracy_ave = accuracy_all / count
          loss_all = 0
          accuracy_all = 0
          count = 0

          print("Epoch: %-2d, Batch: %-4d, Average loss: %-5f, Average accuracy: %-5f." % (asciter.kth_epoch, asciter.ith_batch, loss_ave, accuracy_ave))

        if asciter.ith_batch % 500 == 0 or epoch_done:
          # validation
          val_batch_num = int(len(asciter.val_list) / asciter.batch_size) + 1
          val_accuracy_all = 0
          for i in list(range(val_batch_num)):
            val_data, val_targets = asciter.fetch_data(i*asciter.batch_size, (i+1)*asciter.batch_size)
            if len(val_data) != 0:
              val_accuracy_ = sess.run(accuracy,
                                       feed_dict={
                                       inputs : val_data,
                                       targets : val_targets,
                                       is_training : False,
                                       learning_rate : 0.001 * 0.8**asciter.kth_epoch
                                       })
              val_accuracy_all += val_accuracy_

          saver.save(sess, checkpoint_prefix+"-{}-{}".format(asciter.kth_epoch, asciter.ith_batch))
          asciter.save(os.path.join(MODEL_ROOT, 'ASCDataIterator.ckpt'))
          logger.info("Epoch: %-2d, Batch: %-4d, Validation Accuracy: %-5f, Model saved!" % (asciter.kth_epoch, asciter.ith_batch, val_accuracy_all/val_batch_num))
          
          # # if epoch done, compute the average train accuracy
          # if epoch_done:
          #   train_batch_num = int(len(asciter.train_list) / asciter.batch_size) + 1
          #   train_accuracy_all = 0
          #   for i in list(range(train_batch_num)):
          #     train_data, train_targets = asciter.fetch_data(i*asciter.batch_size, (i+1)*asciter.batch_size, 'train')
          #     if (len(train_data) != 0):
          #       train_accuracy_ = sess.run(accuracy,
          #                                  feed_dict={
          #                                  inputs : train_data,
          #                                  targets : train_targets,
          #                                  is_training : False,
          #                                  learning_rate : 0.001 * 0.9**asciter.kth_epoch
          #                                  })
          #       train_accuracy_all += train_accuracy_
          #   logger.info("Epoch: %-2d, Batch: %-4d, Training Set Accuracy: %-5f" % (asciter.kth_epoch, asciter.ith_batch, train_accuracy_all/train_batch_num))
      
      else:
        break

if __name__ == '__main__':
  train()
