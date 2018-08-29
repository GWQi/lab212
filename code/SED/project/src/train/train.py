# Copyright 2018/8 by wenqi gu
# dcase dataset model training file
# author Wenqi Gu
import tensorflow as tf
import sys
import os

# get the main project path and append it to sys.path
CODE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(CODE_PATH)

from SED.project.src.preprocessing.dcase_preprocess import DCASEData2017Task2

# this directory used to save model and tf.checkpoint
MODEL_DIR = os.path.join(CODE_PATH, 'SED/project/model')

DCASEData2017Task2 source
source.load(os.path.join(MODEL_DIR, 'DCASEData2017Task2.ckp'))







def predict(inputs, keep_prob):
  """
  this function is used to build the model
  params:
    x : input feature, shape=[batch_size, n_oredr, T_frames]
    keep_prob : keep_prob = 1 - drop_out
  return:
    y_ : predict
  """

  # first reshape the data
  with tf.name_scope("reshape"):
    inputs_reshape = tf.reshape(inputs, [-1, source.feaparams.mbe_num, source.feaparams.sequence_length, 1])

  # CNN layer
  with tf.name_scope("conv1"):
    W_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 16]), name="W_1")
    b_1 = tf.Variable(tf.truncated_normal([16]), name="b_1")
    h_conv1 = tf.relu(tf.add(tf.nn.conv2d(inputs_reshape, W_1, strides=[1, 1, 1, 1], padding='SAME'), b_1))
  with tf.name_scope("pooling1"):
    h_pool1 = tf.nn.max_pool(h_conv1, [1, 2, 2, 1], [1, 2, 1, 1], padding='SAME')



  with tf.name_scope("conv2"):
    W_2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32]), name="W_2")
    b_2 = tf.Variable(tf.truncated_normal([32]), name="b_2")
    h_conv2 = tf.relu(tf.add(tf.nn.conv2d(h_pool1, W_2, strides=[1, 1, 1, 1], padding='SAME'), b_2))

  with tf.name_scope("pooling2"):
    h_pool2 = tf.nn.max_pool(h_conv2, [1, 2, 2, 1], [1, 2, 1, 1], padding='SAME')
    

def conv2d(inputs, kernel_size, outchannel, pool_strides, name):
  """
  create a conv2d layer
  params:
    knernel_size : list, [vertical_size, horizonal_size]
    outchannel : int, number of outputs channel
    pool_strides : list, [batch_stride, vertical_stride, horizonal_stride, channel_strid]
    name : string, name prefix of this conv layer
  """
  inputs_shape = tf.shape(inputs)
  inchannel = shape[-1]
  kernel_shape = kernel_size + [inchannel, outchannel]

  w = tf.Variable(tf,truncated_normal([kernel_shape]), name=name+"/w")
  b = tf.Variable(tf.truncated_normal([outchannel]), name=name+"/b")
  h_conv = tf.nn.relu(tf.add(tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding='SAME'), b))
  h_pool = tf.nn.max_pool(h_conv, [1, 2, 2, 1], pool_strides, padding'SAME')

  return h_pool


