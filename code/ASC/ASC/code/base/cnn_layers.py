# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: dnn_layers.py.py
# Description: this file is the base function to construct cnn layers
# *************************************************
import tensorflow as tf

def conv2d_bn_relu_drop_layer(inputs_layer, weight, stride_h, stride_w, keep_prob, is_training):
  """
  a helper function to conv, batch normalization, relu activation
  params:
    inputs_layer : 4D tensor, shape-[batch, hight, width, channel]
    weight : filter
    stride_h : int, stride along with hight
    stride_w : int, stride along with width
    is_training : weather training or validation, training phase if True
    keep_prob : float, keep probability when drop out
  return:
    outputs of this layer
  """

  # convolution layer
  conv_layer = tf.nn.conv2d(inputs_layer, weight, strides=[1, stride_h, stride_w, 1], padding='SAME')
  # batch normalizatiom
  batch_norm = tf.layers.batch_normalization(conv_layer, training=is_training)
  # activation
  outputs_layer = tf.nn.relu(batch_norm)
  # drop out
  outputs_layer = tf.contrib.layers.dropout(outputs_layer, keep_prob=keep_prob, is_training=is_training)

  return outputs_layer

def conv2d_bn_relu_pool_drop_layer(inputs_layer, out_channels, keep_prob, is_training, kernal_size=3, kernal_strides=1, pool_strides=2):
  """
  a helper function to conv, batch normalization, relu activation
  params:
    inputs_layer : 4D tensor, shape-[batch, hight, width, channel]
    out_channels : int
    kernel_size : An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution 
    kernal_strides :  An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width
    pool_strides : An integer or tuple/list of 2 integers, specifying the strides of the max pooling along the height and width
    is_training : weather training or validation, training phase if True
    keep_prob : float, keep probability when drop out
  return:
    outputs of this layer
  """
  # convolution layer
  conv_layer = tf.layers.conv2d(inputs_layer, out_channels, kernal_size, kernal_strides, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
  # batch normalizatiom
  batch_norm = tf.layers.batch_normalization(conv_layer, training=is_training)
  # activation
  relu_layer = tf.nn.relu(batch_norm)
  # max pooling layer
  pool_layer = tf.layers.max_pooling2d(relu_layer, 2, pool_strides, padding='same')
  # drop out
  outputs_layer = tf.contrib.layers.dropout(pool_layer, keep_prob=keep_prob, is_training=is_training)

  return outputs_layer



def residual_block(inputs_layer, kernel_size, out_channels, is_training, keep_prob, active_function, ith_block):
  """
  construct a residual block
  params:
    inputs_layer : float tensor, inputs of this block, 4D, shape=[batch, hight, width, channel]
    kernel_size : weight kernel size
    out_channels : int, channels of outputs
    is_training : tf.bool, weather in training phase
    keep_prob : float, keep probability when drop out
    active_function : string, 'relu' or None, active function of the output
    ith_block : int
  """
  in_channels = inputs_layer.get_shape().as_list()[-1]
  # weather increase channel after this block
  increase_channel = False
  if in_channels * 2 == out_channels:
    increase_channel = True
    stride = 2
  elif in_channels == out_channels:
    stride = 1
  else:
    raise ValueError("Out channel and in channel don't match at {}'th block!!!".format(ith_block))

  with tf.variable_scope("resnet/{}_block".format(ith_block)):
    weight_1 = tf.get_variable('w_1', shape=[kernel_size, kernel_size, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer())
    conv_1 = conv2d_bn_relu_drop_layer(inputs_layer, weight_1, stride, stride, keep_prob, is_training)
    weight_2 = tf.get_variable('w_2', shape=[kernel_size, kernel_size, out_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer())
    conv_2 = conv2d_bn_relu_drop_layer(conv_1, weight_2, 1, 1, keep_prob, is_training)
    
    if increase_channel:
      # if the input channel and output channel of this block are different, use 1x1 filter to pad
      shortcuts_weight = tf.get_variable('w_3', shape=[1, 1, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer())
      shortcuts = tf.nn.conv2d(inputs_layer, shortcuts_weight, strides=[1, stride, stride, 1], padding='SAME')
    else:
      shortcuts = inputs_layer

  outputs_layer = shortcuts + conv_2

  if active_function == 'relu':
    return tf.nn.relu(outputs_layer)
  else:
    return outputs_layer