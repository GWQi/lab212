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


def conv2d_bn_relu_pool_drop_layer(inputs, filters, kernel_size, conv_strides,
                                    pool_size, pool_strides, is_training, keep_prob,
                                    dilation_rate=(1, 1), conv_pad='same', pool_pad='same'):
  """
  construct a convlution layers, with batch normalization, relu activation, max pooling, dropout layers
  params:
    inputs : 4D tensor inputs, inputs.shape=[batches, hight, width, channel]
    filters : int, number of filters/channels of outputs
    kernel_size : int or list/tuple of 2 integers, (hight, width)
    conv_strides : int or list/tuple of 2 integers, (hight, width)
    pool_size : int or list/tuple of 2 integers, (pool_hight, pool_width)
    pool_strides : int or list/tuple of 2 integers,, (hight, width)
    is_training : python bool or tf.bool(normally placeholder), weather in training phase
    keep_prob : keep probability when dropout, between 0 and 1
    dilation_rate : dilation rate along hight and width
    conv_pad : convolutional padding, 'same' or 'valid'
    pool_pad : pooling padding, 'same' or 'valid'
  """
  # convolution layer
  conv_outputs = tf.layers.conv2d(inputs, filters, kernel_size,
                                  strides=conv_strides, padding=conv_pad,
                                  dilation_rate=dilation_rate, use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
  # batch normalization
  batch_norm_outputs = tf.layers.batch_normalization(conv_outputs, training=is_training)
  # relu activation
  relu_outputs = tf.nn.relu(batch_norm_outputs)
  # pooling layer
  pooling_outputs = tf.layers.max_pooling2d(relu_outputs, pool_size,
                                            pool_strides, padding=pool_pad)
  # apply dropout
  dropout_outputs = tf.layers.dropout(pooling_outputs, rate=1-keep_prob,
                                      training=is_training)
  return dropout_outputs
  
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