# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: dnn_layers.py.py
# Description: this file is the base function to construct dnn layers
# *************************************************
import tensorflow as tf

def fnn_bn_relu_drop_layer(inputs_layer, units, keep_prob, is_training, ith_fnn):
  """
  a helper function to construct a fnn
  params:
    inputs_layer : 2D tensor, shape=[batches, dim]
    units : int, number units of this layer
    keep_prob : float, keep probability
    is_training : bool, weather in training phrase
    active_function : 'sigmoid', 'relu'
    ith_fnn : int
  """
  with tf.variable_scope('fnn/{}'.format(ith_fnn)):
    w = tf.get_variable('w', shape=[inputs_layer.get_shape().as_list()[-1], units], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', shape=[units], initializer=tf.zeros_initializer())
    # fnn layer outputs
    fnn_layer = tf.matmul(inputs_layer, w) + b

    # fnn_layer = tf.layers.dense(inputs_layer, units, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    # batch normalization
    batch_norm = tf.layers.batch_normalization(fnn_layer, training=is_training)
    # activation layer
    outputs_layer = tf.nn.relu(batch_norm)
    # dropout layer
    outputs_layer = tf.contrib.layers.dropout(outputs_layer, keep_prob=keep_prob, is_training=is_training)

    return outputs_layer