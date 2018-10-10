# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:2018-10-10
# Email: hey_xiaoqi@163.com
# Filename: classifier.py
# Description: this file define the classifier, trained and test code is ine this file
# *************************************************

import os
import sys
import tensorflow as tf

ASC_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ASC_PROJECT_ROOT)

def create_flags():
  """
  arguments defination
  """
  tf.app.flags.DEFINE_string("model_root", "/home/guwenqi/Documents/ASC/train/model/cnn/classical", "directory where to save the model")
  tf.app.flags.DEFINE_string("ckpt_prefix", "/home/guwenqi/Documents/ASC/train/model/cnn/classical/ckpt", "checkpoint prefix")
  tf.app.flags.DEFINE_string("data_root", "/home/guwenqi/Documents/ASC/train/feature/mbe_feature", "directory from where fetch the data")
  tf.app.flags.DEFINE_string("transcripts", "/home/guwenqi/Documents/ASC/train/feature/mbe_feature/label.txt", "file lists the training files and corresponding transcription")
  tf.app.flags.DEFINE_string("logfile", "/home/guwenqi/Documents/ASC/train/log/log.txt", "log file path")


def train():
  """
  train the network
  """
