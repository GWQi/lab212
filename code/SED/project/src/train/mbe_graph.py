import os
import sys

import tensorflow as tf
import numpy as np


def create_flags():
  """
  some flags used in net
  """

  # file path
  # ****************
  tf.app.flags.DEFINE_string ("train_files_path",                "",                  "file path contains the files path used to train model")
  tf.app.flags.DEFINE_string ("val_files_path",                  "",                  "file path contains the files path used to validation")
  tf.app.flags.DEFINE_string ("test_files_path",                 "",                  "file path contains the files path used to test the model")

  # global constants
  # ****************
  tf.app.flags.DEFINE_boolean("train",                           True,                "weather to train the network")
  tf.app.flags.DEFINE_boolean("test",                            True,                "weather to test the network")
  