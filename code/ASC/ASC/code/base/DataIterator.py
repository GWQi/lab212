# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: DataIterator.py
# Description: this file is the base class of data fetcher
# *************************************************

from abc import ABCMeta, abstractmethod

class DataIterator(object):
  """
  base abstract data iterator class for all data
  """
  
  __metaclass__ = ABCMeta

  def __init__(self):
    """
    initialization
    param root : str, path of data root directory
    """
    # data root directory
    self.data_root = ''
    # list of data files path
    self.data_list = []
    # list of train data files path
    self.train_list = []
    # list of training file indexs
    self.train_indexes = []
    # list of validation data files path
    self.val_list = []
    # list of test data files path
    self.test_list = []
    # 

    # batch counter
    self.ith_batch = 0
    # batch size
    self.batch_size = 0
    # epoch counter
    self.kth_epoch = 1
    # total number of epoch
    self.num_epoch = 0

  @abstractmethod
  def next_batch(self):
    """
    get next batch data and corresponding labels to train the network, 
    """
    pass

  @abstractmethod
  def fetch_data(self):
    """
    fetch some data and corresponding labels
    """
    pass

  @abstractmethod
  def save(self, path):
    """
    save this data iterator to disk
    """
    pass

  @abstractmethod
  def load(self, path):
    """
    load existed data iterotor
    """
    pass

  @abstractmethod
  def configure(self):
    """
    DataIterator configuration
    """
    pass
