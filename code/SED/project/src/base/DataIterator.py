# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

class DataIterator(ABCMeta):
  """
  base abstract data iterator class for all data
  """
  def __init__(self, root):
    """
    initialization
    param root : str, path of data root directory
    """
    # data root directory
    self.data_root = root
    # list of data files path
    self.data_list = []
    # list of train data files path
    self.train_list = []
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
    self.kth_epoch = 0
    # total number of epoch
    self.num_epoch = 0

  @abstractmethod
  def next_batch(self):
    """
    get next batch data and corresponding labels to train the network, 
    """
    pass

  @abstractmethod
  def fetch_data(self, files_list):
    """
    fetch some data and corresponding labels
    """
    pass

  @abstractmethod
  def save(self, path):
    """
    
    """

