# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: ASCDataIterator.py
# Description: this file is the data fetcher of speech/music data
# *************************************************
import os
import pickle
import numpy as np
from random import shuffle
from ASC.code.base.DataIterator import DataIterator

class ASCDataIterator(DataIterator):

  def __init__(self):
    super(ASCDataIterator, self).__init__()
    # self.data_list = [[filepath, label], [filepath, label], ...]

    self.binary_class = {'speech' : 1, 'music' : 0}
    self.multi_class = {'speech' : [1, 0], 'music' : [0, 1]}


  def next_batch(self):
    """
    return:
      data  : list of feature arrays, np.ndarrays. feature.shape=[fea_order, T_frames]
      targets: list of targets
      epoch_done: bool, indicate one epoch done
    """
    # flag, indicate one epoch done
    epoch_done = False
    # fetched data
    data = []
    targets = []
    # advance iterator
    self.ith_batch += 1

    if (self.ith_batch-1) * self.batch_size >= len(self.train_list):
      self.ith_batch = 1
      self.kth_epoch += 1
      epoch_done = True
      # weather all epoch done
      if self.kth_epoch > self.num_epoch:
        return None, None, None

      shuffle(self.train_indexes)

    # fetch data and labels
    for index in self.train_indexes[(self.ith_batch-1)*self.batch_size : self.ith_batch*self.batch_size]:
      filepath, label = self.train_list[index]
      fea = np.load(os.path.join(self.data_root, filepath))
      # translate label to integer
      target = self.binary_class[label]

      data.append(fea)
      targets.append(target)

    return data, targets, epoch_done

  def fetch_data(self, start, end, dataname='val'):
    """
    fetch data set
    param dataname: string, if 'val', fetch validation data, else if 'test', fetch test data
    return:
      data  : list of feature arrays, np.ndarrays. feature.shape=[fea_order, T_frames]
      targets: list of targets
    """
    data_list = []
    if dataname == 'val':
      data_list = self.val_list[start:end]
    elif dataname == 'test':
      data_list = self.test_list[start:end]
    else:
      raise('you just can fetch validation data and test data')

    # fetched data
    data = []
    targets = []

    for filepath, label in data_list:
      fea = np.load(os.path.join(self.data_root, filepath))
      # translate label to integer
      target = self.binary_class[label]

      data.append(fea)
      targets.append(target)

    return data, targets

  def configure(self, root, labelpath):
    """
    configuration of data iteratot
    params:
      root : string, data root path
      labelpath : string, label path
    """
    # initializethe batch size and number epoch
    self.data_root = root
    self.batch_size = 128
    self.num_epoch = 100

    # validation data partion
    val_partion = 0.1

    with open(labelpath, 'r') as f:
      for aline in f.readlines():
        self.data_list.append(aline.strip().split())

    # get train/validationtest data list
    gap = int(1 / val_partion)
    for i in list(range(len(self.data_list))):
      if i % gap == 0:
        self.val_list.append(self.data_list[i])
      else:
        self.train_list.append(self.data_list[i])
    self.test_list = list(self.val_list)

    # generate train indexes
    self.train_indexes = list(range(len(self.train_list)))
    for i in list(range(20)):
      shuffle(self.train_list)
    return

  # @override
  def save(self, ckpt_path):
    """
    save this iterator
    params:
      ckpt_path : checkpoint path
    """
    with open(ckpt_path, 'wb') as f:
      pickle.dump(self, f)


  # @override
  def load(self, ckpt_path):
    """
    restore this iterator
    params:
      ckpt_path : checkpoint path
    """
    with open(ckpt_path, 'rb') as f:
      ckpt = pickle.load(f)
      self.__dict__ = ckpt.__dict__

  def set_root_path(self, root):
    """
    set data root path of this data iterator
    """
    self.data_root = root 