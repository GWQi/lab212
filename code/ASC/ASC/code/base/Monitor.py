# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: Monitor.py
# Description: This class is used to monitor the training session
# *************************************************

class Monitor(object):
  # base class for monitor

  def __init__(self):
    """
    some values need to be monitored
    """
    self.accuracy = []
    self.last_checkpoint=''


  def set_last_checkpoint(self, path):
    """
    set the last check piont path
    """
    self.last_checkpoint = path

  def get_last_checkpoint(self, path):
    """
    get the last check point path
    """
    return self.last_checkpoint

  def load(self):
    pass