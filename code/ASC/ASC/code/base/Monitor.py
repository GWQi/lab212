# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified:
# Email: hey_xiaoqi@163.com
# Filename: Monitor.py
# Description: This class is used to monitor the training session
# *************************************************

from abc import ABCMeta, abstractmethod

class MonitorBase(object):
  # base abstract class for monitor
  __metaclass__ = ABCMeta

  def __init__(self):
    """
    some values need to be monitored
    """
    self.accuracy = []