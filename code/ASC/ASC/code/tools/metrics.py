# -*- coding: utf-8 -*-
# *************************************************
# Author: guwenqi
# Last modified: 2018-10-15
# Email: hey_xiaoqi@163.com
# Filename: metrics.py
# Description: this file is used for system testing
# *************************************************
import os
import sys
import numpy as np

ASC_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ASC_PROJECT_ROOT not in sys.path:
  sys.path.append(ASC_PROJECT_ROOT)

from ASC.code.base import universe

# ***********************************
# * Those is used for music/speech classification
# ************************************

def label2array_truth(label_path, resolution=0.01):
  """
  transform labels to np.array, elements in the
  array represent weather the sample is music or speech
  params:
    label_path : string, label file path
    resolution : float, time resolution in second
  return:
    array : np.ndarray, shape=(time,)
    start : float, start time of this label file
    end   : float, end time of this label file
  """
  with open(label_path) as f:
    lines = f.readlines()
    start, _, __ = lines[0].strip().split()
    end, _, __ = lines[-1].strip().split()

    start = float(start)
    end = float(end)

    num_decisions = int(end/resolution) - int(start/resolution) + 1

    array = np.ones(num_decisions, dtype=np.int32)

    for linenum, aline in enumerate(lines):
      start_, end_, label = aline.strip().split()

      if label in universe.SPEECH_MUSIC_DIC:
        array[int((float(start_)-start)/resolution) : int((float(end_)-start)/resolution)] =\
        universe.SPEECH_MUSIC_DIC.get(label)






def precision(truth_labels, pred_labels):
  """
  compute classification precision
  params:
    truth_labels : directory where store the ground truth labels
    pred_labels : directory where store the predictions
  return:
    
  """

  for root, dirlist, filelist in os.walk(truth_labels):
    for filename in filelist:
      if filename.endswith('.lab'):
        truth_label_path = os.path.join(root, filename)
        pred_label_path = truth_label_path.replace(truth_labels, pred_labels)

